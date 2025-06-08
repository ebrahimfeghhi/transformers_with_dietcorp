import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


# code from Icefall package
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

def forward_ctc(
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = encoder_out.log_softmax(2) # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2).cpu(),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="mean",
        )
        return ctc_loss
    
    

def forward_cr_ctc(
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        ctc_output = encoder_out.log_softmax(2)  # (2 * N, T, C)
        
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )

        # Compute consistency regularization loss
        exchanged_targets = ctc_output.detach().chunk(2, dim=0)
        exchanged_targets = torch.cat(
            [exchanged_targets[1], exchanged_targets[0]], dim=0
        )  # exchange: [x1, x2] -> [x2, x1]
        
        cr_loss = nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(encoder_out_lens, max_len=cr_loss.shape[1]).unsqueeze(-1)
        
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()
      
        return ctc_loss, cr_loss


def memo_loss_from_logits(
    logits_aug: Tensor,
    adjusted_len: int,
    blank_id: Optional[int] = 0,
    T: float = 1
) -> Tensor:
  
    """
    Computes negative entropy loss from augmented logits.

    Parameters
    ----------
    logits_aug : Tensor
        Logits from multiple augmentations. Shape: [n_aug, T, D]
    adjusted_len : int
        If provided, truncate to this length along time dimension.
    blank_id : Optional[int]
        If not None, filter out time steps where the most likely
        token is the blank_id.
    T : float
        Temperature for softmax. 
        
    Returns
    -------
    loss : Tensor
        Scalar loss tensor (requires grad).
    """
    
    logits_aug /= T # temperature scaling
    probs_aug = torch.nn.functional.softmax(logits_aug, dim=-1)   # [n_aug, T, D]
    marginal_probs = probs_aug.mean(dim=0)                        # [T, D]
    marginal_probs = marginal_probs[:adjusted_len] # [T', D], where T' <= T

    if blank_id is not None:
        max_indices = marginal_probs.argmax(dim=1)
        marginal_probs = marginal_probs[max_indices != blank_id]

    loss = - (marginal_probs * marginal_probs.log()).sum(dim=-1).mean()
    return loss