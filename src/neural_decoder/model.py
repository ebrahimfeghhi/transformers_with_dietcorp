import torch
from torch import nn

from .augmentations import GaussianSmoothing



class GRUDecoder(nn.Module):
    """GRU‑based speech decoder.

    Parameters
    ----------
    neural_dim : int
        Number of neural input channels.
    n_classes : int
        Number of output classes (excluding the CTC blank).
    hidden_dim : int
        Hidden state dimensionality of the GRU.
    layer_dim : int
        Number of stacked GRU layers.
    nDays : int
        Number of distinct recording sessions / days (used for day‑specific affine transforms).
    dropout : float
        Dropout probability within the GRU.
    input_dropout : float
        Dropout probability applied to inputs after the day‑specific transform.
    device : torch.device or str
        Device on which to place tensors.
    strideLen : int
        Stride for the unfolding operation (temporal down‑sampling).
    kernelLen : int
        Kernel length for the unfolding operation.
    gaussianSmoothWidth : float
        Standard deviation of Gaussian kernel applied along the temporal dimension.
    bidirectional : bool
        If ``True``, use a bidirectional GRU.
    max_mask_pct : float
        Maximum proportion of the sequence to mask during SpecAugment‑style masking.
    num_masks : int
        Number of temporal masks to apply per sample when training.
    linderman_lab : bool, optional (default = False)
        If ``True``, append a post‑RNN block consisting of LayerNorm → Dropout → Linear → ReLU, as
        described in Feghhi & Linderman (2024).
    """

    def __init__(
        self,
        neural_dim: int,
        n_classes: int,
        hidden_dim: int,
        layer_dim: int,
        nDays: int,
        dropout: float,
        input_dropout: float,
        device: torch.device,
        strideLen: int,
        kernelLen: int,
        gaussianSmoothWidth: float,
        bidirectional: bool,
        max_mask_pct: float,
        num_masks: int,
        linderman_lab: bool = False,
    ) -> None:
        
        super().__init__()

        # Store constructor args
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.linderman_lab = linderman_lab

        # === Input processing layers ===
        self.inputLayerNonlinearity = nn.Softsign()
        self.unfolder = nn.Unfold((self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen)
        self.gaussianSmoother = GaussianSmoothing(neural_dim, 20, self.gaussianSmoothWidth, dim=1)
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x].copy_(torch.eye(neural_dim))

        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
            inp_layer: nn.Linear = getattr(self, f"inpLayer{x}")
            inp_layer.weight.data.add_(torch.eye(neural_dim))

        self.inputDropoutLayer = nn.Dropout(p=self.input_dropout)

        # === GRU ===
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout if layer_dim > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # === Optional post‑RNN block ===
        rnn_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        if self.linderman_lab:
            
            self.post_rnn_block = nn.Sequential( #best sequence
                      nn.LayerNorm(rnn_out_dim),
                      nn.Dropout(p=self.dropout),
                      nn.Linear(rnn_out_dim, rnn_out_dim),
                      nn.SiLU(),
                      nn.LayerNorm(rnn_out_dim),
                      nn.Dropout(p=self.dropout),
                      nn.Linear(rnn_out_dim, rnn_out_dim),
                      nn.SiLU(),
                      nn.Dropout(p=self.dropout)            
            )
            
        else:
            
            self.post_rnn_block = nn.Identity()

        # === Final linear projection ===
        self.fc_decoder_out = nn.Linear(rnn_out_dim, n_classes + 1)  # +1 for CTC blank

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, neuralInput: torch.Tensor, X_len: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:
        
        
        """Parameters
        ----------
        neuralInput : torch.Tensor, shape (batch, time, neural_dim)
        X_len       : torch.Tensor, shape (batch,) – lengths before padding
        dayIdx      : torch.Tensor, shape (batch,) – index specifying the session/day of each sample
        """

        # --- Temporal Gaussian smoothing ---
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B, C, T)
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B, T, C)

        # --- SpecAugment‑style time masking (training only) ---
        if self.training and self.max_mask_pct > 0:
            neuralInput, _ = self.apply_time_masking(neuralInput, X_len)
            
        # moved input dropout before day specific linear layer, as per Linderma Lab submission 
        neuralInput = self.inputDropoutLayer(neuralInput)

        # --- Day‑specific affine transform ---
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (B, C, C)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + torch.index_select(
            self.dayBias, 0, dayIdx
        )
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # --- Temporal unfolding (stride / kernel) ---
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )
        
        # --- GRU encoding ---
        h0_dim = self.layer_dim * 2 if self.bidirectional else self.layer_dim
        h0 = torch.zeros(h0_dim, transformedNeural.size(0), self.hidden_dim, device=self.device)
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())  # (B, T', H[*2])

        # --- Optional post‑RNN refinement ---
        hid = self.post_rnn_block(hid)

        # --- Projection to token logits ---
        seq_out = self.fc_decoder_out(hid)  # (B, T', n_classes+1)
        return seq_out
    
    def compute_length(self, X_len):
        
        return  ((X_len - self.kernelLen) / self.strideLen).to(torch.int32)
    
    def apply_time_masking(self, X, X_len):
        
        """
        Fully vectorized time masking (no loops at all).
        
        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
            masked_indices: list of 1D LongTensors, each with indices of masked patches per batch
            unmasked_indices: list of 1D LongTensors, each with indices of unmasked patches per batch
        """
        B, P, D = X.shape
        device = X.device

        valid_lens = X_len
            
        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        # repeat_interleave works like tile, so values corresponding to the same batch are next to each other
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)            # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)      # (B * num_masks,)
       
        t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
            
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        
        t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()               # (B * num_masks,)

        # Build the global mask (B, P)
        arange = torch.arange(P, device=device).unsqueeze(0)       # (1, P)
        t0_exp = t0.unsqueeze(1)                                   # (B_rep, 1)
        t1_exp = (t0 + t).unsqueeze(1)                             # (B_rep, 1)
        mask_chunks = (arange >= t0_exp) & (arange < t1_exp)       # (B_rep, P)
        
        # Get index of sample in batch for each mask chunk
        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)  # (B * num_masks,)

        # Now scatter all the masks into the full mask (B, P)
        patch_idx = mask_chunks.nonzero(as_tuple=False)  # (N_masked, 2)
        b_indices = batch_idx[patch_idx[:, 0]]           # (N_masked,)
        p_indices = patch_idx[:, 1]                      # (N_masked,)

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True
        
        # mask: (B, P) boolean, True for masked
        #B, P = mask.shape
        # Apply the mask
        X_masked = X.clone()
        X_masked[mask] = 0

        return X_masked, mask