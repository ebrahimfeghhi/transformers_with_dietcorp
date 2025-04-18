import torch 
import torch.nn as nn
from torch import Tensor
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .augmentations import GaussianSmoothing
from .dataset import pad_to_multiple
from .bit import Transformer, create_temporal_mask, pair
from torchmetrics.regression import R2Score

class BiT_with_adapt(nn.Module):
    
    def __init__(self, *, patch_size, dim, depth_phoneme, depth_reconstruct, heads, mlp_dim_ratio,
                 dim_head, dropout, dropout_reconstruct, input_dropout, look_ahead, nDays, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks):
   
        super().__init__()

        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = dim
        self.look_ahead = look_ahead  
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks

            
        self.patch_dim = patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=patch_height, p2=patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Patch embedding split from encoder
        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])
        
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, patch_width, patch_width))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, patch_width))
        self.mask_token = nn.Parameter(torch.randn(self.patch_dim))
        
        self.gaussianSmoother = GaussianSmoothing(
            patch_width, 20, self.gaussianSmoothWidth, dim=1
        )
                
    
        self.dropout = nn.Dropout(input_dropout)
        
        self.calibration_layer = Transformer(dim, 1, heads, dim_head, mlp_dim_ratio, 
                                             dropout_reconstruct, use_relative_bias=self.T5_style_pos)
        
        self.phoneme_decoder = Transformer(dim, depth_phoneme, heads, dim_head, mlp_dim_ratio, 
                                       dropout, use_relative_bias=self.T5_style_pos)
        
        self.reconstruction_decoder = Transformer(dim, depth_reconstruct, heads, dim_head, mlp_dim_ratio, 
                                       dropout_reconstruct, use_relative_bias=self.T5_style_pos)
        
        
       
        self.projection = nn.Linear(dim, nClasses+1)
        
        

    def forward(self, neuralInput, X_len, dayIdx):
        """
        Args:
            neuralInout: Tensor of shape (B, 1, T, F)
            dayIdx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, dim)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        neuralInput = neuralInput.unsqueeze(1)
        
        if self.training and self.max_mask_pct > 0:
            patches = self.to_patch(neuralInput)
            x, mask = self.apply_specaugment_mask(patches, X_len)
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)
            
            
        # if in mae mode, input has already been patched. 
        b, seq_len, _ = x.shape
        
        
        temporal_mask = create_temporal_mask(seq_len, look_ahead=self.look_ahead, device=x.device)
        
        # apply shared layer
        x = self.calibration_layer(x, mask=temporal_mask)
        
        reconstruction_x = self.reconstruction_decoder(x, mask=temporal_mask)
        
        # Compute loss and R²
        # only apply this this loss to patches that were masked
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)
        recon_loss = F.mse_loss(reconstruction_x[mask_expanded], patches[mask_expanded])
        metric = R2Score()
        metric.update(reconstruction_x[mask_expanded].view(-1, reconstruction_x[mask_expanded].shape[-1]),
                      patches[mask_expanded].view(-1, patches[mask_expanded].shape[-1]))
        
        acc = metric.compute()
        
        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        x = torch.einsum(
            "btd,bdk->btk", x, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        x = self.inputLayerNonlinearity(x)
    
        # apply input level dropout. 
        x = self.dropout(x)

        # Apply transformer with temporal masking
        x = self.phoneme_decoder(x, mask=temporal_mask)
        
        # get phoneme logits
        out = self.projection(x)

        return recon_loss, acc, out
    
    def compute_length(self, X_len):
        
        return (X_len / self.patch_height).to(torch.int32)
    
    def apply_specaugment_mask(self, X, X_len, constant_mask=False, mask_range=[]):
        """
        Fully vectorized SpecAugment-style time masking (no loops at all).
        
        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints
            constant_mask_lengths: if True, make the mask lengths the same across all batches

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
            masked_indices: list of 1D LongTensors, each with indices of masked patches per batch
            unmasked_indices: list of 1D LongTensors, each with indices of unmasked patches per batch
        """
        B, P, D = X.shape
        device = X.device

        if constant_mask:
            # get valid len of smallest trial in batch and repeat for all batches. 
            valid_lens = torch.min((X_len // self.patch_height).to(device)).repeat(B)
        else:
            valid_lens = (X_len // self.patch_height).to(device)
            
        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        # repeat_interleave works like tile, so values corresponding to the same batch are next to each other
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)            # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)      # (B * num_masks,)

        if constant_mask:
            # select the same t for every batch. 
            t = (torch.rand(self.num_masks, device=device).repeat(B) * (max_mask_lens_rep + 1).float()).floor().long().clamp(min=1)  # (B * num_masks,)
        else:
            t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
            
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        
        if constant_mask:
            t0 = (torch.rand(self.num_masks, device=device).repeat(B) * max_start.float()).floor().long()               # (B * num_masks,)
        else:
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

        # Number of masked patches per batch (assumed same for all batches)
        if constant_mask:
            N = mask.sum(dim=1)[0].item()
            U = P - N  # Number of unmasked per batch
                            
            masked_indices = mask.nonzero(as_tuple=False)  # (B * N, 2) — rows: [batch_idx, patch_idx]
            masked_indices = masked_indices[:, 1].reshape(B, N)
            masked_indices = torch.sort(masked_indices, dim=-1).values  # sort within batch
        
            unmasked = ~mask  # invert the mask
            unmasked_indices = unmasked.nonzero(as_tuple=False)[:, 1].reshape(B, U)
            unmasked_indices = torch.sort(unmasked_indices, dim=-1).values
        
            return masked_indices, unmasked_indices
        
        # Apply the mask
        X_masked = X.clone()
        X_masked[mask] = self.mask_token

        return X_masked, mask
    
    
    
    