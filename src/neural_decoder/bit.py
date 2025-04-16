import torch 
import torch.nn as nn
from torch import Tensor
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .augmentations import GaussianSmoothing
from .dataset import pad_to_multiple

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def create_temporal_mask(seq_len, look_ahead=0, device=None):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = j <= i + look_ahead
    return mask.unsqueeze(0).unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., max_rel_dist=200, use_relative_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # T5-style relative position bias
        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias
        if self.use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)

    def forward(self, x, temporal_mask=None, original_indices=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        # Add relative positional bias if enabled
        if self.use_relative_bias:
            if original_indices is not None:
                rel_bias = []
                for b in range(x.size(0)):
                    orig = original_indices[b]  # shape: (num_unmasked,)
                    rel_pos = orig[:, None] - orig[None, :]  # (T, T)
                    rel_pos = rel_pos.clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
                    bias = self.rel_pos_bias(rel_pos).squeeze(-1)  # (T, T)
                    rel_bias.append(bias)
                rel_bias = torch.stack(rel_bias, dim=0).unsqueeze(1)  # (B, 1, T, T)
            else:
                seq_len = x.size(1)
                i = torch.arange(seq_len, device=x.device).unsqueeze(1)
                j = torch.arange(seq_len, device=x.device).unsqueeze(0)
                rel_pos = (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
                rel_bias = self.rel_pos_bias(rel_pos).squeeze(-1).unsqueeze(0).unsqueeze(0)
    
        dots = dots + rel_bias

        if temporal_mask is not None:
            dots = dots.masked_fill(temporal_mask == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, dropout=0., use_relative_bias=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, use_relative_bias=use_relative_bias),
                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None, original_indices=None):
        for attn, ffn in self.layers:
            x = attn(x, temporal_mask=mask, original_indices=original_indices) + x
            x = ffn(x) + x
        return self.norm(x)


# Main BiT model
class BiT(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim_ratio,
                 dim_head=64, dropout=0., look_ahead=0):

        super().__init__()

        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = dim
        self.look_ahead = look_ahead  # new

        patch_dim = patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', None, persistent=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, dropout)

    def forward(self, neural_data):
        """
        Args:
            neural_data: Tensor of shape (B, 1, T, F)
        Returns:
            Tensor: (B, num_patches, dim)
        """
        x = self.to_patch_embedding(neural_data)  # (B, T_patches, dim)
        b, seq_len, _ = x.shape

        # Positional encoding
        pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
        x = x + pos_emb.unsqueeze(0)

        x = self.dropout(x)

        # Create temporal mask
        temporal_mask = create_temporal_mask(seq_len, look_ahead=self.look_ahead, device=x.device)

        # Apply transformer with temporal masking
        x = self.transformer(x, mask=temporal_mask)
        
        breakpoint()

        return x
    
class BiT_Phoneme(nn.Module):
    
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim_ratio,
                 dim_head, dropout, input_dropout, look_ahead, nDays, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks):
   
        super().__init__()

        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = dim
        self.look_ahead = look_ahead  # new
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.patch_dim = patch_height * patch_width
        self.num_masks = num_masks

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

        self.dropout = nn.Dropout(input_dropout)
        
        if self.T5_style_pos == False:
            self.register_buffer('pos_embedding', None, persistent=False)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, 
                                       dropout, use_relative_bias=self.T5_style_pos)
        
        
        self.gaussianSmoother = GaussianSmoothing(
            patch_width, 20, self.gaussianSmoothWidth, dim=1
        )
        
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, patch_width, patch_width))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, patch_width))
        
        self.projection = nn.Linear(dim, nClasses+1)
        
        self.mask_token = nn.Parameter(torch.randn(self.patch_dim))
        

    def forward(self, neuralInput, X_len, dayIdx):
        """
        Args:
            neuralInout: Tensor of shape (B, 1, T, F)
            dayIdx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, dim)
        """
        device = neuralInput.device
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        neuralInput = neuralInput.unsqueeze(1)
                
        if self.training and self.max_mask_pct > 0:
            x = self.to_patch(neuralInput)
            x, _ = self.apply_specaugment_mask(x, X_len)
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)
            
        b, seq_len, _ = x.shape

        x = self.dropout(x)

        # Create temporal mask
        temporal_mask = create_temporal_mask(seq_len, look_ahead=self.look_ahead, device=x.device)

        # Apply transformer with temporal masking
        x = self.transformer(x, mask=temporal_mask)
        
        out = self.projection(x)

        return out
    
    def compute_length(self, X_len):
        
        return (X_len / self.patch_height).to(torch.int32)
    
    def simple_mask(self, B, num_patches, device):
        # Mask a subset of tokens
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(B, num_patches, device=device).argsort(dim=-1)
        masked_indices = rand_indices[:, :num_masked]
        unmasked_indices = rand_indices[:, num_masked:]
        # Sort only the selected indices to restore time order
        masked_indices = torch.sort(masked_indices, dim=-1).values
        unmasked_indices = torch.sort(unmasked_indices, dim=-1).values
        
        return masked_indices, unmasked_indices, num_masked
    
    def apply_specaugment_mask(self, X, X_len, constant_mask=False):
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
            t = (torch.rand(self.num_masks, device=device).repeat(B) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
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
        #N = mask.sum(dim=1)[0].item()
        #U = P - N  # Number of unmasked per batch
                        
        #masked_indices = mask.nonzero(as_tuple=False)  # (B * N, 2) — rows: [batch_idx, patch_idx]
        #masked_indices = masked_indices[:, 1].reshape(B, N)
        #masked_indices = torch.sort(masked_indices, dim=-1).values  # sort within batch
    
        #unmasked = ~mask  # invert the mask
        #unmasked_indices = unmasked.nonzero(as_tuple=False)[:, 1].reshape(B, U)
        #unmasked_indices = torch.sort(unmasked_indices, dim=-1).values
        
        # Apply the mask
        X_masked = X.clone()
        X_masked[mask] = self.mask_token

        return X_masked, mask
    
    