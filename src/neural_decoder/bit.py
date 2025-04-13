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
    
# Main BiT model
class BiT_Phoneme(nn.Module):
    
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim_ratio,
                 dim_head=64, dropout=0., look_ahead=0, nDays=24, gaussianSmoothWidth=2.0, 
                 nClasses=40, T5_style_pos=True):
   
        super().__init__()

        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = dim
        self.look_ahead = look_ahead  # new
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.T5_style_pos = T5_style_pos

        patch_dim = patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.dropout = nn.Dropout(dropout)
        if self.T5_style_pos == False:
            self.register_buffer('pos_embedding', None, persistent=False)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, dropout, use_relative_bias=self.T5_style_pos)
        
        
        self.gaussianSmoother = GaussianSmoothing(
            patch_width, 20, self.gaussianSmoothWidth, dim=1
        )
        
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, patch_width, patch_width))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, patch_width))
        
        self.projection = nn.Linear(dim, nClasses+1)

    def forward(self, neuralInput, dayIdx):
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

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        
        neuralInput = neuralInput.unsqueeze(1)
        x = self.to_patch_embedding(neuralInput)  # (B, T_patches, dim)
        b, seq_len, _ = x.shape

        # Positional encoding
        if self.T5_style_pos:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)

        x = self.dropout(x)

        # Create temporal mask
        temporal_mask = create_temporal_mask(seq_len, look_ahead=self.look_ahead, device=x.device)

        # Apply transformer with temporal masking
        x = self.transformer(x, mask=temporal_mask)
        
        out = self.projection(x)

        return out
    
    def compute_length(self, X_len):
        
        return (X_len / self.patch_height).to(torch.int32)