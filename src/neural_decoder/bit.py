import torch 
import torch.nn as nn
from torch import Tensor
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # norm -> linear -> activation -> dropout -> linear -> dropout
            # we first norm with a layer norm
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            # we project in a higher dimension hidden_dim
            nn.GELU(),
            # we apply the GELU activation function
            nn.Dropout(dropout),
            # we apply dropout
            nn.Linear(hidden_dim, dim),
            # we project back to the original dimension dim
            nn.Dropout(dropout)
            # we apply dropout
        )

    def forward(self, x):
        return self.net(x)
        

# Utility to handle tuple input
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Sinusoidal positional embeddings (dynamic for any length)
def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, dim)

# Temporal attention mask allowing each token to attend to all tokens behind it
# and up to `look_ahead` tokens ahead
def create_temporal_mask(seq_len, look_ahead=0, device=None):
    idx = torch.arange(seq_len, device=device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= -look_ahead  # allows [i - k, ..., i + N]
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# Attention block
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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

    def forward(self, x, temporal_mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        if temporal_mask is not None:
            dots = dots.masked_fill(temporal_mask == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

# Transformer with masking support
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None):
        for attn, ffn in self.layers:
            x = attn(x, temporal_mask=mask) + x
            x = ffn(x) + x
        return self.norm(x)

# Main BiT model
class BiT(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim_ratio,
                 dim_head=64, dropout=0., look_ahead=0):
        """
        Args:
            patch_size (tuple): (time, feature) patch size.
            look_ahead (int): how many future patches each token can attend to.
        """
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

        return x