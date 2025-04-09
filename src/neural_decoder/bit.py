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
        
        
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # Calculate the total inner dimension based on the number of attention heads and the dimension per head
        
        # Determine if a final projection layer is needed based on the number of heads and dimension per head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # Store the number of attention heads
        self.scale = dim_head ** -0.5  # Scaling factor for the attention scores (inverse of sqrt(dim_head))

        self.norm = nn.LayerNorm(dim)  # Layer normalization to stabilize training and improve convergence

        self.attend = nn.Softmax(dim=-1)  # Softmax layer to compute attention weights (along the last dimension)
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization during training
        

        # Linear layer to project input tensor into queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Conditional projection layer after attention, to project back to the original dimension if required
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # Linear layer to project concatenated head outputs back to the original input dimension
            nn.Dropout(dropout)         # Dropout layer for regularization
        ) if project_out else nn.Identity()  # Use Identity (no change) if no projection is needed


    def forward(self, x, temporal_mask=None):
        x = self.norm(x)  # Apply normalization to the input tensor

        # Apply the linear layer to get queries, keys, and values, then split into 3 separate tensors
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Chunk the tensor into 3 parts along the last dimension: (query, key, value)

        # Reshape each chunk tensor from (batch_size, num_patches, inner_dim) to (batch_size, num_heads, num_patches, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Calculate dot products between queries and keys, scale by the inverse square root of dimension
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Shape: (batch_size, num_heads, num_patches, num_patches)

        # If causal is True, generate the causal mask
        # If a custom mask is provided, apply it
        #if temporal_mask is not None:
        #    mask = temporal_mask.unsqueeze(1) 
        #    dots = dots.masked_fill(mask == 0, float('-inf'))  # Apply custom mask by setting masked positions to -inf

        # Apply softmax to get attention weights
        attn = self.attend(dots)  # Shape: (batch_size, num_heads, num_patches, num_patches)
        
        attn = self.dropout(attn)

        # Multiply attention weights by values to get the output
        out = torch.matmul(attn, v)  # Shape: (batch_size, num_heads, num_patches, dim_head)

        # Rearrange the output tensor to (batch_size, num_patches, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads dimension with the output dimension

        # Project the output back to the original input dimension if needed
        out = self.to_out(out)  # Shape: (batch_size, num_patches, dim)

        return out  # Return the final output tensor
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, dropout = 0.):
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
            x = attn(x, mask) + x
            x = ffn(x) + x
        return self.norm(x)
    
def pair(t):
    """
    Converts a single value into a tuple of two values.
    If t is already a tuple, it is returned as is.
    
    Args:
        t: A single value or a tuple.
    
    Returns:
        A tuple where both elements are t if t is not a tuple.
    """
    return t if isinstance(t, tuple) else (t, t)

class HybridSpatiotemporalPosEmb(nn.Module):
    def __init__(self, num_space, max_time, embedding_dim, temporal_only=False):
        """
        num_space: number of spatial positions (N)
        max_time: number of time steps (T)
        embedding_dim: size of each positional embedding (must be even)
        temporal_only: if True, don't add spatial embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.N = num_space
        self.T = max_time
        self.temporal_only = temporal_only

        assert embedding_dim % 2 == 0, "Embedding dimension must be even for sin/cos"

        # Learnable spatial embeddings
        self.space_embedding = nn.Parameter(torch.randn(num_space, embedding_dim))

        # Fixed sinusoidal temporal embeddings
        self.register_buffer("time_embedding", self._build_sin_cos_embedding(max_time, embedding_dim))

    def _build_sin_cos_embedding(self, length, dim):
        """
        Generate fixed sinusoidal embeddings of shape (length, dim)
        """
        position = torch.arange(1, length + 1).unsqueeze(1).float()  # (length, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # (dim/2,)
        sinusoid = torch.zeros(length, dim)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)
        return sinusoid  # (length, dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_patches, embedding_dim)
        Assumes patches are ordered as:
            [t0_p0, t0_p1, ..., t0_pN-1, t1_p0, ..., tT-1_pN-1]
        """
        batch_size, num_patches, _ = x.size()
        T = num_patches // self.N

        # Compute spatial and temporal indices
        spatial_idx = torch.arange(num_patches, device=x.device) % self.N
        temporal_idx = torch.arange(num_patches, device=x.device) // self.N

        # Lookup embeddings
        pos_space_embedding = self.space_embedding[spatial_idx]     # (num_patches, embedding_dim)
        pos_time_embedding = self.time_embedding[temporal_idx]      # (num_patches, embedding_dim)

        # Combine and expand to batch
        if self.temporal_only:
            pos_embedding = pos_time_embedding
        else:
            pos_embedding = pos_space_embedding + pos_time_embedding
            
        pos_embedding = pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        return pos_embedding

    
class BiT(nn.Module):
    def __init__(self, *, trial_size, patch_size, dim, depth, 
                 heads, mlp_dim_ratio, dim_head=64, dropout=0.):
        """
        Initializes a Brain Transformer (BiT) model.
        
        Args:
            trial_size (tuple): time x features
            patch_size (int or tuple): Size of each patch (height, width).
            dim (int): Dimension of the embedding space.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the feedforward network.
            dim_head (int): Dimension of each attention head.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Convert image size and patch size to tuples if they are single values
        trial_length, num_features = pair(trial_size)
        patch_height, patch_width = pair(patch_size)
        
        self.trial_length = trial_length
        
          # Ensure that the image dimensions are divisible by the patch size
        assert trial_length % patch_height == 0 and num_features % patch_width == 0, 'Trial dimensions must be divisible by the patch size.'

        # Calculate the number of patches and the dimension of each patch
        num_patches = (trial_length // patch_height) * (num_features // patch_width)
        
        self.num_patches = num_patches
        self.dim = dim

        # Calculate the number of patches and the dimension of each patch
        patch_dim = patch_height * patch_width

        # Define the patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            # Rearrange the input tensor to (batch_size, num_patches, patch_dim)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # Normalize each patch
            nn.Linear(patch_dim, dim),  # Project patches to embedding dimension
            nn.LayerNorm(dim)  # Normalize the embedding
        )

        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.pos_embedding = HybridSpatiotemporalPosEmb(num_space=16, max_time=10000, embedding_dim=dim)
        
        # Define the transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, dropout)
 
    def forward(self, neural_data):
        """
        Forward pass through the Brain Transformer model.
        
        Args:
            img (Tensor): Input image tensor of shape (batch_size, channels, height, width).
        
        Returns:
            dict: A dictionary containing the class token, feature map, and classification result.
        """
        # Convert image to patch embeddings
        
        x = self.to_patch_embedding(neural_data) # Shape: (batch_size, num_patches, dim)
        b, n, _ = x.shape  # Get batch size, number of patches, and embedding dimension

        # Add positional embeddings to the input
        #pos_embeddings = self.pos_embedding(x)
        #x += pos_embeddings
        x += self.pos_embedding[:, :n]
        
        # Apply dropout for regularization
        x = self.dropout(x)

        # Pass through transformer encoder
        x = self.transformer(x) # Shape: (batch_size, num_patches + 1, dim)

        # Extract class token and feature map
        #cls_token = x[:, 0]  # Extract class token
        #feature_map = x[:, 1:]  # Remaining tokens are feature map

        # Apply pooling operation: 'cls' token or mean of patches
        #pooled_output = cls_token if self.pool == 'cls' else feature_map.mean(dim=1)
        # Use the final token representation

        # Apply the identity transformation (no change to the tensor)
        #pooled_output = self.to_latent(pooled_output)

        # Apply the classification head to the pooled output
        #classification_result = self.mlp_head(pooled_output)
        #classification_result = self.mlp_head(x)
    
        # Return a dictionary with the required components
        return x