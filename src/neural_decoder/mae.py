import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .bit import Transformer, HybridSpatiotemporalPosEmb
from .dataset import pad_to_multiple

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        encoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()
        # Ensure the masking ratio is valid
        assert 0 < masking_ratio < 1, 'masking ratio must be between 0 and 1'
        self.masking_ratio = masking_ratio

        # Save the encoder (a Vision Transformer to be trained)
        self.encoder = encoder
        
        num_patches = self.encoder.num_patches
        encoder_dim = self.encoder.dim

        # Separate the patch embedding layers from the encoder
        # The first layer converts the image into patches
        self.to_patch = encoder.to_patch_embedding[0]
        # The remaining layers embed the patches
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        # Determine the dimensionality of the pixel values per patch
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # Set up decoder parameters
        self.decoder_dim = decoder_dim
        # Map encoder dimensions to decoder dimensions if they differ
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        # Learnable mask token for masked patches
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # Define the decoder transformer
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim_ratio=4
            )
        # Positional embeddings for the decoder tokens
        #self.decoder_pos_emb = HybridSpatiotemporalPosEmb(num_space=16, max_time=10000, embedding_dim=decoder_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        
        # Linear layer to reconstruct pixel values from decoder outputs
        self.to_neural = nn.Linear(decoder_dim, pixel_values_per_patch)
        
    def create_temporal_attention_mask(self, num_patches, device, patches_per_timestep=16, N=20):
        
        mask = torch.full((num_patches, num_patches), 0)

        timesteps = num_patches // patches_per_timestep
        
        for t_q in range(timesteps):  # time index of query
            for dt in range(N + 1):  # how far back to look
                t_k = t_q - dt  # key timestep
                if t_k < 0:
                    continue
                q_start = t_q * patches_per_timestep
                q_end = (t_q + 1) * patches_per_timestep
                k_start = t_k * patches_per_timestep
                k_end = (t_k + 1) * patches_per_timestep
                # allow attention: set to 0 (non-masked)
                mask[q_start:q_end, k_start:k_end] = 1

        return mask.to(device)  # shape: [Num Patches, Num Patches]

    def forward(self, img):
        device = img.device

        # Convert the input image into patches
        img = pad_to_multiple(img, multiple=4)
        img = torch.unsqueeze(img, axis=1)
        patches = self.to_patch(img)  # Shape: (batch_size, num_patches, patch_size)
        batch_size, num_patches, *_ = patches.shape
 
        # Embed the patches using the encoder's patch embedding layers
   
        tokens = self.patch_to_emb(patches)  # Shape: (batch_size, num_patches, encoder_dim)
  
        #pos_embeddings = self.encoder.pos_embedding(tokens)
        #tokens += pos_embeddings
        tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        # Determine the number of patches to mask
        num_masked = int(self.masking_ratio * num_patches)

        # Generate random indices for masking
        rand_indices = torch.rand(batch_size, num_patches, device=device).argsort(dim=-1)
        masked_indices = rand_indices[:, :num_masked]
        unmasked_indices = rand_indices[:, num_masked:]

        # Select the tokens corresponding to unmasked patches
        batch_range = torch.arange(batch_size, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Select the original patches that are masked (for reconstruction loss)
        masked_patches = patches[batch_range, masked_indices]

        # Encode the unmasked tokens using the encoder's transformer
        #temporal_mask = self.create_temporal_attention_mask(num_patches, device)
        # Assume temporal_mask has shape (num_patches, num_patches)
        #temporal_mask = temporal_mask.unsqueeze(0)  # Now shape is (1, num_patches, num_patches)
        #temporal_mask = temporal_mask.repeat(batch_size, 1, 1)  # Now shape is (batch_size, num_patches, num_patches)
        
        B, T = unmasked_indices.shape

        # First, gather the rows (dim=1)
        # This will give you shape: (B, T, N)
        #masked_rows = torch.gather(temporal_mask, dim=1, index=unmasked_indices.unsqueeze(-1).expand(-1, -1, temporal_mask.size(2)))

        # Now gather the columns (dim=2)
        # This will give you shape: (B, T, T)
        #masked_submatrix = torch.gather(masked_rows, dim=2, index=unmasked_indices.unsqueeze(1).expand(-1, T, -1))
        encoded_tokens = self.encoder.transformer(tokens)

        # Map encoded tokens to decoder dimensions if necessary
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        
        # Add positional embeddings to the decoder tokens of unmasked patches
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        
        # Add positional embeddings to the decoder tokens of unmasked patches
        #decoder_pos_embeddings = self.decoder_pos_emb(patches)
        # Step 1: Expand unmasked_indices to shape (8, 450, 64)
        #expanded_indices = unmasked_indices.unsqueeze(-1).expand(-1, -1, 64)
        # Step 2: Use torch.gather to pull the right positions per batch
        #decoder_selected = torch.gather(decoder_pos_embeddings, dim=1, index=expanded_indices)
        
        #unmasked_decoder_tokens = decoder_tokens + decoder_selected

        # Create mask tokens for the masked patches and add positional embeddings
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch_size, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        
        # Step 1: Expand unmasked_indices to shape (8, 450, 64)
        #expanded_indices = masked_indices.unsqueeze(-1).expand(-1, -1, 64)
        # Step 2: Use torch.gather to pull the right positions per batch
        #decoder_selected = torch.gather(decoder_pos_embeddings, dim=1, index=expanded_indices)
        #mask_tokens = mask_tokens + decoder_selected

        # Initialize the full sequence of decoder tokens
        decoder_sequence = torch.zeros(
            batch_size, num_patches, self.decoder_dim, device=device
        )
        # Place unmasked decoder tokens and mask tokens in their original positions
        decoder_sequence[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_sequence[batch_range, masked_indices] = mask_tokens

        # Decode the full sequence
        decoded_tokens = self.decoder(decoder_sequence)

        # Extract the decoded tokens corresponding to the masked patches
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]

        # Reconstruct the pixel values from the masked decoded tokens
        pred_pixel_values = self.to_neural(masked_decoded_tokens)

        # Compute the reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        
        return recon_loss