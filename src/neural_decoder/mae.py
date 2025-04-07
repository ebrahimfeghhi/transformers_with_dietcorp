import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from bit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
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

        # Extract the number of patches and the encoder's dimensionality from the positional embeddings
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

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
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        # Linear layer to reconstruct pixel values from decoder outputs
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # Convert the input image into patches
        patches = self.to_patch(img)  # Shape: (batch_size, num_patches, patch_size)
        batch_size, num_patches, *_ = patches.shape
 
        # Embed the patches using the encoder's patch embedding layers
        tokens = self.patch_to_emb(patches)  # Shape: (batch_size, num_patches, encoder_dim)

      
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
        encoded_tokens = self.encoder.transformer(tokens)

        # Map encoded tokens to decoder dimensions if necessary
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # Add positional embeddings to the decoder tokens of unmasked patches
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # Create mask tokens for the masked patches and add positional embeddings
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch_size, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

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
        pred_pixel_values = self.to_pixels(masked_decoded_tokens)

        # Compute the reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss