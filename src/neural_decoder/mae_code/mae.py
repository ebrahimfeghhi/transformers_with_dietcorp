import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .bit import Transformer
from .dataset import pad_to_multiple
from torcheval.metrics import R2Score
from .augmentations import GaussianSmoothing
from .dataset import sliding_chunks
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchmetrics.regression import R2Score
from .bit import create_temporal_mask, get_sinusoidal_pos_emb

'''
Code adapted from Fracois Porcher: https://github.com/FrancoisPorcher/vit-pytorch
'''


class MAE_with_mask(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        encoder_dim,
        whiteNoiseSD,
        constantOffsetSD,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        gaussianSmoothWidth
    ):
        
        super().__init__()

        self.encoder = encoder
        self.decoder_dim = decoder_dim
        self.whiteNoiseSD = whiteNoiseSD
        self.constantOffsetSD = constantOffsetSD

        # Gaussian smoothing
        self.gaussianSmoother = GaussianSmoothing(
            self.encoder.patch_width, 20, gaussianSmoothWidth, dim=1
        )
        
        self.gaussianSmoother_whiteNoise = GaussianSmoothing(
            decoder_dim, 20, gaussianSmoothWidth, dim=1
        )

        # Patch embedding split from encoder
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # Linear map encoder → decoder dim if needed
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim_ratio=4
        )
        
        self.to_neural = nn.Linear(decoder_dim, pixel_values_per_patch)
        
    def forward(self, neuralInput, X_len, dayIdx):
        device = neuralInput.device

        neuralInput = pad_to_multiple(neuralInput, multiple=self.encoder.patch_height)
        
        # Apply Gaussian smoothing to denoise
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Patchify input: (B, 1, T, F) → (B, num_patches, patch_size)
        neuralInput = neuralInput.unsqueeze(1)
        patches = self.to_patch(neuralInput)
        
        B, num_patches, _ = patches.shape

        # Embed patches
        tokens = self.patch_to_emb(patches)

        masked_indices, unmasked_indices = self.encoder.apply_time_mask(patches, X_len, constant_mask=True)
        
        # Create and embed mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=B, n=masked_indices.shape[1])
        batch_range = torch.arange(B, device=device)[:, None]
        
        # adding masked tokens here just so the encoder can learn a mask for when it is trained on the phoneme task. 
        encoder_sequence = torch.zeros_like(tokens, device=device)
        encoder_sequence[batch_range, unmasked_indices] = tokens[batch_range, unmasked_indices]
        encoder_sequence[batch_range, masked_indices] = mask_tokens
        
        masked_patches = patches[batch_range, masked_indices]
        
        # Encode the unmasked tokens
        seq_len = encoder_sequence.shape[1]
        temporal_mask = create_temporal_mask(seq_len=seq_len, 
                        look_ahead=self.encoder.look_ahead, device=device)
        
        decoder_sequence = self.encoder.transformer(encoder_sequence, mask=temporal_mask)

        # Decode
        seq_len = decoder_sequence.shape[1]
        decoder_mask = create_temporal_mask(seq_len=seq_len, look_ahead=self.encoder.look_ahead, device=device)

        # Apply masked decoder
        decoded_tokens = self.decoder(decoder_sequence, mask=decoder_mask)
          
        # Extract outputs for masked positions
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]
        pred_neural_values = self.to_neural(masked_decoded_tokens)

        # Compute loss and R²
        recon_loss = F.mse_loss(pred_neural_values, masked_patches)
        metric = R2Score()
        metric.update(pred_neural_values.view(-1, pred_neural_values.shape[-1]),
                      masked_patches.view(-1, masked_patches.shape[-1]))
        
        acc = metric.compute()
        
        return recon_loss, acc
    
    def compute_length(self, X_len):
        
        return self.encoder.compute_length(X_len)


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        phoneme_decoder, 
        decoder_dim,
        encoder_dim,
        whiteNoiseSD,
        constantOffsetSD,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        gaussianSmoothWidth
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder_dim = decoder_dim
        self.whiteNoiseSD = whiteNoiseSD
        self.constantOffsetSD = constantOffsetSD

        # Gaussian smoothing
        self.gaussianSmoother = GaussianSmoothing(
            self.encoder.patch_width, 20, gaussianSmoothWidth, dim=1
        )
        
        self.gaussianSmoother_whiteNoise = GaussianSmoothing(
            decoder_dim, 20, gaussianSmoothWidth, dim=1
        )

        # Patch embedding split from encoder
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # Linear map encoder → decoder dim if needed
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim_ratio=4
        )
        
        self.to_neural = nn.Linear(decoder_dim, pixel_values_per_patch)
        
        self.phoneme_decoder = phoneme_decoder

    def forward(self, neuralInput, X_len, dayIdx):
        device = neuralInput.device

        neuralInput = pad_to_multiple(neuralInput, multiple=self.encoder.patch_height)
        
        # Apply Gaussian smoothing to denoise
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Patchify input: (B, 1, T, F) → (B, num_patches, patch_size)
        neuralInput = neuralInput.unsqueeze(1)
        patches = self.to_patch(neuralInput)
        
        B, num_patches, _ = patches.shape

        # Embed patches
        tokens = self.patch_to_emb(patches)

        masked_indices, unmasked_indices = self.encoder.apply_specaugment_mask(patches, X_len, constant_mask=True)
        
        batch_range = torch.arange(B, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices] # B x P x D, B x P 
        masked_patches = patches[batch_range, masked_indices]
        
        
        # Encode the unmasked tokens
        seq_len = unmasked_tokens.shape[1]
        temporal_mask = create_temporal_mask(seq_len=seq_len, 
                        look_ahead=self.encoder.look_ahead, device=device)
        
        encoded_tokens = self.encoder.transformer(unmasked_tokens, mask=temporal_mask, original_indices=unmasked_indices)

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        unmasked_decoder_tokens = decoder_tokens
    
        # Create and embed mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=B, n=masked_indices.shape[1])

        # Initialize full decoder input
        decoder_sequence = torch.zeros(B, num_patches, self.decoder_dim, device=device)
        decoder_sequence[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_sequence[batch_range, masked_indices] = mask_tokens

        # Decode
        seq_len = decoder_sequence.shape[1]
        decoder_mask = create_temporal_mask(seq_len=seq_len, look_ahead=self.encoder.look_ahead, device=device)

        # Apply masked decoder
        decoded_tokens = self.decoder(decoder_sequence, mask=decoder_mask)
        
          
        # Extract outputs for masked positions
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]
        pred_neural_values = self.to_neural(masked_decoded_tokens)

        # Compute loss and R²
        recon_loss = F.mse_loss(pred_neural_values, masked_patches)
        metric = R2Score()
        metric.update(pred_neural_values.view(-1, pred_neural_values.shape[-1]),
                      masked_patches.view(-1, masked_patches.shape[-1]))
        
        acc = metric.compute()
        
        if self.phoneme_decoder is not None:
            
            if self.constantOffsetSD > 0:
                
                constantOffset = (
                    torch.randn([decoded_tokens.shape[0], 1, decoded_tokens.shape[2]], device=device)
                    * self.constantOffsetSD
                )
                decoded_tokens[batch_range, unmasked_indices] += constantOffset
                    
            if self.whiteNoiseSD > 0:
                whiteNoise = torch.randn(decoded_tokens[batch_range, unmasked_indices].shape, 
                                        device=device) * self.whiteNoiseSD
                whiteNoise = torch.permute(whiteNoise, (0,2,1))
                blurred_whiteNoise = self.gaussianSmoother_whiteNoise(whiteNoise)
                blurred_whiteNoise = torch.permute(whiteNoise, (0,2,1))
                decoded_tokens[batch_range, unmasked_indices] += blurred_whiteNoise
            
            phoneme_logits = self.phoneme_decoder(decoded_tokens, X_len, dayIdx)
            
        
            return recon_loss, acc, phoneme_logits
        
        return recon_loss, acc
    
    
    def compute_length(self, X_len):
        
        return self.encoder.compute_length(X_len)