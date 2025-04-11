import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .bit import Transformer, HybridSpatiotemporalPosEmb
from .dataset import pad_to_multiple
from torcheval.metrics import R2Score
from .augmentations import GaussianSmoothing
from .dataset import sliding_chunks
from einops.layers.torch import Rearrange


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        encoder_dim,
        day_specific,
        day_specific_tokens, 
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64, 
        gaussianSmoothWidth=2.0, 
        nDays=24
    ):
        super().__init__()
        # Ensure the masking ratio is valid
        assert 0 < masking_ratio < 1, 'masking ratio must be between 0 and 1'
        self.masking_ratio = masking_ratio

        # Save the encoder (a Vision Transformer to be trained)
        self.encoder = encoder
        
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.day_specific = day_specific
        
        self.gaussianSmoother = GaussianSmoothing(
            self.encoder.num_features, 20, self.gaussianSmoothWidth, dim=1
        )
        
        num_patches = self.encoder.num_patches
        encoder_dim = self.encoder.dim
        self.nDays = nDays
        
        if self.day_specific:
            self.dayWeights = torch.nn.Parameter(torch.randn(nDays, decoder_dim,
                                                            decoder_dim))
            self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, decoder_dim))
        
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
        
        self.day_specific_tokens = day_specific_tokens
        
        if self.day_specific_tokens:
            
            self.day_specific_patches = nn.Parameter(torch.randn(1, nDays, decoder_dim)) 
            self.decoder_pos_emb_dayspecific = nn.Embedding(nDays, decoder_dim)
            
    
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
            
            
        # Linear layer to reconstruct pixel values from decoder outputs
        self.to_neural = nn.Linear(decoder_dim, pixel_values_per_patch)
        
    def forward(self, neuralInput, dayIdx):
        
        device = neuralInput.device
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        #transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # Convert the input image into patches
        neuralInput = torch.unsqueeze(neuralInput, axis=1)
        patches = self.to_patch(neuralInput)  # Shape: (batch_size, num_patches, patch_size)
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
        
        #if self.day_specific_tokens:
            # batch size x nDays x patch Dim
        #    day_specific_patches = repeat(self.day_specific_patches, '1 n d -> b n d', b=B, n=self.nDays)
        #    day_specific_patches = day_specific_patches[torch.arange(B), dayIdx] + self.decoder_pos_emb_dayspecific(dayIdx)
        #    day_specific_patches = day_specific_patches.unsqueeze(1)
        #    decoder_sequence = torch.cat((decoder_sequence, day_specific_patches), dim=1)
        
        
        # Decode the full sequence
        decoded_tokens = self.decoder(decoder_sequence)

        # Extract the decoded tokens corresponding to the masked patches
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]

        # Reconstruct the pixel values from the masked decoded tokens
        pred_neural_values = self.to_neural(masked_decoded_tokens)
        
        # apply day layer
        #if self.day_specific:
        #    dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        #    
        #    pred_neural_values = torch.einsum(
        #        "btd,bdk->btk", pred_neural_values, dayWeights
        #    ) + torch.index_select(self.dayBias, 0, dayIdx)
            
        # Compute the reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(pred_neural_values, masked_patches)
        
        metric = R2Score()

        input_r2 = pred_neural_values.view(-1, pred_neural_values.shape[-1])
        target_r2 = masked_patches.view(-1, masked_patches.shape[-1])

        metric.update(input_r2, target_r2)
        acc = metric.compute()
        
        return recon_loss, acc
    
    
class MAE_EncoderOnly(nn.Module):
    
    def __init__(self, mae_model):
        
        super().__init__()
        
        self.encoder = mae_model.encoder
        self.to_patch = Rearrange('b t (h p1) (w p2) -> b t (h w) (p1 p2)', p1=32, p2=2)
        self.patch_to_emb = mae_model.patch_to_emb
        self.pos_embedding = mae_model.encoder.pos_embedding
        self.gaussianSmoother = mae_model.gaussianSmoother

    def forward(self, x):
    
        # Convert to patches
        patches = self.to_patch(x)
        tokens = self.patch_to_emb(patches)
        
        # Add position embeddings
        pos_emb = self.pos_embedding.to(x.device, dtype=tokens.dtype)
        tokens = tokens + pos_emb
        B, T, num_patch, patch_dim = tokens.shape
        # Pass through encoder transformer
        encoded = self.encoder.transformer(tokens.reshape(B*T, num_patch, patch_dim))
        
        return encoded.reshape(B,T,num_patch,patch_dim)
    
class GRU_MAE(nn.Module):
    
    def __init__(self, 
                 mae_encoder, 
                 n_classes=40,
                 dropout=0.4, 
                 layer_dim=5,
                 hidden_dim=1024,
                 kernelLen = 32,
                 strideLen = 4,
                 neural_dim=256,
                 whiteNoiseSD=0.2,
                 constantOffsetSD=0.2,
                 nDays=24) -> None:
        
        
        super().__init__()
        self.encoder = mae_encoder
        self.kernelLen = kernelLen
        self.stride=strideLen
        self.hidden_dim = hidden_dim
        self.bidirectional = False
        self.dropout = dropout
        self.whiteNoiseSD = whiteNoiseSD
        self.constantOffsetSD = constantOffsetSD
        
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim,
                                                        neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        
        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
                
        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1) 


    def forward(self, x, dayIdx):
        
        
        device = x.device
        
        # Apply Gaussian smoothing
        x = torch.permute(x, (0, 2, 1))
        x = self.encoder.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
        
        
        x = sliding_chunks(x, chunk_size=self.kernelLen, stride=self.stride)
                
        X = self.encoder(x)
        
        breakpoint()
        
        
           
         # Noise augmentation is faster on GPU
        if self.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=device) * self.whiteNoiseSD

        # add a constant offset to each patch 
        if self.constantOffsetSD > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2], 1], device=device)
                * self.constantOffsetSD
            )
     
            
        # apply day layer
        
        
        
        transformedNeural = torch.einsum("btpd,bpdk->btpk", X, dayWeights)
        
        breakpoint()
        
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", X, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
        
        
        hid, _ = self.gru_decoder(transformedNeural)
        
        seq_out = self.fc_decoder_out(hid)
        
        return seq_out

'''
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
'''
        