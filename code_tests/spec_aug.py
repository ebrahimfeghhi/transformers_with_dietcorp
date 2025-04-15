import torch
import torch


class DummySpecAug:
    def __init__(self, mask_token, patch_height=1, max_mask_pct=0.2):
        self.mask_token = mask_token
        self.patch_height = patch_height
        self.max_mask_pct = max_mask_pct

    def apply_specaugment_mask(self, X, X_len, num_masks=2):
            """
            Fully vectorized SpecAugment-style time masking (no loops at all).
            Args:
                X: (B, P, D) input tensor
                X_len: (B,) valid lengths in timepoints
            Returns:
                X_masked: (B, P, D) with masked patches
            """
            B, P, D = X.shape
            device = X.device

            valid_lens = (X_len // self.patch_height).to(device) 
            max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

            # Repeat B num_masks times to simulate multiple masks per sample
            B_rep = B * num_masks

            # Expand inputs for vectorized masking
            valid_lens_rep = valid_lens.repeat_interleave(num_masks)            # (B * num_masks,)
            max_mask_lens_rep = max_mask_lens.repeat_interleave(num_masks)      # (B * num_masks,)

            # Random mask lengths and start positions
            t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()                   # (B * num_masks,)
            max_start = (valid_lens_rep - t + 1).clamp(min=1) # don't start later than this
            t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()  # (B * num_masks,)

            # Build the global mask (B, P)
            arange = torch.arange(P, device=device).unsqueeze(0)              # (1, P)
            t0_exp = t0.unsqueeze(1)                                          # (B_rep, 1)
            t1_exp = (t0 + t).unsqueeze(1)                                    # (B_rep, 1)
            mask_chunks = (arange >= t0_exp) & (arange < t1_exp)              # (B_rep, P)

            # Get index of sample in batch for each mask chunk
            batch_idx = torch.arange(B, device=device).repeat_interleave(num_masks)  # (B * num_masks,)

            # Now scatter all the masks into the full mask (B, P)
            patch_idx = mask_chunks.nonzero(as_tuple=False)     # (N_masked, 2)
            b_indices = batch_idx[patch_idx[:, 0]]               # map each row to actual batch index
            p_indices = patch_idx[:, 1]                          # patch/time index

            mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            mask[b_indices, p_indices] = True
            breakpoint()
            # Apply the mask
            X_masked = X.clone()
            #mask_tokens = repeat(self.mask_token, 'd -> 1 n d', n=t)
            X_masked[mask] = self.mask_token
                            
            return X_masked, mask


# --------------------------
# Create dummy input
B, P, D = 2, 120, 4  # batch, patches, dim
X = torch.randn(B, P, D)
X_len = torch.tensor([120, 100])  # valid lengths (in timepoints)

# Define mask token and module
mask_token = torch.tensor([-99.0] * D)
specaug = DummySpecAug(mask_token=mask_token, patch_height=2, max_mask_pct=0.3)

# Apply
X_masked, mask = specaug.apply_specaugment_mask(X, X_len, num_masks=2)
breakpoint()
# --------------------------
# Inspect the result
for b in range(B):
    print(f"\nSample {b}:")
    print("Original:")
    print(X[b])
    print("Masked:")
    print(X_masked[b])
    print("Mask positions:", mask[b].nonzero(as_tuple=True)[0].tolist())
