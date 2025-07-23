import torch
import torch.nn.functional as F

def train_step(self,
               model, 
               inputs,
               layerIdx,
               labels,
               time_steps,
               confidences,
               sess_idx,
               white_noise_sd,
               constant_offset_sd,
               random_walk_sd,
               random_walk_axis,
               max_seq_len=500,
               grad_clip_value=10.0):

    B, T, C = inputs.shape  # assuming inputs shape is (B, T, C)

    # Add noise
    inputs = inputs + torch.randn_like(inputs) * white_noise_sd
    inputs = inputs + torch.randn(B, 1, C, device=inputs.device) * constant_offset_sd

    random_walk = torch.randn_like(inputs) * random_walk_sd
    inputs = inputs + torch.cumsum(random_walk, dim=random_walk_axis)

    inputs = self.gauss_smooth(inputs)  # Replace with your PyTorch-compatible smoothing function

    logits = model(inputs)  # (B, T, V)

    # Subsample time steps
    time_steps = self.nsd.model.getSubsampledTimeSteps(time_steps)

    # Prepare CTC loss inputs
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    input_lengths = time_steps  # (B,)

    # Convert labels to CTC-compatible format (concatenated targets)
    flat_labels = []
    target_lengths = []
    for b in range(B):
        target = labels[b][labels[b] != 0] - 1  # Remove padding, shift labels down
        flat_labels.append(target)
        target_lengths.append(len(target))
    targets = torch.cat(flat_labels)
    target_lengths = torch.tensor(target_lengths, device=logits.device)

    ctc_loss = F.ctc_loss(
        log_probs.transpose(0, 1),  # (T, B, V)
        targets,
        input_lengths,
        target_lengths,
        blank=log_probs.shape[-1] - 1,  # assuming blank is last
        reduction='none'
    )

    ctc_loss = torch.mean(ctc_loss * confidences.to(ctc_loss.device))

    # Regularization loss (PyTorch doesn't track these automatically)
    reg_loss = sum(p.norm(2) ** 2 for p in self.nsd.model.parameters() if p.requires_grad) * 0
    reg_loss += sum(p.norm(2) ** 2 for p in self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].parameters() if p.requires_grad) * 0

    total_loss = ctc_loss + reg_loss

    # Determine trainable parameters
    if self.config.freeze_backbone:
        trainables = self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].parameters()
    else:
        trainables = list(self.nsd.model.parameters()) + \
                     list(self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].parameters())

    # Backward pass and optimization
    self.nsd.optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainables, grad_clip_value)
    self.nsd.optimizer.step()

    return ctc_loss.item(), reg_loss.item(), total_loss.item(), grad_norm.item()