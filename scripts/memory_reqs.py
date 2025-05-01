import torch
from contextlib import contextmanager

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params, trainable_params)

@contextmanager
def _temporarily(model, training=True):
    was_training = model.training
    model.train(training)
    yield
    model.train(was_training)

def activation_size_MB(model, *forward_args, device="cuda"):
    """
    Returns the cumulative size (in MB) of all tensors that PyTorch autograd
    must keep for the backward pass.

    Parameters
    ----------
    model : nn.Module
    *forward_args :  the exact arguments you normally pass to model(...)
    device : "cuda" | "cpu"
    """
    model = model.to(device)

    handles, seen, total_bytes = [], set(), 0

    def hook(_, __, out):
        nonlocal total_bytes
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if torch.is_tensor(t) and t.requires_grad and id(t) not in seen:
                seen.add(id(t))
                total_bytes += t.numel() * t.element_size()

    for m in model.modules():
        if not any(m.children()):          # leaf module
            handles.append(m.register_forward_hook(hook))

    with _temporarily(model, training=True):
        forward_args = [arg.to(device) if torch.is_tensor(arg) else arg
                        for arg in forward_args]
        model(*forward_args)

    for h in handles:
        h.remove()

    return total_bytes / 1_048_576          # bytes → MB
