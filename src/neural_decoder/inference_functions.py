import torch
import pickle
import os
from bit import BiT_Phoneme  # adjust import path if different

def load_bit_phoneme_model(folder: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load a BiT_Phoneme model from a folder containing 'args' and 'modelWeights'.

    Args:
        folder (str): Path to folder containing 'args' (pickle) and 'modelWeights' (torch).
        device (torch.device): Device to map the model onto.

    Returns:
        torch.nn.Module: The loaded BiT_Phoneme model in eval mode.
    """
    # Load args
    args_path = os.path.join(folder, "args")
    with open(args_path, "rb") as handle:
        args = pickle.load(handle)

    # Ensure defaults
    if 'mask_token_zero' not in args:
        args['mask_token_zero'] = False

    # Instantiate model
    model = BiT_Phoneme(
        patch_size=args['patch_size'],
        dim=args['dim'],
        dim_head=args['dim_head'],
        nClasses=args['nClasses'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dropout=0,
        input_dropout=0,
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'],
        max_mask_pct=0.0,
        num_masks=0,
        mask_token_zeros=args['mask_token_zero'],
        num_masks_channels=0,
        max_mask_channels=0,
        dist_dict_path=0
    ).to(device)

    # Load weights
    ckpt_path = os.path.join(folder, "modelWeights")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model, args
