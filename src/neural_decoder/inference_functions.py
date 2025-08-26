import torch
import pickle
import os
from bit import BiT_Phoneme  # adjust import path if different
import re
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from dataset import SpeechDataset  # adjust if your path differs
from edit_distance import SequenceMatcher


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

def evaluate_model(
    model: torch.nn.Module,
    loadedData: Dict[str, List[Dict[str, Any]]],
    args: Dict[str, Any],
    partition: str,               # "test" or "competition"
    device: torch.device,
    fill_max_day: bool = False,   # optional, keep behavior you had
    verbose: bool = True
) -> Tuple[Dict[str, List[Any]], float, List[float]]:
    """
    Minimal evaluation: runs `model` over `partition`, collects outputs, and computes CER.
    Returns (model_outputs, overall_CER, per_day_CER_list).
    """

    # Decide day indices
    if partition == "competition":
        day_indices = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    elif partition == "test":
        day_indices = list(range(len(loadedData[partition])))
    else:
        raise ValueError(f"Unknown partition '{partition}'")

    # Pull common flags from args with safe defaults
    restricted_days = set(args.get("restricted_days", []))
    ventral_6v_only = bool(args.get("ventral_6v_only", False))

    # Accumulators
    outputs = {"logits": [], "logitLengths": [], "trueSeqs": [], "transcriptions": []}
    per_day_cer: List[float] = []
    total_edit, total_len = 0, 0

    model.eval()

    for idx_in_enum, day_idx in enumerate(day_indices):
        if restricted_days and (day_idx not in restricted_days):
            continue

        # one-day dataset/loader (mirror your original)
        one_day = loadedData[partition][idx_in_enum]
        loader = torch.utils.data.DataLoader(SpeechDataset([one_day]), batch_size=1, shuffle=False, num_workers=0)

        day_edit, day_len = 0, 0

        for j, (X, y, X_len, y_len, _) in enumerate(loader):
            X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)
            day_tensor = torch.tensor([day_idx], dtype=torch.int64, device=device)

            if ventral_6v_only:
                X = X[:, :, :128]

            with torch.no_grad(): 
                pred = model.forward(X, X_len, day_tensor)[:,:, :30]

            # Output lengths
            if hasattr(model, "compute_length"):
                out_lens = model.compute_length(X_len)
            else:
                # fallback: conv-style
                out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            # Batch loop (batch_size=1, but keep general)
            for b in range(pred.shape[0]):
                tlen = int(y_len[b].item())
                true_seq = np.array(y[b][:tlen].cpu().numpy())

                logits_b = pred[b].detach().cpu().numpy()
                Lb = int(out_lens[b].item())

                outputs["logits"].append(logits_b)
                outputs["logitLengths"].append(Lb)
                outputs["trueSeqs"].append(true_seq)

                # Greedy CTC decode (blank=0), collapse repeats
                decoded = torch.argmax(pred[b, :Lb, :], dim=-1)
                decoded = torch.unique_consecutive(decoded).cpu().numpy()
                decoded = decoded[decoded != 0]
                 
                matcher = SequenceMatcher(
                    a=true_seq.tolist(), b=decoded.tolist()
                )
            
                ed = matcher.distance()
                total_edit += ed
                total_len += len(true_seq)
                day_edit += ed
                day_len += len(true_seq)

            # normalized transcript
            t = one_day["transcriptions"][j].strip()
            t = re.sub(r"[^a-zA-Z\- \']", "", t).replace("--", "").lower()
            outputs["transcriptions"].append(t)

        if day_len > 0:
            day_cer = day_edit / day_len
            per_day_cer.append(day_cer)
            if verbose:
                print(f"CER DAY {day_idx}: {day_cer:.6f}")

    cer = (total_edit / total_len) if total_len > 0 else float("nan")
    if verbose:
        print("Model performance (CER):", cer)

    return outputs, cer, per_day_cer
