import torch
import pickle
import os
from neural_decoder.bit import BiT_Phoneme
import re
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from dataset import SpeechDataset  # adjust if your path differs
from edit_distance import SequenceMatcher

def evaluate_model(
    model: torch.nn.Module,
    loadedData: Dict[str, List[Dict[str, Any]]],
    args: Dict[str, Any],
    partition: str,               # "test" or "competition"
    device: torch.device,
    fill_max_day: bool = False,   # (kept for compatibility; unused here)
    verbose: bool = True
) -> Tuple[Dict[str, List[Any]], float, List[float]]:
    """
    Minimal evaluation: runs `model` over `partition`, collects outputs, and computes CER.
    Returns (model_outputs, overall_CER, per_day_CER_list).
    If args['nClasses_2'] is not None, also computes/records a second-head CER
    into outputs['cer2'] and outputs['per_day_cer2'].
    """

    # Decide day indices
    if partition == "competition":
        day_indices = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    elif partition == "test" or partition == 'train':
        day_indices = list(range(len(loadedData[partition])))
    else:
        raise ValueError(f"Unknown partition '{partition}'")

    restricted_days = set(args.get("restricted_days", []))
    ventral_6v_only = bool(args.get("ventral_6v_only", False))
    have_second = args.get("nClasses_2") is not None

    # Accumulators
    outputs = {
        "logits": [], "logitLengths": [], "trueSeqs": [], "decodedSeqs": [], "transcriptions": []
    }
    if have_second:
        outputs.update({
            "logits2": [], "logitLengths2": [], "trueSeqs2": [], "decodedSeqs2": []
        })

    per_day_cer: List[float] = []
    total_edit, total_len = 0, 0
    if have_second:
        per_day_cer2: List[float] = []
        total_edit2, total_len2 = 0, 0

    model.eval()

    for day_idx in day_indices:
        if restricted_days and (day_idx not in restricted_days):
            continue

        # Use the actual day index (not enumerate index)
        one_day = loadedData[partition][day_idx]
        # For batch_size=1 default collate is fine
        loader = torch.utils.data.DataLoader(SpeechDataset([one_day]), batch_size=1, shuffle=False, num_workers=0)

        day_edit, day_len = 0, 0
        if have_second:
            day_edit2, day_len2 = 0, 0

        for j, batch in enumerate(loader):
            if have_second:
                X, y, X_len, y_len, _, y2, y2_len = batch
            else:
                X, y, X_len, y_len, _ = batch

            X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)
            if have_second:
                y2, y2_len = y2.to(device), y2_len.to(device)
            day_tensor = torch.tensor([day_idx], dtype=torch.int64, device=device)

            if ventral_6v_only:
                X = X[:, :, :128]

            with torch.no_grad():
                if have_second:
                    pred, pred2 = model.forward(X, X_len, day_tensor)
                else:
                    pred = model.forward(X, X_len, day_tensor)

            # Output lengths
            if hasattr(model, "compute_length"):
                out_lens = model.compute_length(X_len)
            else:
                out_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            # Batch loop (batch_size=1 but keep general)
            B = pred.shape[0]
            for b in range(B):
                tlen = int(y_len[b].item())
                true_seq = np.array(y[b][:tlen].cpu().numpy())

                logits_b = pred[b].detach().cpu().numpy()
                Lb = int(out_lens[b].item())

                outputs["logits"].append(logits_b)
                outputs["logitLengths"].append(Lb)
                outputs["trueSeqs"].append(true_seq)

                # Greedy CTC decode (blank=0)
                decoded = torch.argmax(pred[b, :Lb, :], dim=-1)
                decoded = torch.unique_consecutive(decoded).cpu().numpy()
                decoded = decoded[decoded != 0]
                outputs["decodedSeqs"].append(decoded)

                # CER1
                ed = SequenceMatcher(a=true_seq.tolist(), b=decoded.tolist()).distance()
                total_edit += ed
                total_len  += len(true_seq)
                day_edit   += ed
                day_len    += len(true_seq)

                # Second head (if present)
                if have_second:
                    breakpoint()
                    tlen2 = int(y2_len[b].item())
                    true_seq2 = np.array(y2[b][:tlen2].cpu().numpy())

                    logits_b2 = pred2[b].detach().cpu().numpy()
                    outputs["logits2"].append(logits_b2)
                    outputs["logitLengths2"].append(Lb)  # same Lb as pred

                    decoded2 = torch.argmax(pred2[b, :Lb, :], dim=-1)
                    decoded2 = torch.unique_consecutive(decoded2).cpu().numpy()
                    decoded2 = decoded2[decoded2 != 0]
                    outputs["decodedSeqs2"].append(decoded2)
                    outputs["trueSeqs2"].append(true_seq2)

                    ed2 = SequenceMatcher(a=true_seq2.tolist(), b=decoded2.tolist()).distance()
                    total_edit2 += ed2
                    total_len2  += len(true_seq2)
                    day_edit2   += ed2
                    day_len2    += len(true_seq2)

            # normalized transcript (for display/logging)
            t = one_day["transcriptions"][j].strip()
            t = re.sub(r"[^a-zA-Z\- \']", "", t).replace("--", "").lower()
            outputs["transcriptions"].append(t)

        if day_len > 0:
            day_cer = day_edit / day_len
            per_day_cer.append(day_cer)
            if verbose:
                print(f"CER DAY {day_idx}: {day_cer:.6f}")
        if have_second and day_len2 > 0:
            day_cer2 = day_edit2 / day_len2
            per_day_cer2.append(day_cer2)
            if verbose:
                print(f"CER2 DAY {day_idx}: {day_cer2:.6f}")

    cer = (total_edit / total_len) if total_len > 0 else float("nan")
    if verbose:
        print("Model performance (CER):", cer)

    if have_second:
        cer2 = (total_edit2 / total_len2) if total_len2 > 0 else float("nan")
        if verbose:
            print("Model performance (CER2):", cer2)
            
        return outputs, cer, per_day_cer, cer2, per_day_cer2

    return outputs, cer, per_day_cer



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
        nClasses_2=args['nClasses_2'],
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


def decode_outputs(outputs, num_examples=880):
    # Character vocab and mappings
    CHAR_VOCAB = [
        "<sp>",          # space token
        "!", ",", ".", "?", "'",   # punctuation (incl. apostrophe)
    ] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 'a'..'z'

    _CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
    _ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}

    def idToChar(i: int) -> str:
        return _ID_TO_CHAR[i]

    # Phone definitions and mappings
    PHONE_DEF = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'B',  'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G',
        'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V',
        'W', 'Y', 'Z', 'ZH'
    ]
    PHONE_DEF_SIL = PHONE_DEF + ['SIL']

    def idToPhoneme(i: int) -> str:
        return PHONE_DEF_SIL[i]

    # Lists to collect outputs
    phoneme_decoded_strs = []
    character_decoded_strs = []
    true_seq_strs = []

    # Iterate and decode
    for i in range(min(num_examples, len(outputs['decodedSeqs']))):
        character_decoded = outputs['decodedSeqs'][i]
        phoneme_decoded = outputs['decodedSeqs2'][i]

        character_decoded_str = "".join(
            idToChar(idx - 1) for idx in character_decoded
        ).replace('<sp>', " ")

        phoneme_decoded_str = "".join(
            idToPhoneme(idx - 1) for idx in phoneme_decoded
        ).replace('SIL', " ")

        true_seq_str = outputs['transcriptions'][i]

        phoneme_decoded_strs.append(phoneme_decoded_str)
        character_decoded_strs.append(character_decoded_str)
        true_seq_strs.append(true_seq_str)

    return phoneme_decoded_strs, character_decoded_strs, true_seq_strs
