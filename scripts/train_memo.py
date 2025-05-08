import os
import argparse
import torch
import numpy as np

from neural_decoder.neural_decoder_trainer_memo import trainModel
from neural_decoder.bit import BiT_Phoneme

# === COMMAND‑LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Memo‑style training script (formatted).")
parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use (default: 0)")
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
cli_args = parser.parse_args()

DEVICE = f"cuda:{cli_args.cuda}"
SEED = cli_args.seed

# === PATH CONFIGURATION ===
BASE_PATHS = {
    "obi": "/data/willett_data",
    "leia": "/home3/skaasyap/willett"
}

DATA_PATHS = {
    'obi': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc'),
    'obi_log': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both'),
    'obi_log_held_out': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days'),
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days')
}

SERVER = "leia"  # change to "obi" if running on Obi
DATA_PATH_KEY = f"{SERVER}_log"

MODEL_TO_RESTORE = "neurips_transformer_time_masked_seed_0"
MODEL_NAME = "memo_best_params_all_day"
OUTPUT_DIR = os.path.join(BASE_PATHS[SERVER], "outputs", MODEL_NAME)

# === EXPERIMENT CONFIG ===
args = {
    # bookkeeping
    "seed": SEED,
    "modelName": MODEL_NAME,
    "outputDir": OUTPUT_DIR,
    "datasetPath": DATA_PATHS[DATA_PATH_KEY],

    # memo‑specific settings
    "memo_augs": 16,
    "memo_epochs": 16,
    "next_trial_memo": False,
    "evenDaysOnly": False,

    # model restore settings
    "model_to_restore": MODEL_TO_RESTORE,
    "restore_model_each_update": True,
    "restore_model_each_day": False,
    "modelWeightPath": os.path.join(BASE_PATHS["leia"], "outputs", MODEL_TO_RESTORE, "modelWeights"),
    "optimizer_path": os.path.join(BASE_PATHS["leia"], "outputs", MODEL_TO_RESTORE, "optimizer"),

    # architecture
    "patch_size": (5, 256),
    "dim": 384,
    "depth": 7,
    "heads": 6,
    "mlp_dim_ratio": 4,
    "dim_head": 64,
    "dropout": 0.0,
    "input_dropout": 0.0,
    "num_masks": 20,
    "max_mask_pct": 0.05,
    "gaussianSmoothWidth": 2.0,
    "nClasses": 40,
    "T5_style_pos": True,

    # optimisation
    "optimizer": "AdamW",
    "load_optimizer_state": False,
    "l2_decay": 0.0,
    "lrStart": 1e-4,
    "lrEnd": 1e-4,
    "beta1": 0.90,
    "beta2": 0.999,

    # misc
    "look_ahead": 0,
    "extra_notes": "",
    "device": DEVICE,

    # fine‑tuning flags
    "freeze_all_except_patch_linear": False,
    "unfreeze_layer_1": False,
}

print(f"Using dataset: {args['datasetPath']}")
print(f"Saving outputs to: {args['outputDir']}")

os.makedirs(args["outputDir"], exist_ok=True)

# === REPRODUCIBILITY ===
torch.manual_seed(args["seed"])
np.random.seed(args["seed"])

# === MODEL INITIALISATION ===
model = BiT_Phoneme(
    patch_size=args["patch_size"],
    dim=args["dim"],
    dim_head=args["dim_head"],
    nClasses=args["nClasses"],
    depth=args["depth"],
    heads=args["heads"],
    mlp_dim_ratio=args["mlp_dim_ratio"],
    dropout=args["dropout"],
    input_dropout=args["input_dropout"],
    look_ahead=args["look_ahead"],
    gaussianSmoothWidth=args["gaussianSmoothWidth"],
    T5_style_pos=args["T5_style_pos"],
    max_mask_pct=args["max_mask_pct"],
    num_masks=args["num_masks"], 
    mask_token_zeros=False,
    num_masks_channels=0,
    max_mask_channels=0,
    dist_dict_path=None
).to(args["device"])

# === LOAD PRETRAINED WEIGHTS ===
model.load_state_dict(
    torch.load(args["modelWeightPath"], map_location=args["device"]),
    strict=True
)
print(f"Loaded pretrained weights from {args['modelWeightPath']}")

# === OPTIONAL PARAMETER FREEZING ===
if args["freeze_all_except_patch_linear"]:
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze patch‑embedding linear projection (assumed third module)
    for p in model.to_patch_embedding[2].parameters():
        p.requires_grad = True

    if args["unfreeze_layer_1"]:
        for p in model.transformer.layers[0].parameters():
            p.requires_grad = True

    # Log trainable params
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"[Trainable] {n}")

# === TRAIN ===
trainModel(args, model)
