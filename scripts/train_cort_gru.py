import os
import argparse
import torch
import numpy as np

from neural_decoder.neural_decoder_trainer_cort import trainModel
from neural_decoder.model import GRUDecoder

# === COMMAND‑LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Memo‑style training script (formatted).")
parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use (default: 0)")
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
cli_args = parser.parse_args()


# this script is only for measuring memory and speed. Did not evaluate performance with memo
# on GRU. 

DEVICE = f"cuda:{cli_args.cuda}"
SEED = cli_args.seed

# === PATH CONFIGURATION ===
BASE_PATHS = {
    'obi': '/data/willett_data',
    'leia': '/home3/skaasyap/willett'
}

DATA_PATHS = {
    'obi': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc'),
    'obi_log': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both'),
    'obi_log_held_out': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days'),
    'obi_log_held_out_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_1'),
    'obi_log_held_out_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_2'),
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days'), 
    'leia_log_held_out_1': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_1'), 
    'leia_log_held_out_2': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_2')
}

SERVER = "obi"  # change to "obi" if running on Obi
DATA_PATH_KEY = f"{SERVER}_log_held_out_2"

MODEL_TO_RESTORE = f"gru_fully_held_out_days_mod_2_seed_3"
MODEL_NAME = f"scratch"

OUTPUT_DIR = os.path.join(BASE_PATHS[SERVER], "outputs", MODEL_NAME)

# === EXPERIMENT CONFIG ===
args = {
      # bookkeeping
    "seed": SEED,
    "modelName": MODEL_NAME,
    "outputDir": OUTPUT_DIR,
    "datasetPath": DATA_PATHS[DATA_PATH_KEY],
    'skip_days': [],
    
    # memo‑specific settings
    "memo_augs": 16,
    "memo_epochs": 1,
    "evenDaysOnly": False,
    
    # model restore settings
    "model_to_restore": MODEL_TO_RESTORE,
    "restore_model_each_update": False,
    "restore_model_each_day": False,
    "modelWeightPath": os.path.join(BASE_PATHS[SERVER], "outputs", MODEL_TO_RESTORE, "modelWeights"),
    "optimizer_path": os.path.join(BASE_PATHS[SERVER], "outputs", MODEL_TO_RESTORE, "optimizer"),
    
    # Model hyperparameters
    'nInputFeatures': 256,
    'nClasses': 40,
    'nUnits': 1024,
    'nLayers': 5,
    'dropout': 0.40,
    'input_dropout': 0,
    'bidirectional': False,

    # Data preprocessing
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'gaussianSmoothWidth': 2.0,
    'strideLen': 4,
    'kernelLen': 32,
    'restricted_days': [],
    'maxDay': 14,
    'nDays': 24,

    # Optimization
    'AdamW': False,
    'lrStart': 0.02,
    'lrEnd': 0.02,
    'l2_decay': 1e-5,
    'beta1': 0.90,
    'beta2': 0.999,
    'learning_scheduler': 'None',
    'milestones': [400],
    'gamma': 0.1,
    'n_epochs': 73,
    'batchSize': 64,

    # Optional loading
    'load_pretrained_model': '', 
    'wandb_id': '', 
    'start_epoch': 0,
    
    'ventral_6v_only': False, 
    
    'max_mask_pct': 0.05, 
    'num_masks': 20, 
    'device': 'cuda:1', 
    'optimizer': 'AdamW', 
    'load_optimizer_state': False,
    'lrDecrease': False
}

# === Instantiate Model ===
torch.manual_seed(args["seed"])
np.random.seed(args["seed"])

model = GRUDecoder(
    neural_dim=args["nInputFeatures"],
    n_classes=args["nClasses"],
    hidden_dim=args["nUnits"],
    layer_dim=args["nLayers"],
    nDays=args['nDays'],
    dropout=args["dropout"],
    device=args["device"],
    strideLen=args["strideLen"],
    kernelLen=args["kernelLen"],
    gaussianSmoothWidth=args["gaussianSmoothWidth"],
    bidirectional=args["bidirectional"],
    input_dropout=args['input_dropout'], 
    max_mask_pct=args['max_mask_pct'],
    num_masks=args['num_masks']
).to(args["device"])

# === LOAD PRETRAINED WEIGHTS ===
model.load_state_dict(
    torch.load(args["modelWeightPath"], map_location=args["device"]),
    strict=True
)
print(f"Loaded pretrained weights from {args['modelWeightPath']}")


# === OPTIONAL PARAMETER FREEZING ===
if True:
    for name, p in model.named_parameters():
        if name in {
            "dayWeights",
            "dayBias"
        }:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Log trainable params
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"[Trainable] {n}")
            

# === TRAIN ===
trainModel(args, model)
