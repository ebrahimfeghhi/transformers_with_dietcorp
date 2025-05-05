import os
import torch

from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.model import GRUDecoder

# === CONFIGURATION ===
SEEDS_LIST = [0,1,2,3]

SERVER = 'obi'  # Change to 'leia' if needed

BASE_PATHS = {
    'obi': '/data/willett_data',
    'leia': '/home3/skaasyap/willett'
}

DATA_PATHS = {
    'obi': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc'),
    'obi_log': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both'),
    'obi_log_held_out': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days'),
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days')
}

MODEL_NAME_BASE = "neurips_gru_adamw_datalog_time_masked"
DATA_PATH_KEY = f"{SERVER}_log"  # Change to e.g., "leia_log_held_out" if needed

# === MAIN LOOP ===
for seed in SEEDS_LIST:

    model_name = f"{MODEL_NAME_BASE}_seed_{seed}"
    output_dir = os.path.join(BASE_PATHS[SERVER], 'outputs', model_name)
    dataset_path = DATA_PATHS[DATA_PATH_KEY]

    print(f"Using dataset: {dataset_path}")

    # Warn if output directory exists
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Press 'c' to continue.")
        breakpoint()

    # === Experiment Config ===
    args = {
        'seed': seed,
        'outputDir': output_dir,
        'datasetPath': dataset_path,
        'modelName': model_name,
        'device': 'cuda:0',

        # Model hyperparameters
        'nInputFeatures': 256,
        'nClasses': 40,
        'nUnits': 1024,
        'nLayers': 5,
        'dropout': 0.35,
        'input_dropout': 0.2,
        'bidirectional': False,

        # Data preprocessing
        'whiteNoiseSD': 0.2,
        'constantOffsetSD': 0.05,
        'gaussianSmoothWidth': 2.0,
        'strideLen': 4,
        'kernelLen': 32,
        'restricted_days': [],
        'maxDay': 14,
        'nDays': 24,

        # Optimization
        'AdamW': True,
        'lrStart': 0.001,
        'lrEnd': 0.001,
        'l2_decay': 1e-5,
        'beta1': 0.90,
        'beta2': 0.999,
        'learning_scheduler': 'None',
        'n_epochs': 600,
        'batchSize': 64,

        # Optional loading
        'load_pretrained_model': '', 
        'wandb_id': '', 
        'start_epoch': 0,
        
        'ventral_6v_only': False, 
        
        'max_mask_pct': 0.075, 
        'num_masks': 20
    }

    # === Instantiate Model ===
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

    # === Train ===
    trainModel(args, model)
