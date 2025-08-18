import os
import torch
import numpy as np

from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.measure_memory import trainModel_mem
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
    'obi_log_char': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_char'),
    'obi_held_out': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days'),
    'obi_held_out_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days_1'),
    'obi_held_out_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days_2'),
    'obi_big_0': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_held_out_days_big_0'), 
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days')
}

MODEL_NAME_BASE = "baseline_gru_char"
DATA_PATH_KEY = f"{SERVER}_log_char"  # Change to e.g., "leia_log_held_out" if needed

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
        'device': 'cuda:1',

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
        'maxDay': None, # SET TO NONE IF NOT DOING HELD OUT DAYS TESTING
        'nDays': 24,
        
        # Optimization
        'AdamW': False,
        'lrStart': 0.02,
        'lrEnd': 0.02,
        'l2_decay': 1e-5,
        'beta1': 0.90,
        'beta2': 0.999,
        'learning_scheduler': 'none',
        'milestones': [400],
        'gamma': 0.1,
        'n_epochs': 73,
        'batchSize': 64,

        # Optional loading
        'load_pretrained_model': '', 
        'wandb_id': '', 
        'start_epoch': 0,
        
        'ventral_6v_only': False, 
        
        'max_mask_pct': 0.0, 
        'num_masks': 0, 
        'linderman_lab': False
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
        num_masks=args['num_masks'], 
        linderman_lab=args['linderman_lab']
    ).to(args["device"])
    
    
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        
    count_parameters(model)


    # === Train ===
    trainModel(args, model)
