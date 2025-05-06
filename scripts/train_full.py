import os
import torch
import numpy as np

from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.bit import BiT_Phoneme

# === CONFIGURATION ===
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


seed_list = [0,1,2,3]

SERVER = 'leia'  # Change to 'leia' if needed
DATA_PATH_KEY = f"{SERVER}_log"  # Change to e.g., "leia_log_held_out" if needed
model_name_base = "neurips_transformer_ablation_no_lRscheduler"

# === MAIN LOOP ===
for seed in seed_list:
    
    model_name = f"{model_name_base}_seed_{seed}"
    output_dir = os.path.join(BASE_PATHS[SERVER], 'outputs', model_name)
    dataset_path = DATA_PATHS[DATA_PATH_KEY]

    # Create config dictionary
    args = {
        'seed': seed,
        'outputDir': output_dir,
        'datasetPath': dataset_path,
        'modelName': model_name,
        'testing_on_held_out': False,
        'maxDay': 14,
        'restricted_days': [],
        'patch_size': (5, 256),
        'dim': 384,
        'depth': 7,
        'heads': 6,
        'mlp_dim_ratio': 4,
        'dim_head': 64,
        'T5_style_pos': True,
        'nClasses': 40,
        'whiteNoiseSD': 0.2,
        'gaussianSmoothWidth': 2.0,
        'constantOffsetSD': 0.05,
        'l2_decay': 1e-5,
        'input_dropout': 0.2,
        'dropout': 0.35,
        'AdamW': True,
        'learning_scheduler': 'None',
        'lrStart': 0.001,
        'lrEnd': 0.001,
        'batchSize': 64,
        'beta1': 0.90,
        'beta2': 0.999,
        'n_epochs': 600,
        'milestones': [400],
        'gamma': 0.1,
        'look_ahead': 0,
        'extra_notes': "",
        'device': 'cuda:3',
        'load_pretrained_model': "",
        'wandb_id': "",
        'start_epoch': 0,
        'ventral_6v_only': False,
        'mask_token_zero' : False,
        'num_masks_channels' : 0, # number of masks per grid
        'max_mask_channels' : 0, # maximum number of channels to mask per mask
        'max_mask_pct' : 0.075, 
        'num_masks' : 20,
        'dist_dict_path': '/home3/skaasyap/willett/outputs/dist_dict.pt' 
    }

    print(f"Using dataset: {args['datasetPath']}")

    # Warn if output directory exists
    if os.path.exists(args['outputDir']):
        print(f"Output directory '{args['outputDir']}' already exists. Press 'c' to continue.")
        breakpoint()
        
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    # Instantiate model
    model = BiT_Phoneme(
        patch_size=args['patch_size'],
        dim=args['dim'],
        dim_head=args['dim_head'],
        nClasses=args['nClasses'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        look_ahead=args['look_ahead'],
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'],
        max_mask_pct=args['max_mask_pct'],
        num_masks=args['num_masks'], 
        mask_token_zeros=args['mask_token_zero'], 
        num_masks_channels=args['num_masks_channels'], 
        max_mask_channels=args['max_mask_channels'], 
        dist_dict_path=args['dist_dict_path']
    ).to(args['device'])

    # Load pretrained model if specified
    if args['load_pretrained_model']:
        ckpt_path = os.path.join(args['load_pretrained_model'], 'modelWeights')
        model.load_state_dict(torch.load(ckpt_path, map_location=args['device']), strict=True)
        print(f"Loaded pretrained model from {ckpt_path}")

    # Train
    trainModel(args, model)
