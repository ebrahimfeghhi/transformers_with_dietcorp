import os 

# === CONFIGURATION ===
SEEDS_LIST = [0,1,2,3]

SERVER = 'leia'  # Change to 'leia' if needed

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

MODEL_NAME_BASE = "mae_masked_20_075"
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

    args = {}
    args['outputDir'] = output_dir
    args['datasetPath'] = dataset_path
    args['modelName'] = model_name

    args['patch_size']= (5, 256) #TODO
    args['dim'] = 384 #TODO
    args['depth'] = 7 #TODO
    args['heads'] = 6
    args['mlp_dim_ratio'] = 4 #TODO
    args['dim_head'] = 64
    args['dropout'] = 0.1
    args['T5_style_pos'] = True
    args['look_ahead'] = 0 
    args['input_dropout'] = 0.2
    args['max_mask_pct'] = 0.075
    args['num_masks'] = 20

    args['whiteNoiseSD'] = 0
    args['constantOffsetSD'] = 0
    args['decoder_dim'] = 384
    args['num_decoder_layers'] = 3 #TODO
    args['num_decoder_heads'] = 6
    args['decoder_dim_head'] = 64
    args['nClasses'] = 40

    args['batchSize'] = 64

    args['weight_decay'] = 1e-5
    args['learning_rate'] = 1e-3
    args['num_epochs'] = 600
    args['gaussianSmoothWidth'] = 2.0

    args['cosineAnnealing'] = False

    args['look_ahead'] = 0 

    args['extra_notes'] = ("")

    args['device'] = 'cuda:2'

    from neural_decoder.mae_main import trainModel

    trainModel(args)