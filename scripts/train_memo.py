import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0,
                    help='CUDA device number to use (default: 0)')

args_device = parser.parse_args()



device = f'cuda:{args_device.cuda}'


possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/data/willett_data/ptDecoder_ctc_both', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both',
                      '/home3/skaasyap/willett/data_log_both_held_out_days']


args = {}
modelName = 'memo_16_16_dropout_01'
args['modelName'] = modelName
args['outputDir'] = possiblePath_dir[1] + modelName
args['datasetPath'] = possiblePaths_data[-1]

args['memo_augs'] = 16
args['memo_epochs'] = 16
args['next_trial_memo'] = False
args['evenDaysOnly'] = False
args['model_to_restore'] = 'neurips_transformer_time_masked_held_out_days_seed_0'

args['freeze_all_except_patch_linear'] = False
args['unfreeze_layer_1'] = False
args['restore_model_each_update'] = True # restore original model after every update.
args['restore_model_each_day'] = False
args['modelWeightPath'] = f"/home3/skaasyap/willett/outputs/{args['model_to_restore']}/modelWeights"

args['patch_size']= (5, 256) #TODO
args['dim'] = 384 #TODO
args['depth'] = 7 #TODO
args['heads'] = 6
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0.1
args['input_dropout'] = 0
args['num_masks'] = 20
args['max_mask_pct'] = 0.05

args['gaussianSmoothWidth'] = 2.0
args['nClasses'] = 40

args['optimizer'] = 'AdamW'
args['load_optimizer_state'] = False
args['optimizer_path'] = f"/home3/skaasyap/willett/outputs/{args['model_to_restore']}/optimizer"

args['l2_decay'] = 0

args['lrStart'] = 0.0001
args['lrEnd'] = 0.0001

args['beta1'] = 0.90
args['beta2'] = 0.999

args['look_ahead'] = 0 

args['extra_notes'] = ("")

args['device'] = device

args['seed'] = 0

args['T5_style_pos'] = True

from neural_decoder.neural_decoder_trainer_memo import trainModel
from neural_decoder.bit import BiT_Phoneme

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
    look_ahead=0,
    gaussianSmoothWidth=args['gaussianSmoothWidth'],
    T5_style_pos=args['T5_style_pos'], 
    max_mask_pct=args['max_mask_pct'], 
    num_masks=args['num_masks'], 
).to(args['device'])

import torch

model.load_state_dict(torch.load(args['modelWeightPath'],
                    map_location=args['device']), strict=True)

if args['freeze_all_except_patch_linear']:
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze just the Linear layer
    linear_layer = model.to_patch_embedding[2]  # assumes Linear is the third module in Sequential
    for param in linear_layer.parameters():
        param.requires_grad = True
        
    if args['unfreeze_layer_1']:
        first_transformer_layer = model.transformer.layers[0]  # assumes standard transformer format
        for param in first_transformer_layer.parameters():
            param.requires_grad = True
        
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}")
        

#if args['freeze_params_except_day']:
#    freeze_except_day_specific(model)

trainModel(args, model)