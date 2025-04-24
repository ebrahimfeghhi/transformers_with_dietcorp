import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0,
                    help='CUDA device number to use (default: 0)')

args_device = parser.parse_args()

device = f'cuda:{args_device.cuda}'

modelName = 'None'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/data/willett_data/ptDecoder_ctc_both', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both']

args = {}
args['outputDir'] = possiblePath_dir[1] + modelName
args['datasetPath'] = possiblePaths_data[-1]

args['memo_augs'] = 32
args['memo_epochs'] = 16
args['evenDaysOnly'] = True

args['patch_size']= (5, 256) #TODO
args['dim'] = 384 #TODO
args['depth'] = 7 #TODO
args['heads'] = 6
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0
args['input_dropout'] = 0

args['gaussianSmoothWidth'] = 2.0
args['nDays'] = 24
args['nClasses'] = 40

args['optimizer'] = 'AdamW'
args['load_optimizer_state'] = False
args['optimizer_path'] = "/home3/skaasyap/willett/outputs/spec_aug_time_best_4_22/optimizer"

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

args['modelWeightPath'] = "/home3/skaasyap/willett/outputs/spec_aug_time_best_4_22/modelWeights"

args['mask_token_zero'] = False
args['num_masks_channels'] = 0
args['max_mask_channels'] = 0
args['max_mask_pct'] = 0.05
args['num_masks'] = 20

args['freeze_params_except_day'] = True
args['restore_model'] = True


args['dist_dict_path'] = '/home3/skaasyap/willett/outputs/dist_dict.pt'

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
    nDays=args['nDays'],
    gaussianSmoothWidth=args['gaussianSmoothWidth'],
    T5_style_pos=args['T5_style_pos'], 
    max_mask_pct=args['max_mask_pct'], 
    num_masks=args['num_masks'], 
    mask_token_zeros=args['mask_token_zero'], 
    num_masks_channels=args['num_masks_channels'], 
    max_mask_channels=args['max_mask_channels'], 
    dist_dict_path=args['dist_dict_path']
).to(args['device'])

import torch


model.load_state_dict(torch.load(args['modelWeightPath'],
                    map_location=args['device']), strict=True)

def freeze_except_day_specific(model):
    print("Freezing all parameters except day-specific weights and biases...\n")
    for name, param in model.named_parameters():
        if 'dayWeights' in name or 'dayBias' in name:
            param.requires_grad = True
            print(f"[Unfrozen] {name}")
        else:
            param.requires_grad = False
            print(f"[Frozen] {name}")
    print("\nDone freezing.")
    
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

if args['freeze_params_except_day']:
    freeze_except_day_specific(model)

trainModel(args, model)