
modelName = 'spec_aug_time_best_4_22'

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

args['patch_size']= (5, 256) #TODO
args['dim'] = 384 #TODO
args['depth'] = 7 #TODO
args['heads'] = 6
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0.35
args['input_dropout'] = 0.2

args['whiteNoiseSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['constantOffsetSD'] = 0.05
args['nDays'] = 24
args['nClasses'] = 40
args['batchSize'] = 64

args['l2_decay'] = 1e-5

args['AdamW'] = True
args['learning_scheduler'] = 'multistep'

args['lrStart'] = 0.001
args['lrEnd'] = 0.001

args['milestones'] = [400] # number of epochs after which to drop the learning rate
args['gamma'] = 0.1 # factor by which to drop the learning rate at milestone 

args['T_0'] = 500
args['T_mult'] = 2

args['beta1'] = 0.90
args['beta2'] = 0.999

args['look_ahead'] = 0 

args['extra_notes'] = ("")

args['device'] = 'cuda:2'

args['seed'] = 0

args['T5_style_pos'] = True

args['n_epochs'] = 2000


args['load_pretrained_mae'] = ""

args['mask_token_zero'] = False
args['num_masks_channels'] = 0
args['max_mask_channels'] = 0
args['max_mask_pct'] = 0.075
args['num_masks'] = 20

args['dist_dict_path'] = '/home3/skaasyap/willett/outputs/dist_dict.pt'

from neural_decoder.neural_decoder_trainer import trainModel
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

if len(args['load_pretrained_mae']) > 0:
    print("LOADING PRETRAINED MAE WEIGHTS")
    model.load_pretrained_transformer(args['load_pretrained_mae'])
    

trainModel(args, model)