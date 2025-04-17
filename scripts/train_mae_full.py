
modelName = 'phoneme_run_1'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/data/willett_data/ptDecoder_ctc_both', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both']

args = {}
args['outputDir'] = possiblePath_dir[0] + modelName
args['datasetPath'] = possiblePaths_data[1]


# define parameters for baseline encoder 
args['patch_size']= (5, 256) #TODO
args['dim'] = 384 #TODO
args['depth'] = 5 #TODO
args['heads'] = 6
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0.1
args['T5_style_pos'] = True
args['look_ahead'] = 0 
args['max_mask_pct'] = 0.05
args['num_masks'] = 20
args['nDays'] = 24

# define parameters for phoneme decoder
args['patch_size_phon']= None
args['dim_phon'] = 384 #TODO
args['depth_phon'] = 7 #TODO
args['heads_phon'] = 6
args['mlp_dim_ratio_phon'] = 4 #TODO
args['dim_head_phon'] = 64
args['dropout_phon'] = 0.4
args['input_dropout'] = 0.2
args['whiteNoiseSD'] = 0.4
args['constantOffsetSD'] = 0.1
args['nClasses'] = 40


# define parameters for reconstruction loss
args['decoder_dim'] = 384
args['num_decoder_layers'] = 5 #TODO
args['num_decoder_heads'] = 6
args['decoder_dim_head'] = 64

args['batchSize'] = 64

args['l2_decay'] = 1e-5
args['lrStart'] = 0.001
args['lrEnd'] = 0.001
args['mae_scalar_loss'] = 0
args['phoneme_scalar_loss'] = 1
args['num_epochs'] = 10000
args['gaussianSmoothWidth'] = 2.0

args['extra_notes'] = ("")

args['device'] = 'cuda:1'

args['seed'] = 0

args['T5_style_pos'] = True

args['n_epochs'] = 10000

args['AdamW'] = True
args['cosineAnnealing'] = True


from neural_decoder.neural_decoder_trainer_mae import trainModel
from neural_decoder.bit import BiT_Phoneme
from neural_decoder.mae import MAE


# define the encoder module. 

# Encoder part. 
enc_model = BiT_Phoneme(
    patch_size=args['patch_size'],
    dim=args['dim'],
    dim_head=args['dim_head'],
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
    nClasses=args['nClasses'], 
    mae_mode=False
).to(args['device'])


# Decoder for phoneme logits. 
phoneme_decoder = BiT_Phoneme(
    patch_size=None,
    dim=args['dim_phon'],
    dim_head=args['dim_head_phon'], 
    nClasses=args['nClasses'],
    depth=args['depth_phon'],
    heads=args['heads_phon'],
    mlp_dim_ratio=args['mlp_dim_ratio_phon'],
    dropout=args['dropout_phon'],
    input_dropout=args['input_dropout'],
    look_ahead=0,
    nDays=args['nDays'],
    gaussianSmoothWidth=0,
    T5_style_pos=args['T5_style_pos'], 
    max_mask_pct=None, 
    num_masks=None, 
    mae_mode=True
).to(args['device'])


model = MAE(
    encoder=enc_model,
    phoneme_decoder=phoneme_decoder, 
    encoder_dim = args['dim'], 
    decoder_dim = args['decoder_dim'], #same shape as the encoder model outputs
    decoder_depth=args['num_decoder_layers'],
    decoder_heads = args['num_decoder_heads'],
    decoder_dim_head = args['decoder_dim_head'], 
    gaussianSmoothWidth = args['gaussianSmoothWidth'], 
    constantOffsetSD=args['constantOffsetSD'], 
    whiteNoiseSD=args['whiteNoiseSD']
).to(args['device'])


trainModel(args, model)