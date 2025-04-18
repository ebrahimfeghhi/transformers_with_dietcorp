
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
args['dropout'] = 0.4
args['dropout_reconstruct'] = 0.1
args['input_dropout'] = 0.2
args['T5_style_pos'] = True
args['look_ahead'] = 0 
args['max_mask_pct'] = 0.05
args['num_masks'] = 20
args['nDays'] = 24

args['whiteNoiseSD'] = 0
args['constantOffsetSD'] = 0


args['batchSize'] = 64

args['l2_decay'] = 1e-5
args['lrStart'] = 0.001
args['lrEnd'] = 0.001
args['mae_scalar_loss'] = 1
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
from neural_decoder.bit_with_adapt import BiT_with_adapt


mdoel = BiT_with_adapt(patch_size=args['patch_size'], 
               dim=args['dim'], depth_phoneme=args['depth_phoneme'], 
               depth_reconstruct=args['depth_reconstruct'], heads=args['heads'], 
               mlp_dim_ratio=args['mlp_dim_ratio'], dim_head=args['dim_head'], 
               dropout=args['dropout'], dropout_reconstruct=args['dropout_reconstruct'], 
               input_dropout=args['input_dropout'], look_ahead=0, nDays=args['nDays'], 
               gaussianSmoothWidth=args['gaussianSmoothWidth'], nClasses=args['nClasses'], 
               T5_style_pos=True, max_mask_pct=args['max_mask_pct'], num_masks=args['num_masks']).to(args['deivce'])




trainModel(args, model)