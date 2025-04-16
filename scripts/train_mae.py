
modelName = 'mask_0.5_2_pad'

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

args['patch_size']= (5, 256) #TODO
args['dim'] = 384 #TODO
args['depth'] = 9 #TODO
args['heads'] = 6
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0.1
args['T5_style_pos'] = True
args['look_ahead'] = 0 
args['input_dropout'] = 0.2
args['max_mask_pct'] = 0.05
args['num_masks'] = 20
args['nDays'] = 24

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
args['num_epochs'] = 10000
args['gaussianSmoothWidth'] = 2.0

args['look_ahead'] = 0 

args['extra_notes'] = ("")

args['device'] = 'cuda:3'

from neural_decoder.mae_main import trainModel

trainModel(args)