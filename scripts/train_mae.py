
modelName = 'mask_0.7'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both']

args = {}
args['outputDir'] = possiblePath_dir[1] + modelName
args['datasetPath'] = possiblePaths_data[3]

args['trial_size'] = (32, 256)
args['patch_size']= (32, 2) #TODO
args['dim'] = 64 #TODO
args['depth'] = 12 #TODO
args['heads'] = 4
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 16
args['dropout'] = 0.1

args['decoder_dim'] = 64
args['masking_ratio'] = 0.7
args['num_decoder_layers'] = 3 #TODO
args['num_decoder_heads'] = 4
args['decoder_dim_head'] = 16

args['batchSize'] = 275

args['weight_decay'] = 1e-5
args['learning_rate'] = 1e-3
args['num_epochs'] = 10000
args['gaussianSmoothWidth'] = 2.0

args['day_specific'] = False
args['day_specific_tokens'] = False

args['extra_notes'] = ("")

args['device'] = 'cuda:3'

from neural_decoder.mae_main import trainModel

trainModel(args)