
modelName = 'scratch'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/home3/skaasyap/willett/data']

args = {}
args['outputDir'] = possiblePath_dir[1] + modelName
args['datasetPath'] = possiblePaths_data[1]

args['trial_size'] = (100, 256)
args['patch_size']= (100, 2) #TODO
args['dim'] = 200 #TODO
args['depth'] = 10 #TODO
args['heads'] = 10
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 20
args['dropout'] = 0.1

args['decoder_dim'] = 200
args['masking_ratio'] = 0.5
args['num_decoder_layers'] = 2 #TODO
args['num_decoder_heads'] = 20
args['decoder_dim_head'] = 10

args['batchSize'] = 256

args['weight_decay'] = 1e-5
args['learning_rate'] = 1e-3
args['num_epochs'] = 10000
args['gaussianSmoothWidth'] = 2.0



args['device'] = 'cuda:1'

from neural_decoder.mae_main import trainModel

trainModel(args)