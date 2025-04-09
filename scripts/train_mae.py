
modelName = 'scratch'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'

args['trial_size'] = (200, 256)
args['patch_size']= (10, 16) #TODO
args['dim'] = 128 #TODO
args['depth'] = 6 #TODO
args['heads'] = 10
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 16
args['dropout'] = 0.1

args['decoder_dim'] = 160
args['masking_ratio'] = 0.05
args['num_decoder_layers'] = 2 #TODO
args['num_decoder_heads'] = 10
args['num_decoder_dim_head'] = 16

args['batchSize'] = 16

args['weight_decay'] = 1e-5
args['learning_rate'] = 1e-4
args['num_epochs'] = 10000


args['device'] = 'cuda:0'

from neural_decoder.mae_main import trainModel

trainModel(args)