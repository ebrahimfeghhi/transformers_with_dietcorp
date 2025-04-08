
modelName = 'scratch'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'

args['patch_size']= (4, 16) #TODO
args['dim'] = 64 #TODO
args['depth'] = 6 #TODO
args['heads'] = 4
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 16
args['dropout'] = 0.1

args['decoder_dim'] = 64
args['masking_ratio'] = 0.75
args['num_decoder_layers'] = 2 #TODO
args['num_decoder_heads'] = 4
args['num_decoder_dim_head'] = 16

args['batchSize'] = 8

args['weight_decay'] = 1e-5
args['learning_rate'] = 0.005
args['num_epochs'] = 10000


args['device'] = 'cuda:0'

from neural_decoder.mae_main import trainModel

trainModel(args)