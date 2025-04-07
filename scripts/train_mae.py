
modelName = 'scratch'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'

args['patch_size']= (4, 16) #TODO
args['dim'] = 4*16 #TODO
args['depth'] = 6 #TODO
args['heads'] = 10
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 32
args['dropout'] = 0.1

args['decoder_dim'] = 64
args['encoder_dim'] = 64
args['masking_ratio'] = 0.75
args['num_decoder_layers'] = 2 #TODO
args['num_decoder_heads'] = 8
args['num_decoder_dim_head'] = 32

args['batchSize'] = 8

args['weight_decay'] = 1e-5
args['learning_rate'] = 0.005
args['num_epochs'] = 5


args['device'] = 'cuda:2'

from neural_decoder.mae_main import trainModel

trainModel(args)