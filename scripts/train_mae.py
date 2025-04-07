
modelName = 'scratch'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'

args['trial_size']= (100, 256) #TODO
args['patch_size']= (4, 4) #TODO
args['num_classes'] = 40
args['dim'] = 768 #TODO
args['depth'] = 12 #TODO
args['heads'] = 8
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64 
args['dropout'] = 0.1

args['masking_ratio'] = 0.75
args['num_decoder_layers'] = 1 #TODO
args['num_decoder_heads'] = 8
args['num_decoder_dim_head'] = 64

args['device'] = 'cuda:2'

from neural_decoder.mae_main import trainModel

trainModel(args)