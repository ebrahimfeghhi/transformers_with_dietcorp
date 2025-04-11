
modelName = 'mask_0.7'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both']

args = {}
args['outputDir'] = possiblePath_dir[0] + modelName
args['datasetPath'] = possiblePaths_data[0]

args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 3
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0
args['constantOffsetSD'] = 0
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

args['freeze_mae_encoder'] = True

args['weight_decay'] = 1e-5
args['learning_rate'] = 1e-3
args['num_epochs'] = 10000
args['gaussianSmoothWidth'] = 2.0

args['decoder_dim'] = 64
args['masking_ratio'] = 0.7
args['num_decoder_layers'] = 3 #TODO
args['num_decoder_heads'] = 4
args['decoder_dim_head'] = 16


args['trial_size'] = (32, 256)
args['patch_size']= (32, 2) #TODO
args['dim'] = 64 #TODO
args['depth'] = 12 #TODO
args['heads'] = 4
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 16
args['dropout'] = 0.1

args['best_model_path'] = possiblePath_dir[0] + modelName + '/save_best.pth'

args['extra_notes'] = ("")

args['device'] = 'cuda:1'

from neural_decoder.mae_phoneme_main import trainModel

trainModel(args)