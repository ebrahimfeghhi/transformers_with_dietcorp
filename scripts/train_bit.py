args = {}
modelName = 'BiT'
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.006
args['lrEnd'] = 0.006
args['patch_height'] = 32
args['patch_width'] = 16
args['n_heads'] = 64
args['mlp_ratio'] = 4
args['embedding_dim'] = 1024
args['nBatch'] = 30000 #3000
args['n_layers'] = 3
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['l2_decay'] = 1e-5
args['device'] = 'cuda:2'

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)