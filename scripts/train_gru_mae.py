
modelName = 'scratch'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['learning_rate'] = 0.005
args['nUnits'] = 1024
args['num_epochs'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 1024
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0 
args['constantOffsetSD'] = 0 
args['gaussianSmoothWidth'] = 0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['weight_decay'] = 1e-5
args['device'] = 'cuda:0'

from neural_decoder.mae_main import trainModel

trainModel(args)