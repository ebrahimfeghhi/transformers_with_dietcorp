
modelName = 'speechBaseline1'

args = {}
args['outputDir'] = '/Users/ebrahimfeghhi/neural_seq_decoder/output/' + modelName
args['datasetPath'] = '/Users/ebrahimfeghhi/neural_seq_decoder/data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses_list'] = [40, 3, 10, 7, 4, 4, 3] # add one to the classes for SIL
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

import sys
sys.path.append('/Users/ebrahimfeghhi/neural_seq_decoder/src/')
from neural_decoder.neural_decoder_trainer_articulation import trainModel

trainModel(args)