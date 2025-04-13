
modelName = 'original_model'

args = {}
args['outputDir'] = '/data/willett_data/outputs/' + modelName
args['datasetPath'] = '/data/willett_data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['n_epochs'] = 30000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['device'] = 'cuda:0'
args['nDays'] = 24

from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.model import GRUDecoder

model = GRUDecoder(
    neural_dim=args["nInputFeatures"],
    n_classes=args["nClasses"],
    hidden_dim=args["nUnits"],
    layer_dim=args["nLayers"],
    nDays=args['nDays'],
    dropout=args["dropout"],
    device=args["device"],
    strideLen=args["strideLen"],
    kernelLen=args["kernelLen"],
    gaussianSmoothWidth=args["gaussianSmoothWidth"],
    bidirectional=args["bidirectional"],
).to(args["device"])

trainModel(args, model)