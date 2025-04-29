
import os
import sys


modelName = 'unidirectional_gru_baseline'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/data/willett_data/ptDecoder_ctc_both', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both',
                      '/home3/skaasyap/willett/data_log_both_held_out_days']

args = {}
args['datasetPath'] = possiblePaths_data[0] # -1 is now held out days 
args['outputDir'] = possiblePath_dir[0] + modelName
args['modelName'] = modelName

if os.path.exists(args['outputDir']):
    print(f"Output directory '{args['outputDir']}' already exists. Press c to continue.")
    breakpoint()
    
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['n_epochs'] = 2000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['input_dropout'] = 0
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['device'] = 'cuda:0'
args['nDays'] = 24
args['testing_on_held_out'] = False
args['restricted_days'] = []
args['maxDay'] = None
args['AdamW'] = False
args['learning_scheduler'] = 'None'
args['load_pretrained_model'] = ''
args['consistency'] = False

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
    input_dropout=args['input_dropout']
).to(args["device"])

trainModel(args, model)