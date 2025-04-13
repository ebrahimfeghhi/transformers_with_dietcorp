
modelName = 'phoneme_run_1'

possiblePath_dir = ['/data/willett_data/outputs/', 
                    '/home3/skaasyap/willett/outputs/']
possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                      '/data/willett_data/ptDecoder_ctc_both', 
                      '/home3/skaasyap/willett/data', 
                      '/home3/skaasyap/willett/data_log', 
                      '/home3/skaasyap/willett/data_log_both']

args = {}
args['outputDir'] = possiblePath_dir[0] + modelName
args['datasetPath'] = possiblePaths_data[1]

args['patch_size']= (5, 256) #TODO
args['dim'] = 1280 #TODO
args['depth'] = 4 #TODO
args['heads'] = 20
args['mlp_dim_ratio'] = 4 #TODO
args['dim_head'] = 64
args['dropout'] = 0.1

args['whiteNoiseSD'] = 0.8
args['gaussianSmoothWidth'] = 2.0
args['constantOffsetSD'] = 0.2
args['nDays'] = 24
args['nClasses'] = 40
args['batchSize'] = 64

args['l2_decay'] = 1e-5
args['lrStart'] = 0.03
args['lrEnd'] = 0.03

args['look_ahead'] = 0 

args['extra_notes'] = ("")

args['device'] = 'cuda:1'

args['seed'] = 0

args['n_epochs'] = 1000


from neural_decoder.neural_decoder_trainer import trainModel
from neural_decoder.bit import BiT_Phoneme

model = BiT_Phoneme(
    patch_size=args['patch_size'],
    dim=args['dim'],
    depth=args['depth'],
    heads=args['heads'],
    mlp_dim_ratio=args['mlp_dim_ratio'],
    dropout=args['dropout'],
    look_ahead=0,
    nDays=args['nDays'],
    gaussianSmoothWidth=args['gaussianSmoothWidth']
).to(args['device'])

trainModel(args, model)