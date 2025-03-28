
modelName = 'scratch'

args = {}
args['outputDir'] = '/home3/skaasyap/willett/neural_seq_decoder/output/' + modelName
args['datasetPath'] = '/home3/skaasyap/willett/data'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 16
args['lrStart'] = 0.002
args['lrEnd'] = 0.002
args['nUnits'] = 1024
args['nBatch'] = 30000 #3000
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
args['device'] = 'cuda:2'

args['latentspeech_dim_stage2'] = 1024
args['outputdim_stage2'] = 256
args['nLayers_stage2'] = 6
args['num_heads'] = 8
args['dropout_stage2'] = 0.4
args['nUnits_stage2'] = 512
args['l2_decay_stage2'] = 1e-5
args['dropout_stage2'] = 0.4
args["bucket_size"] = 8000 # half a second of data (audio is in 16Khz)
args["min_samples_in_bucket"] = 8


from neural_decoder.speech2neural_transformer import trainModel

trainModel(args)