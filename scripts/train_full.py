
import os
import sys
import torch

num_seeds = 2
start = 2

for seed in range(start, num_seeds+start):

    modelName = f'masked_transformer_{seed}'

    possiblePath_dir = ['/data/willett_data/outputs/', 
                        '/home3/skaasyap/willett/outputs/']
    
    possiblePaths_data = ['/data/willett_data/ptDecoder_ctc', 
                        '/data/willett_data/ptDecoder_ctc_both', 
                        '/home3/skaasyap/willett/data', 
                        '/home3/skaasyap/willett/data_log_both',
                        '/home3/skaasyap/willett/data_log_both_held_out_days']

    args = {}
    args['outputDir'] = possiblePath_dir[0] + modelName
    args['datasetPath'] = possiblePaths_data[1] # -1 is now held out days 
    args['modelName'] = modelName

    args['testing_on_held_out'] = False # set to true if using held_out_days split
    args['maxDay'] = 15 # only applies if testing_on_held_out is true
    args['restricted_days'] = [] # only uses restricted_days 

    if os.path.exists(args['outputDir']):
        print(f"Output directory '{args['outputDir']}' already exists. Press c to continue.")
        breakpoint()
        
    # model related parameters
    args['patch_size']= (5, 256) #TODO
    args['dim'] = 384 #TODO
    args['depth'] = 7 #TODO
    args['heads'] = 6
    args['mlp_dim_ratio'] = 4 #TODO
    args['dim_head'] = 64
    args['T5_style_pos'] = True
    args['nClasses'] = 40

    # other overfitting stuff
    args['whiteNoiseSD'] = 0.2
    args['gaussianSmoothWidth'] = 2.0
    args['constantOffsetSD'] = 0.05
    args['num_masks'] = 20
    args['max_mask_pct'] = 0.075
    args['l2_decay'] = 1e-5
    args['input_dropout'] = 0.2
    args['dropout'] = 0.35
    
    # learning stuff 
    args['AdamW'] = True
    args['learning_scheduler'] = 'multistep'
    args['gamma'] = 0.1 # factor by which to drop the learning rate at milestone 
    args['lrStart'] = 0.001
    args['lrEnd'] = 0.001
    args['batchSize'] = 64
    args['beta1'] = 0.90
    args['beta2'] = 0.999
    
    # whether to do actual epochs or just sample batches 
    args['batchStyle'] = False
    args['nBatch'] = 276000
    args['n_epochs'] = 600
    # number of epochs/batches after which to drop the learning rate
    if args['batchStyle']:
        args['milestones'] = [552] 
        args['early_stop'] = float('inf')
    else:
        args['milestones'] = [400] 
        args['early_stop'] = float('inf')
        
    args['look_ahead'] = 0 
    args['extra_notes'] = ("")
    args['device'] = 'cuda:0'
    args['seed'] = seed

    args['load_pretrained_model'] = '' # empty string to not load any previous models. 
    
        
    from neural_decoder.neural_decoder_trainer import trainModel
    from neural_decoder.bit import BiT_Phoneme

    model = BiT_Phoneme(
        patch_size=args['patch_size'],
        dim=args['dim'],
        dim_head=args['dim_head'], 
        nClasses=args['nClasses'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        look_ahead=args['look_ahead'],
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'], 
        max_mask_pct=args['max_mask_pct'], 
        num_masks=args['num_masks']
    ).to(args['device'])

    if len(args['load_pretrained_model']) > 0:
        checkpoint = torch.load(f"{args['load_pretrained_model']}/modelWeights", map_location=args['device'])
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded pretrained model from {args['load_pretrained_model']}")

    trainModel(args, model)