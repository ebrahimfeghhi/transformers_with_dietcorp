import re
import time
import pickle
import numpy as np
import sys
from edit_distance import SequenceMatcher
import torch
from dataset import SpeechDataset
import matplotlib.pyplot as plt
from neural_decoder.dataset import getDatasetLoaders
import neural_decoder.lm_utils as lmDecoderUtils
from neural_decoder.model import GRUDecoder
import pickle
import argparse
import matplotlib.pyplot as plt
from neural_decoder.dataset import getDatasetLoaders
import neural_decoder.lm_utils as lmDecoderUtils
from neural_decoder.lm_utils import build_llama_1B
from neural_decoder.model import GRUDecoder
from neural_decoder.bit import BiT_Phoneme
import pickle
import argparse
from lm_utils import _cer_and_wer
import json
import os
import copy
from torch.utils.data import ConcatDataset
from loss import memo_loss_from_logits, forward_ctc
from collections import deque

import wandb
import math

from tta_utils import convert_sentence, compute_lambda, clean_transcription, get_phonemes, get_data_file, reverse_dataset, get_dataloader, decode_sequence



saveFolder_data = "/home3/ebrahim/paper_results_wer/"
saveFolder_transcripts = "/data/willett_data/model_transcriptions_comp/"

base_dir = "/home3/skaasyap/willett"

load_lm = True

# LM decoding hyperparameters
acoustic_scale = 0.8
blank_penalty = np.log(2)

run_for_llm = False

if run_for_llm:
    return_n_best = True
    rescore = False
    nbest = 100
    print("RUNNING IN LLM MODE")
else:
    return_n_best = False
    rescore = False
    nbest = 1
    print("RUNNING IN N-GRAM MODE")
    
if load_lm and 'ngramDecoder' not in globals():
        
    lmDir = base_dir +'/lm/languageModel'
    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir,
        acoustic_scale=acoustic_scale, #1.2
        nbest=nbest,
        beam=18
    )
    print("loaded LM")
    
elif load_lm:
    print("Already loaded LM")
    
#models_to_run = ['gru_held_out_days_redo']
all_models = ['transformer_shortened_held_out_days_big',
              'transformer_held_out_big_0', 
              'transformer_short_held_out_normal', 
              'gru_shortened', 
              'bidirectional_gru_time_masked']

idx = -1
models_to_run = [all_models[idx]]
print(models_to_run)

if idx == 0 or idx == 1:
    data_file = '/home3/ebrahim/data/ptDecoder_ctc_both_held_out_days_big_0'
elif idx == 2 or idx == -1:
    data_file = '/home3/ebrahim/data/ptDecoder_ctc_both_held_out_days'
else:
    data_file = None

shared_output_file = 'transformer_held_out_more_dietcorp_updated'
val_save_file = 'SCRATCH'

output_file = 'leia'
device = "cuda:2"

if output_file == 'obi':
    model_storage_path = '/data/willett_data/outputs/'
elif output_file == 'leia':
    if idx == 0 or idx == 1 or idx == -1:
        model_storage_path = '/home3/ebrahim/obi_outputs/'
    else:
        model_storage_path = '/home3/skaasyap/willett/outputs/'

seeds_list = [0]

if len(shared_output_file) > 0:
    write_mode = "a"
else:
    write_mode = "w"
    

blank_id = 0
num_classes = 41

# no tta
baseline_args = {
    'dropout': 0, 
    'input_dropout': 0, 
    'max_mask_pct': 0, 
    'num_masks': 0, 
    'gru': False, 
    'max_day': 14,
    'repeats': [1]
}

# corp
corp_args = {
    'learning_rate': [1e-3], 
    'repeats': [64],
    'adaptation_steps': 1,
    'WN+BS': True,
    'white_noise': 0.2,
    'baseline_shift': 0.05,
    'dropout': 0.35, 
    'input_dropout': 0.2, 
    'l2_decay': 1e-5, 
    'max_mask_pct': 0.075, 
    'num_masks': 20, 
    'freeze_patch': True,
    'freeze_linear': True,
    'gru': True, 
    'max_day': None
}

tta_mode = 'corp'

if tta_mode == 'corp':
    updated_args = corp_args  
else:
    updated_args = baseline_args

skip_models = []
skip_seeds = []

def get_lm_outputs(tf_logits):
    
    # prepare logits for n-gram language model decoding 
    logits_np = tf_logits.detach().cpu().numpy()
    logits_np = np.concatenate([
        logits_np[:, :, 1:],   # classes 1 to C-1
        logits_np[:, :, 0:1]   # class 0, preserved in its own dimension
    ], axis=-1)
    
    logits_np = lmDecoderUtils.rearrange_speech_logits(logits_np, has_sil=True)
    
    # obtain sentence from n-gram language model 
    decoded = lmDecoderUtils.lm_decode(
        ngramDecoder, logits_np[0],
        blankPenalty=blank_penalty,
        returnNBest=return_n_best, rescore=rescore
    )

    decoded = clean_transcription(decoded)
    
    y_pseudo, y_len_pseudo = get_phonemes(decoded)
    
    return decoded, y_pseudo, y_len_pseudo

times_ms = []

for n_augs in updated_args['repeats']:
    for mn, model_name_str in enumerate(models_to_run):
        if model_name_str in skip_models:
            continue

        day_wer_dict, total_wer_dict = {}, {}

        for seed in seeds_list:
            if seed in skip_seeds:
                continue

            print(f"Running model: {model_name_str}_seed_{seed}")
            day_wer_dict[seed] = []

            modelPath = f"{model_storage_path}{model_name_str}_seed_{seed}"
            output_file = (
                f"{shared_output_file}_seed_{seed}"
                if shared_output_file
                else f"{model_name_str}_seed_{seed}"
            )

            # Load args
            with open(f"{modelPath}/args", "rb") as handle:
                args = pickle.load(handle)
                
                    # obtain beam search + LM corrected outputs
            # do this before adaptation on that trial to make 
            # sure results are compatabile with a streaming system 
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            # Build model (dedented to the same level as 'with')
            if updated_args['gru']:
                
                model = GRUDecoder(
                    neural_dim=args["nInputFeatures"],
                    n_classes=args["nClasses"],
                    hidden_dim=args["nUnits"],
                    layer_dim=args["nLayers"],
                    nDays=args["nDays"],
                    dropout=args["dropout"],
                    device=device,
                    strideLen=args["strideLen"],
                    kernelLen=args["kernelLen"],
                    gaussianSmoothWidth=args["gaussianSmoothWidth"],
                    bidirectional=args['bidirectional'],
                    input_dropout=args["input_dropout"],
                    max_mask_pct=args["max_mask_pct"],
                    num_masks=args["num_masks"]
                ).to(device)
                
                
                
            else:
                
                model = BiT_Phoneme(
                    patch_size=args['patch_size'], dim=args['dim'], dim_head=args['dim_head'],
                    nClasses=args['nClasses'], depth=args['depth'], heads=args['heads'],
                    mlp_dim_ratio=args['mlp_dim_ratio'], dropout=updated_args['dropout'], input_dropout=updated_args['input_dropout'],
                    gaussianSmoothWidth=args['gaussianSmoothWidth'],
                    T5_style_pos=args['T5_style_pos'], max_mask_pct=updated_args['max_mask_pct'],
                    num_masks=updated_args['num_masks'], mask_token_zeros=args['mask_token_zero'], max_mask_channels=0,
                    num_masks_channels=0, dist_dict_path=None
                ).to(device)

            if data_file is None:
                data_file = args['datasetPath']

            trainLoader, testLoaders, loadedData = getDatasetLoaders(data_file, 64)
                    
            args.setdefault('mask_token_zero', False)

            model.load_state_dict(torch.load(f"{modelPath}/modelWeights", map_location=device), strict=True)

            if tta_mode != 'baseline':
                
                print(updated_args['learning_rate'][mn])
                optimizer = torch.optim.AdamW(model.parameters(), lr=updated_args['learning_rate'][mn], 
                                            weight_decay=updated_args['l2_decay'],
                                                betas=(args['beta1'], args['beta2']))

                if updated_args['freeze_linear'] and updated_args['gru']:
                    for name, p in model.named_parameters():
                        p.requires_grad = name in {
                            "dayWeights", "dayBias"
                        }
                        
                if updated_args['freeze_patch'] and updated_args['gru'] == False:
                    for name, p in model.named_parameters():
                        p.requires_grad = name in {
                            "to_patch_embedding.1.weight", "to_patch_embedding.1.bias",
                            "to_patch_embedding.2.weight", "to_patch_embedding.2.bias",
                            "to_patch_embedding.3.weight", "to_patch_embedding.3.bias"
                        }

            testDayIdxs = np.arange(len(loadedData['test']))
            print(len(testDayIdxs))
            
                
            model_outputs = {"logits": [], "logitLengths": [], "trueSeqs": [], "transcriptions": []}
          
            decoded_list_all_days = []
            transcripts_all_days = []
            
            for test_day_idx, testDayIdx in enumerate(testDayIdxs):
                
                print("day ", test_day_idx)
            
                val_ds = SpeechDataset([loadedData['test'][test_day_idx]], return_transcript=True)
                data_loader = get_dataloader(val_ds)                        
                transcriptions_list = []
                decoded_list = []
                
                test_day_decoded_sents = []
                
                for trial_idx, (X, y, X_len, y_len, day_idx, transcript) in enumerate(data_loader):
                                              
                    total_start = time.time()
                    
                    X, y, X_len, y_len = map(lambda x: x.to(device), [X, y, X_len, y_len])
                    
                    if updated_args['max_day'] is not None:
                        day_idx = torch.tensor([updated_args['max_day']], dtype=torch.int64).to(device)
                    else:
                        day_idx = torch.tensor([day_idx],  dtype=torch.int64).to(device)
                        
                    adjusted_len = model.compute_length(X_len)
                    
                    model.eval()
                    logits_eval = model(X, X_len, day_idx)
                    decoded, y_pseudo, y_len_pseudo = get_lm_outputs(logits_eval)
                    
                    if tta_mode != 'baseline':
                    
                        # generate multiple versions of the same input
                        if n_augs > 0:
                            
                            X = X.repeat(n_augs, 1, 1)
                            y = y.repeat(n_augs, 1)
                            y_len = y_len.repeat(n_augs)
                            X_len = X_len.repeat(n_augs)
                            adjusted_len = adjusted_len.repeat(n_augs)
                            y_pseudo = y_pseudo.unsqueeze(0).repeat(n_augs, 1).to(device) 
                            y_len_pseudo = y_len_pseudo.repeat(n_augs).to(device)
                            
                        
                        # add white noise and baseline shift augmentations to each sample
                        if updated_args['WN+BS'] == True:
                            
                            X += torch.randn(X.shape, 
                                        device=device) * updated_args['white_noise']
                        
                            X += (
                                torch.randn([X.shape[0], 1, X.shape[2]], 
                                device=device)
                                * updated_args['baseline_shift']
                            )      
                        
                        model.train()
                        
                        for _ in range(updated_args['adaptation_steps']):
                            
                            #torch.cuda.synchronize(device)
                            #t0 = time.perf_counter()
                    
                            logits = model(X, X_len, day_idx)
                            corp_loss = forward_ctc(logits, adjusted_len, y_pseudo, y_len_pseudo)
                            optimizer.zero_grad()
                            corp_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                            optimizer.step()
                            #torch.cuda.synchronize(device)
                            #t1 = time.perf_counter()
                            #times_ms.append((t1 - t0) * 1000.0)


                    model.eval()
                    
                    decoded_list.append(decoded)
                    transcriptions_list.append(clean_transcription(transcript[0]))
                    
                print(f"mean: {np.mean(times_ms):.2f} ms | std: {np.std(times_ms):.2f} ms | n={len(times_ms)}")

                to_gib = 1024**3
                torch.cuda.synchronize(device)
                peak_alloc_gib = torch.cuda.max_memory_allocated(device) / to_gib      # tensors/activations
                peak_res_gib  = torch.cuda.max_memory_reserved(device) / to_gib       # allocator footprint

                print(f"PyTorch peak allocated: {peak_alloc_gib:.3f} GiB")
                print(f"PyTorch peak reserved : {peak_res_gib:.3f} GiB")
                
                breakpoint()