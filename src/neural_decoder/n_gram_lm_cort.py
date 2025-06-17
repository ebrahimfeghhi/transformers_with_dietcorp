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

saveFolder_data = "/data/willett_data/paper_results_wer/"
saveFolder_transcripts = "/data/willett_data/model_transcriptions_comp/"

output_file = 'leia'
device = "cuda:2"

if output_file == 'obi':
    model_storage_path = '/data/willett_data/outputs/'
elif output_file == 'leia':
    model_storage_path = '/data/willett_data/leia_outputs/'
    

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
    

models_to_run = ['neurips_transformer_time_masked_held_out_days']


seeds_list = [0,1,2,3]

evaluate_comp = True
use_lm = True

partition = "competition" 
blank_id = 0
num_classes = 41

# no tta
baseline_args = {
    'dropout': 0, 
    'input_dropout': 0, 
    'max_mask_pct': 0, 
    'num_masks': 0, 
    'comp_day_idxs': [0,1,3,4,5]
}

# corp
corp_args = {
    'learning_rate': [1e-3], 
    'comp_day_idxs': [0,1,3,4,5],
    'repeats': 64,
    'adaptation_steps': 1,
    'WN+BS': True,
    'white_noise': 0.2,
    'baseline_shift': 0.05,
    'dropout': 0.35, 
    'input_dropout': 0.2, 
    'l2_decay': 1e-5, 
    'max_mask_pct': 0.075, 
    'num_masks': 20, 
    'freeze_patch': True
}


tta_mode = ['corp', 'baseline']

shared_output_file = ['transformer_held_out_final_dietcorp', 'transformer_held_out_final']
val_save_file = 'transformer_held_out_final_dietcorp'

if tta_mode == 'corp':
    updated_args = corp_args  
else:
    updated_args = baseline_args

skip_models = []


def convert_sentence(s):
    
    s = s.lower()
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 "'", ' ']
    ans = []
    for i in s:
        if(i in charMarks):
            ans.append(i)
    
    return ''.join(ans)

def clean_transcription(text):
    
    """
    Cleans a transcription string by:
    1. Removing leading/trailing whitespace
    2. Removing all characters except letters, hyphens, spaces, and apostrophes
    3. Removing double hyphens
    4. Converting to lowercase
    """
    
    text = str(text).strip()
    text = re.sub(r"[^a-zA-Z\- ']", '', text)
    text = text.replace('--', '')
    return text.lower()

def get_phonemes(thisTranscription):
    
    phonemes = []
    
    for p in g2p(thisTranscription):
        
        if p == ' ':
            phonemes.append('SIL')
        p = re.sub(r'[0-9]', '', p)  # Remove stress
        if re.match(r'^[A-Z]+$', p):  # Only keep phonemes (uppercase only)
            phonemes.append(p)
    
    phonemes.append('SIL')  # Add trailing SIL
    
    PHONE_DEF = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'B',  'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G',
        'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V',
        'W', 'Y', 'Z', 'ZH','SIL'
    ]
    
    PHONE_DEF_SIL = PHONE_DEF + ['SIL']

    phoneme_ids = [PHONE_DEF_SIL.index(p) + 1 for p in phonemes]

    return torch.tensor(phoneme_ids, dtype=torch.long), torch.tensor([len(phoneme_ids)], dtype=torch.long)

def get_data_file(path):
    
    suffix_map = {
        "data_log_both": "/data/willett_data/ptDecoder_ctc_both",
        "data": "/data/willett_data/ptDecoder_ctc",
        "data_log_both_held_out_days": "/data/willett_data/ptDecoder_ctc_both_held_out_days",
        "data_log_both_held_out_days_1": "/data/willett_data/ptDecoder_ctc_both_held_out_days_1",
        "data_log_both_held_out_days_2": "/data/willett_data/ptDecoder_ctc_both_held_out_days_2",
    }
    suffix = path.rsplit('/', 1)[-1]
    return suffix_map.get(suffix, path)

def reverse_dataset(dataset):
    return Subset(dataset, list(reversed(range(len(dataset)))))

def get_dataloader(dataset, batch_size=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=0)

def decode_sequence(pred, adjusted_len):
    pred = torch.argmax(pred[:adjusted_len], dim=-1)
    pred = torch.unique_consecutive(pred)
    return np.array([i for i in pred.cpu().numpy() if i != 0])



day_edit_distance = 0
day_seq_length = 0

for mn, model_name_str in enumerate(models_to_run):
    
    day_cer_dict, total_wer_dict = {}, {}

    for seed in seeds_list:
        
        print(f"Running model: {model_name_str}_seed_{seed}")
        
        day_cer_dict[seed], total_wer_dict[seed] = [], []

        modelPath = f"{model_storage_path}{model_name_str}_seed_{seed}"
        output_file = f"{shared_output_file}_seed_{seed}" if shared_output_file else f"{model_name_str}_seed_{seed}"

        with open(f"{modelPath}/args", "rb") as handle:
            args = pickle.load(handle)
            
        model = BiT_Phoneme(
        patch_size=args['patch_size'], dim=args['dim'], dim_head=args['dim_head'],
        nClasses=args['nClasses'], depth=args['depth'], heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'], dropout=0, input_dropout=0,
        look_ahead=args['look_ahead'], gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'], max_mask_pct=max_mask_pct,
        num_masks=num_masks, mask_token_zeros=args['mask_token_zero'], max_mask_channels=0,
        num_masks_channels=0, dist_dict_path=None
        ).to(device)

        data_file = get_data_file(args['datasetPath'])

        trainLoaders, testLoaders, loadedData = getDatasetLoaders(data_file, 8)
        args.setdefault('mask_token_zero', False)

        model.load_state_dict(torch.load(f"{modelPath}/modelWeights", map_location=device), strict=True)
        model.eval()

        optimizer = torch.optim.AdamW(model.parameters(), lr=memo_lr[mn], weight_decay=0,
                                      betas=(args['beta1'], args['beta2']))

        for name, p in model.named_parameters():
            p.requires_grad = name in {
                "to_patch_embedding.1.weight", "to_patch_embedding.1.bias",
                "to_patch_embedding.2.weight", "to_patch_embedding.2.bias",
                "to_patch_embedding.3.weight", "to_patch_embedding.3.bias"
            }

        testDayIdxs = np.arange(5)
        valDayIdxs = [0, 1, 3, 4, 5] if mn == 2 else [0, 1, 2, 3, 4]

        model_outputs = {"logits": [], "logitLengths": [], "trueSeqs": [], "transcriptions": []}
        
        total_edit_distance = total_seq_length = 0
        nbest_outputs = []
        nbest_outputs_val = []
        
        for i, testDayIdx in enumerate(testDayIdxs):
            
            ve = valDayIdxs[i]
            val_ds = reverse_dataset(SpeechDataset([loadedData['test'][ve]]))
            test_ds = reverse_dataset(SpeechDataset([loadedData['competition'][i]]))
            combined_ds = ConcatDataset([val_ds, test_ds])
            data_loader = get_dataloader(combined_ds)

            if tta:
                
                for trial_idx, (X, y, X_len, y_len, _) in enumerate(data_loader):
                                    
                    total_start = time.time()

                    X, y, X_len, y_len = map(lambda x: x.to(device), [X, y, X_len, y_len])
                    dayIdx = torch.tensor([ve], dtype=torch.int64).to(device)

                    model.train()
                    memo_loss = li_loss = torch.tensor(0.0, device=device)

                    for _ in range(tta_epochs):
                        
                        adjusted_len = model.compute_length(X_len)
    
                        logits = model(X, X_len, ve, memo_augs, nptl_augs, nptl_aug_params) # (augs x T x C)
                        logits_np = logits.detach().cpu().numpy()
                        logits_np = np.concatenate([
                            logits_np[:, :, 1:],   # classes 1 to C-1
                            logits_np[:, :, 0:1]   # class 0, preserved in its own dimension
                        ], axis=-1)
                        logits_np = lmDecoderUtils.rearrange_speech_logits(logits_np, has_sil=True)
                        
                        #if run_memo:
                        #    memo_loss = memo_loss_from_logits(logits_aug, adjusted_len, blank_id)

                        #if run_lang_informed:
                    
                        decoded = lmDecoderUtils.lm_decode(
                            ngramDecoder, logits_np[0],
                            blankPenalty=blank_penalty,
                            returnNBest=return_n_best, rescore=rescore
                        )
                        