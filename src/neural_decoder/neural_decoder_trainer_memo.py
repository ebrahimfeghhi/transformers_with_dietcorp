import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import getDatasetLoaders
from .augmentations import mask_electrodes
import torch.nn.functional as F
from .loss import forward_cr_ctc, forward_ctc
import time


import wandb

def compute_batch_cer(pred, y, adjustedLens, y_len):
    
    total_edit_distance = 0
    total_seq_length = 0

    for i in range(pred.shape[0]):
        decodedSeq = torch.argmax(pred[i, :adjustedLens[i]], dim=-1)
        decodedSeq = torch.unique_consecutive(decodedSeq)
        decodedSeq = decodedSeq[decodedSeq != 0].cpu().numpy()

        trueSeq = y[i, :y_len[i]].cpu().numpy()

        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
        total_edit_distance += matcher.distance()
        total_seq_length += len(trueSeq)

    cer = total_edit_distance / total_seq_length
    return cer, total_edit_distance, total_seq_length


def trainModel(args, model):
    
    wandb.init(project="MEMO", entity="skaasyap-ucla", config=dict(args),  name=args['modelName'])
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    _, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        1
    )
        
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms
    
    if args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['lrStart'], weight_decay=args['l2_decay'], 
                                    betas=(args['beta1'], args['beta2']))
    if args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lrStart'], 
                    weight_decay=args['l2_decay'])
        
    if args['load_optimizer_state']:
        optimizer.load_state_dict(torch.load(args['optimizer_path']))
        
    
    import copy
    
    original_state_dict = copy.deepcopy(model.state_dict())

    cer_before_memo = []
    cer_after_memo = []
    
    
    total_edit_distance = 0
    total_seq_length = 0
    
    prev_day = 0
    for X, y, X_len, y_len, testDayIdx in testLoader:
        
        if args['evenDaysOnly']:
            if testDayIdx[0] % 2 != 0: 
                continue
        
        X, y, X_len, y_len, testDayIdx = (
            X.to(args["device"]),
            y.to(args["device"]),
            X_len.to(args["device"]),
            y_len.to(args["device"]),
            testDayIdx.to(args["device"]),
        )
        
        if args['next_trial_memo']:
            
            with torch.no_grad():
                
                model.eval()
                pred = model.forward(X, X_len, testDayIdx)
                adjustedLens = model.compute_length(X_len)
                cer, ed, seq_len = compute_batch_cer(pred, y, adjustedLens, y_len)
                total_edit_distance += ed
                total_seq_length += seq_len
                wandb.log({'cer': cer})
            
        if prev_day != testDayIdx[0]:
            
            cer_day = total_edit_distance / total_seq_length
            print(prev_day, cer_day)
            wandb.log({'day_cer': cer_day})
            
    
        if args['memo_augs'] > 0: 
            
            # restore model before next MEMO update
            if args['restore_model_each_update']:
                model.load_state_dict(original_state_dict)
                
            if prev_day != testDayIdx[0]:
                if args['restore_model_each_day']:
                    model.load_state_dict(original_state_dict)
                
            model.train()
    
            for i in range(args['memo_epochs']):
                
                print(f"Epoch: ", i)
                logits_aug = model.forward(X, X_len, testDayIdx, args['memo_augs'])  # [memo_augs, T, D]
                probs_aug = torch.nn.functional.softmax(logits_aug, dim=-1)  # [memo_augs, T, D]
                marginal_probs = probs_aug.mean(dim=0)  # [T, D]
                
                adjustedLens = model.compute_length(X_len)
                marginal_probs = marginal_probs[:adjustedLens]
                
                loss = - (marginal_probs * marginal_probs.log()).sum(dim=-1).mean() # sum across classes, then take mean across time. 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
        if args['next_trial_memo'] == False:
            
            with torch.no_grad():
                
                model.eval()
                pred = model.forward(X, X_len, testDayIdx)
                adjustedLens = model.compute_length(X_len)
                cer, ed, seq_len = compute_batch_cer(pred, y, adjustedLens, y_len)
                total_edit_distance += ed
                total_seq_length += seq_len
                wandb.log({'cer': cer})
                

        prev_day = testDayIdx[0]
        
        
    cer_memo = total_edit_distance / total_seq_length
    print("FINAL VERDICT", cer_memo)
    
    wandb.log({'final_cer_before': cer_memo})
    
    
            
