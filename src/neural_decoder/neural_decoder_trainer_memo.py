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

from .model import GRUDecoder
from .dataset import getDatasetLoaders
from .augmentations import mask_electrodes
import torch.nn.functional as F
from .loss import forward_cr_ctc, forward_ctc


import wandb

def trainModel(args, model):
    
    wandb.init(project="MEMO", entity="skaasyap-ucla", config=dict(args))
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    _, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        memo=True # ensure batch size is 1 
    )
        
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lrStart'], weight_decay=args['l2_decay'], 
                                betas=(args['beta1'], args['beta2']))
        
    if args['load_optimizer_state']:
        
        optimizer.load_state_dict(torch.load(args['optimizer_path']))
        
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    
    allLoss = []
    
    for X, y, X_len, y_len, testDayIdx in testLoader:
        
        if args['memo_augs'] > 0: 
            
            model.train()
            
            X, y, X_len, y_len, testDayIdx = (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                testDayIdx.to(args["device"]),
            )
            
            for i in range(args['memo_epochs']):
                
                X_augmented = []
            
                for aug in range(args['memo_augs']):
                    X_aug, _ = model.apply_spec_augment_mask(X)
                    X_augmented.append(X_aug)
                
                X_augmented = torch.stack(X_augmented)
            
                # Expand metadata to match augmented batch size
                X_len_aug = X_len.repeat(args['memo_augs'])         # [memo_augs]
                testDayIdx_aug = testDayIdx.repeat(args['memo_augs'])        
                
                logits_aug = model.forward(X_augmented, X_len_aug, testDayIdx_aug)  # [memo_augs, T, D]
                probs_aug = torch.nn.functional.softmax(logits_aug, dim=-1)  # [memo_augs, T, D]
                marginal_probs = probs_aug.mean(dim=0)  # [T, D]
                loss = - (marginal_probs * marginal_probs.log()).sum(dim=-1).mean() # sum across classes, then take mean across time. 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
         # === STEP 6: Evaluate on unaugmented input ===
        model.eval()
        
        with torch.no_grad():
            
            pred = model.forward(X, X_len, testDayIdx)  # [1, T, D]

            adjustedLens = model.compute_length(X_len)
            
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                adjustedLens,
                y_len,
            )
            
            allLoss.append(loss.cpu().detach().numpy())

            for iterIdx in range(pred.shape[0]):
                decodedSeq = torch.argmax(
                    torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

            avgDayLoss = np.mean(allLoss)
            cer = total_edit_distance / total_seq_length

            endTime = time.time()
            print(
                f"Cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
            )
                
            # Log the metrics to wandb
            wandb.log({
                "ctc_loss": avgDayLoss,
                "cer": cer,
                "time_per_epoch": (endTime - startTime) / 100, 
            })
            
            startTime = time.time()





