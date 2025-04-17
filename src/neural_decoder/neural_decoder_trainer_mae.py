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
from .dataset import getDatasetLoaders_MAE
from .augmentations import mask_electrodes

import wandb

def trainModel(args, model):
    
    wandb.init(project="Neural Decoder with MAE", 
               entity="skaasyap-ucla", config=dict(args))
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders_MAE(
        args["datasetPath"],
        args["batchSize"],
    )
        
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    #optimizer = torch.optim.Adam(
    #    model.parameters(),
    #    lr=args["lrStart"],
    #    betas=(0.9, 0.999),
    #    eps=0.1,
    #    weight_decay=args["l2_decay"],
    #)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lrStart'], weight_decay=args['l2_decay'])
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["n_epochs"],
    )
    
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for epoch in range(args['n_epochs']):
        
        train_loss = []
        train_mae_loss = []
        train_mae_r2 = []
        
        
        model.train()
        
        for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc="Training")):
                            
            X, y, X_len, y_len, dayIdx = (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                dayIdx.to(args["device"]),
            )
            
            
            # Compute prediction error
            mae_loss, mae_r2, pred = model.forward(X, X_len, dayIdx)
            
            adjustedLens = model.compute_length(X_len)

            #loss = loss_ctc(
            #    torch.permute(pred.log_softmax(2), [1, 0, 2]),
            #    y,
            #   adjustedLens,
            #    y_len,
            #)
            
            total_loss = mae_loss
        
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            #train_loss.append(loss.cpu().detach().numpy())
            train_loss.append(0)
            train_mae_loss.append(mae_loss.cpu().detach().numpy())
            train_mae_r2.append(mae_r2.cpu().detach().numpy())
            # print(endTime - startTime)
            
        print(np.mean(train_mae_r2))
        breakpoint()

        with torch.no_grad():
            
            avgTrainLoss = np.mean(train_loss)
            avgTrainMaeLoss = np.mean(train_mae_loss)
            avgTrainR2 = np.mean(train_mae_r2)
            
            
            model.eval()
            allLoss = []
            allMaeLoss = []
            allR2 = []
            total_edit_distance = 0
            total_seq_length = 0
            
            for X, y, X_len, y_len, testDayIdx in testLoader:
                
                X, y, X_len, y_len, testDayIdx = (
                    X.to(args["device"]),
                    y.to(args["device"]),
                    X_len.to(args["device"]),
                    y_len.to(args["device"]),
                    testDayIdx.to(args["device"]),
                )

                mae_loss, mae_r2, pred = model.forward(X, X_len, testDayIdx)
                
                adjustedLens = model.compute_length(X_len)
                
                loss = loss_ctc(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y,
                    adjustedLens,
                    y_len,
                )
                
                allLoss.append(loss.cpu().detach().numpy())
                allMaeLoss.append(mae_loss.cpu().detach().numpy())
                allR2.append(mae_r2.cpu().detach().numpy())
                

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
            avgDayMaeLoss = np.mean(allMaeLoss)
            avgDayR2 = np.mean(allR2)
            
            cer = total_edit_distance / total_seq_length

            endTime = time.time()
            print(
                f"Epoch {epoch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
            )
                
            # Log the metrics to wandb
            wandb.log({
                "train_ctc_Loss": avgTrainLoss,
                "train_mae_Loss": avgTrainMaeLoss,
                "train_mae_R2": avgTrainR2,
                "ctc_loss": avgDayLoss,
                "mae_loss": avgDayMaeLoss,
                "mae_r2": avgDayR2,
                "cer": cer,
                "time_per_epoch": (endTime - startTime) / 100
            })
            
            startTime = time.time()

        if len(testCER) > 0 and cer < np.min(testCER):
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            
        testLoss.append(avgDayLoss)
        testCER.append(cer)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)


