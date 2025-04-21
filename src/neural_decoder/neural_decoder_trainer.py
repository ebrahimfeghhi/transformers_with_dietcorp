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


def lr_lambda(epoch):
    if epoch < 200:
        return 1.0    # 1e-3
    elif epoch < 300:
        return 0.5    # 5e-4
    elif epoch < 400:
        return 0.1    # 1e-4
    elif epoch < 500:
        return 0.05   # 5e-5
    else:
        return 0.01   # 1e-5


def trainModel(args, model):
    
    wandb.init(project="Neural Decoder", entity="skaasyap-ucla", config=dict(args))
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
        
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    if args['AdamW']:
        
         optimizer = torch.optim.AdamW(model.parameters(), lr=args['lrStart'], weight_decay=args['l2_decay'], 
                                       betas=(args['beta1'], args['beta2']))
    else:
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
        
    if args['learning_scheduler'] == 'multistep': 

        print("Multistep scheduler")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])
        
    elif args['learning_scheduler'] == 'cosine':
        
        print("Cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['n_epochs'],     # Total epochs to decay over
            eta_min=args['lrEnd']    # Final learning rate
        )
            
    elif args['learning_scheduler'] == 'warmcosine':
        
        print("Warm Cosine Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args['T_0'],       # first cosine decay cycle
            T_mult=args['T_mult'],      # next cycle is 1000 long (up to 1500)
            eta_min=args['lrEnd']
        )
        
    elif args['learning_scheduler'] == 'custom':
        print("Custom scheduler")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
    else:
        
        print("Linear scheduler")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrStart"] / args["lrStart"],
            total_iters=args["n_epochs"],
        )
    
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for epoch in range(args['n_epochs']):
        
        train_loss = []
        train_kl_loss = []
        model.train()
        
        for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc="Training")):
                            
            X, y, X_len, y_len, dayIdx = (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                dayIdx.to(args["device"]),
            )

            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                    * args["constantOffsetSD"]
                )

            # Compute prediction error
            pred = model.forward(X, X_len, dayIdx)
                        
            adjustedLens = model.compute_length(X_len)

                
            loss = forward_ctc(pred, adjustedLens, y, y_len)
            train_loss.append(loss.cpu().detach().numpy())
            

            #loss = torch.sum(loss)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            
        
            # print(endTime - startTime)


        with torch.no_grad():
            
            avgTrainLoss = np.mean(train_loss)
            
            model.eval()
            allLoss = []
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

                pred = model.forward(X, X_len, testDayIdx)
                
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
                f"Epoch {epoch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
            )
                
            # Log the metrics to wandb
            wandb.log({
                "train_ctc_Loss": avgTrainLoss,
                "ctc_loss": avgDayLoss,
                "cer": cer,
                "time_per_epoch": (endTime - startTime) / 100, 
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
            
        scheduler.step()


