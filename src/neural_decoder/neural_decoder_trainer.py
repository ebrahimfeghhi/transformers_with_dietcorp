import os
import pickle
import time

from edit_distance import SequenceMatcher
import numpy as np
import torch
from tqdm import tqdm

from .dataset import getDatasetLoaders
import torch.nn.functional as F
from .loss import forward_ctc


import wandb


def trainModel(args, model):
    
    if len(args['wandb_id']) > 0:
        
        wandb.init(project="Neural Decoder", entity="skaasyap-ucla", 
                   config=dict(args), name=args['modelName'], 
                   resume="must", id=args["wandb_id"])
    else:
        wandb.init(project="Neural Decoder", 
                   entity="skaasyap-ucla", config=dict(args), name=args['modelName'])
        
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        args['restricted_days'], 
        args['ventral_6v_only']
    )
    
    # Watch the model
    wandb.watch(model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    if args['AdamW']:
        
         optimizer = torch.optim.AdamW(model.parameters(), lr=args['lrStart'], weight_decay=args['l2_decay'], 
                                       betas=(args['beta1'], args['beta2']))
    else:
        print("USING VANILLA ADAM")
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
        
    else:
        
        print("Linear scheduler")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrStart"] / args["lrStart"],
            total_iters=args["n_epochs"],
        )
    
    if len(args['load_pretrained_model']) > 0:
        optimizer_path = os.path.join(args['load_pretrained_model'], 'optimizer')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=args['device']))
        
        scheduler_path = os.path.join(args['load_pretrained_model'], 'scheduler')
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=args['device']))
        print(f"Loaded optimizer and scheduler state from {args['load_pretrained_model']}")
        
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    train_loss = []
    
    for epoch in range(args["start_epoch"], args['n_epochs']):
        
        train_loss = []
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
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        
        with torch.no_grad():
            
            avgTrainLoss = np.mean(train_loss)
            
            model.eval()
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            
            
            for X, y, X_len, y_len, testDayIdx in testLoader:
                
                if testDayIdx.unique().shape[0] == 1 and testDayIdx[0] == 0:
                    testDayIdx.fill_(args['maxDay'])
                
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
            torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer")
            torch.save(scheduler.state_dict(), args['outputDir'] + '/scheduler')
            
        testLoss.append(avgDayLoss)
        testCER.append(cer)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)
            
        scheduler.step()
                    
    wandb.finish()
    return 


'''
# check if cer is within 1% of the lowest CER.
if len(testCER) > 0 and cer < min(testCER) + 0.01: 
    
    lowest_cer_idx = np.argmin(testCER)
    
    # did a previously saved model also lie within the margin?
    if best_cer_ctc_model < min(testCER) + 0.01:
        
        # does this model achieve a lower ctc loss saved model and the current best saved model?
        if avgDayLoss < best_ctc_model_loss and avgDayLoss < testLoss[lowest_cer_idx]:
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights_ctc")
            torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer_ctc")
            torch.save(scheduler.state_dict(), args["outputDir"] + "/scheduler_ctc")
            
            # update metrics
            best_ctc_model_loss = avgDayLoss 
            best_cer_ctc_model = cer
        
    # No previous model saved within CER margin       
    else:
        
        # does this model achieve a lower ctc loss than the best cer model?
        if avgDayLoss < testLoss[lowest_cer_idx]:
            
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights_ctc")
            torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer_ctc")
            torch.save(scheduler.state_dict(), args["outputDir"] + "/scheduler_ctc")
            
            best_ctc_model_loss = avgDayLoss
            best_cer_ctc_model = cer
'''