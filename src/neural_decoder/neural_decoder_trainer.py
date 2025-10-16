import os
import pickle
import time

from edit_distance import SequenceMatcher
import numpy as np
import torch
from tqdm import tqdm

from .dataset import getDatasetLoaders
import torch.nn.functional as F
from .loss import forward_ctc, forward_cr_ctc


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
    testCER2 = []
    startTime = time.time()
    train_loss = []
    train_kl_loss = []
    
    for epoch in range(args["start_epoch"], args['n_epochs']):
        
        train_loss = []
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(trainLoader, desc="Training")):
           
            # Base case: always unpack the first 5
            X, y, X_len, y_len, dayIdx = batch[:5]

            # Send to device
            X      = X.to(args["device"])
            y      = y.to(args["device"])
            X_len  = X_len.to(args["device"])
            y_len  = y_len.to(args["device"])
            dayIdx = dayIdx.to(args["device"])

            if args.get("nClasses_2") is not None:
                y2, y2_len = batch[5], batch[6]
                y2     = y2.to(args["device"])
                y2_len = y2_len.to(args["device"])

            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                    * args["constantOffsetSD"]
                )
                
            adjustedLens = model.compute_length(X_len)
            
            # Compute prediction error
            if args.get('nClasses_2') is not None:
                pred, pred2 = model.forward(X, X_len, dayIdx)
                loss1 = forward_ctc(pred, adjustedLens, y, y_len)
                loss2 = forward_ctc(pred2, adjustedLens, y2, y2_len)
                loss = loss1 + loss2
                
            else:
                
                pred = model.forward(X, X_len, dayIdx)
                
                if args['consistency']:
                    
                    ctc_loss, kl_loss = forward_cr_ctc(pred, adjustedLens.repeat(2), 
                                                   torch.cat([y, y], dim=0), y_len.repeat(2))
                    ctc_loss = ctc_loss*0.5
                    kl_loss = kl_loss*0.5

                    train_loss.append(ctc_loss.cpu().detach().numpy())
                    train_kl_loss.append(kl_loss.cpu().detach().numpy())
                    
                    loss = ctc_loss + args['consistency_scalar']*kl_loss
                    
                else:
                    
                    loss = forward_ctc(pred, adjustedLens, y, y_len)
                
            train_loss.append(loss.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
    
        with torch.no_grad():
    
            avgTrainLoss = np.mean(train_loss)
            avgTrainKLLoss = np.mean(train_kl_loss)
            
            
            model.eval()
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            have_second = args.get('nClasses_2') is not None  # <-- added
            if have_second: 
                total_edit_distance2 = 0
                total_seq_length2 = 0
            
            # <-- unpack batch conditionally
            for batch in testLoader:
                if have_second:
                    X, y, X_len, y_len, testDayIdx, y2, y2_len = batch   # <-- changed
                else:
                    X, y, X_len, y_len, testDayIdx = batch                # <-- changed
                
                if args['maxDay'] is not None:
                    testDayIdx.fill_(args['maxDay'])
                
                # move to device (and y2 if present)
                X, y, X_len, y_len, testDayIdx = (
                    X.to(args["device"]),
                    y.to(args["device"]),
                    X_len.to(args["device"]),
                    y_len.to(args["device"]),
                    testDayIdx.to(args["device"]),
                )
                if have_second:
                    y2     = y2.to(args["device"])                        # <-- added
                    y2_len = y2_len.to(args["device"])                    # <-- added
                
                adjustedLens = model.compute_length(X_len)
            
                # Compute prediction error (use testDayIdx, not dayIdx)
                if have_second:
                    pred, pred2 = model.forward(X, X_len, testDayIdx)     # <-- fixed dayIdx -> testDayIdx
                    loss1 = forward_ctc(pred,  adjustedLens, y,  y_len)
                    loss2 = forward_ctc(pred2, adjustedLens, y2, y2_len)  # <-- uses pred2, y2, y2_len
                    loss = loss1 + loss2
                    allLoss.append(loss2.item())                                # <-- simpler & safe
                    
                else:
                    pred = model.forward(X, X_len, testDayIdx)            # <-- fixed dayIdx -> testDayIdx
                    loss = forward_ctc(pred, adjustedLens, y, y_len)
                    allLoss.append(loss.item())                                # <-- simpler & safe

                for iterIdx in range(pred.shape[0]):
                    # no need to wrap with torch.tensor(...)
                    decodedSeq = torch.argmax(pred[iterIdx, 0:adjustedLens[iterIdx], :], dim=-1)  # <-- simplified
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])

                    trueSeq = np.array(y[iterIdx][0:y_len[iterIdx]].cpu().detach())

                    matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(trueSeq)
                    
                if have_second:
                    for iterIdx in range(pred2.shape[0]):
                        decodedSeq2 = torch.argmax(pred2[iterIdx, 0:adjustedLens[iterIdx], :], dim=-1)  # <-- simplified
                        decodedSeq2 = torch.unique_consecutive(decodedSeq2, dim=-1)
                        decodedSeq2 = decodedSeq2.cpu().detach().numpy()
                        decodedSeq2 = np.array([i for i in decodedSeq2 if i != 0])

                        trueSeq2 = np.array(y2[iterIdx][0:y2_len[iterIdx]].cpu().detach())

                        matcher = SequenceMatcher(a=trueSeq2.tolist(), b=decodedSeq2.tolist())
                        total_edit_distance2 += matcher.distance()
                        total_seq_length2 += len(trueSeq2)                 # <-- fixed trueSeq -> trueSeq2

            avgDayLoss = np.mean(allLoss) if allLoss else 0.0
            cer = total_edit_distance / total_seq_length if total_seq_length > 0 else float('nan')
            if have_second:
                cer2 = total_edit_distance2 / total_seq_length2 if total_seq_length2 > 0 else float('nan')

            endTime = time.time()
            print(
                f"Epoch {epoch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}"
                + (f", cer2: {cer2:>7f}" if have_second else "")
                + f", time/batch: {(endTime - startTime)/100:>7.3f}"
            )
                
            # Log the metrics to wandb
            log_dict = {
                "train_ctc_Loss": avgTrainLoss,
                "ctc_loss": avgDayLoss,
                "cer": cer,
                "time_per_epoch": (endTime - startTime) / 100,
            }
            if have_second:
                log_dict["per"] = cer2   # (consider renaming to 'cer2')
                
            if args['consistency']:
                log_dict['train_kl_loss'] = avgTrainKLLoss
                
            wandb.log(log_dict)

        if len(testCER) > 0 and cer < np.min(testCER):
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer")
            torch.save(scheduler.state_dict(), args['outputDir'] + '/scheduler')
            
        if len(testLoss) > 0 and avgDayLoss < np.min(testLoss):
            torch.save(model.state_dict(), args["outputDir"] + "/modelWeights_ctc")
            
        if have_second:
            if len(testCER2) > 0 and cer2 < np.min(testCER2):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights2")
                
        testLoss.append(avgDayLoss)
        testCER.append(cer)
        if have_second:
            testCER2.append(cer2)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)
            
        scheduler.step()
                    
    wandb.finish()
    return 
