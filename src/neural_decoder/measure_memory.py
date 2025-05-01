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
from .dataset import getDatasetLoaders, training_batch_generator
from .augmentations import mask_electrodes
import torch.nn.functional as F
from .loss import forward_cr_ctc, forward_ctc




def trainModel(args, model):
    
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        args['restricted_days']
    )
    
    print(len(testLoader))
    breakpoint()

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
        
        schedular_path = os.path.join(args['load_pretrained_model'], 'scheduler')
        scheduler.load_state_dict(torch.load(optimizer_path, map_location=args['device']))
        print(f"Loaded optimizer and scheduler state from {args['load_pretrained_model']}")
        
        
    # --train--
    testLoss = []
    testCER = []
    train_loss = []
    
    # Start of epoch — reset peak memory tracker
    torch.cuda.reset_peak_memory_stats(args["device"])
    
    startTime = time.time()

    for X, y, X_len, y_len, dayIdx, compute_val in training_batch_generator(trainLoader, args):
        
        
        model.train()

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
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
        
        # signals end of epoch. 
        if compute_val:
            endTime = time.time()
            print(endTime - startTime)
            # End of epoch — print peak GPU memory
            peak_mem_epoch = torch.cuda.max_memory_allocated(args["device"])
            print(f"[Epoch Peak] Peak memory allocated during full epoch: {peak_mem_epoch / 1e6:.2f} MB")
            
            with torch.no_grad():
                            
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                
                startTime = time.time()
                
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    
                    #if args['testing_on_held_out']:
                    #    testDayIdx.fill_(args['maxDay'])

                    X, y, X_len, y_len, testDayIdx = (
                        X.to(args["device"]),
                        y.to(args["device"]),
                        X_len.to(args["device"]),
                        y_len.to(args["device"]),
                        testDayIdx.to(args["device"]),
                    )

                    pred = model.forward(X, X_len, testDayIdx)
                    
                endTime = time.time()
                print(endTime - startTime)
                breakpoint()
                    