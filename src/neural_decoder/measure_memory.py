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


def trainModel_mem(args, model):
    
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
    
    avg_epoch_time = []
    for epoch in range(args["start_epoch"], args['n_epochs']):
        
        train_loss = []
        model.train()
        
        
        startTime = time.time()
        for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc="Training")):
            
               
            X, y, X_len, y_len, dayIdx = (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                dayIdx.to(args["device"]),
            )

            
            
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
        endTime = time.time()
        print("Training time: ", endTime - startTime)
        # End of epoch — print peak GPU memory
        peak_mem_epoch = torch.cuda.max_memory_allocated(args["device"])
        print(f"[Epoch Peak] Peak memory allocated during full epoch: {peak_mem_epoch / 1e6:.2f} MB")
        avg_epoch_time.append(endTime-startTime)
        if epoch == 9:
            print(np.mean(avg_epoch_time), np.std(avg_epoch_time))
            breakpoint()
        continue
        
        with torch.no_grad():
                        
            model.eval()
            timings = []
            
            total_len = 0
                        
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
                
                torch.cuda.synchronize()  # Optional but recommended if using GPU
                start = time.time()

                pred = model.forward(X, X_len, testDayIdx)

                torch.cuda.synchronize()
                end = time.time()
                
                total_len += X_len

                # Measure time per trial (per sample in batch)
                elapsed_per_sample = (end - start) / X.shape[0]
                timings.append(elapsed_per_sample)
                
            timings_ms = np.array(timings) * 1000
            mean_time = np.mean(timings_ms)
            std_time = np.std(timings_ms)
            print(f"Inference time per trial: {mean_time:.3f} ± {std_time:.3f} ms")
            print(f"Avg trial length: {total_len/len(testLoader)}")
            
            breakpoint()
                        