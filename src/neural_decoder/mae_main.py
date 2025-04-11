# main.py

import argparse
import json
from .mae_trainer import Trainer
# DAWN and DELL optimization
import torch

import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .mae import MAE
from .bit import BiT
from .model import GRUDecoder
from .dataset import getDatasetLoaders_MAE
from .augmentations import mask_electrodes

import wandb

def trainModel(args):
    
    
    wandb.init(project="MAE", entity="skaasyap-ucla", config=dict(args))

    # Initialize the model

    enc_model = BiT(
        trial_size=args['trial_size'],
        patch_size=args['patch_size'],
        dim=args['dim'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dim_head=args['dim_head'],
        dropout=args['dropout']
    )

    model = MAE(
        encoder=enc_model,
        encoder_dim = args['dim'], 
        decoder_dim = args['decoder_dim'], #same shape as the encoder model outputs
        masking_ratio=args['masking_ratio'],
        decoder_depth=args['num_decoder_layers'],
        decoder_heads = args['num_decoder_heads'],
        decoder_dim_head = args['decoder_dim_head'], 
        gaussianSmoothWidth = args['gaussianSmoothWidth'], 
        day_specific=args['day_specific'], 
        day_specific_tokens = args['day_specific_tokens']
    )

    
    train_loader, test_loader, loadedData = getDatasetLoaders_MAE(
        args["datasetPath"],
        args["batchSize"])
    
    model = model.to(args['device'])
    print("model moved to device")

    # Get data loaders
    # train_loader, val_loader = get_food101_dataloader(batch_size = args.batch_size, num_workers = args.num_workers)

    print("dataloaders loaded")
    
    device = args['device']

    # Get the Trainer
    trainer = Trainer(model = model, train_loader = train_loader, 
                      val_loader = test_loader, device = device, args = args)

    print("trainer loaded")

    # Save the configuration
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    print("configuration saved")

    # Start training
    trainer.train()


