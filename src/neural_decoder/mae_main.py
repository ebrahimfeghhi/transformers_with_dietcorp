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
from .dataset import SpeechDataset_MAE
from .augmentations import mask_electrodes

import wandb


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, days, X_len = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)

        return (
            X_padded,
            torch.stack(days), 
            torch.stack(X_len)
        )

    train_ds = SpeechDataset_MAE(loadedData["train"], transform=None)
    test_ds = SpeechDataset_MAE(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

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
        gaussianSmoothWidth = args['gaussianSmoothWidth']
    )

    
    train_loader, test_loader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
    
    model = model.to(args['device'])
    print("model moved to device")

    # Get data loaders
    # train_loader, val_loader = get_food101_dataloader(batch_size = args.batch_size, num_workers = args.num_workers)

    print("dataloaders loaded")
    
    device = args['device']

    # Get the Trainer
    trainer = Trainer(model = model, train_loader = train_loader, val_loader = test_loader, device = device, args = args)

    print("trainer loaded")

    # Save the configuration
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    print("configuration saved")

    # Start training
    trainer.train()


