# main.py

import argparse
import json
from .mae_trainer import Trainer
# DAWN and DELL optimization
import torch
import intel_extension_for_pytorch as ipex

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
from .dataset import SpeechDataset
from .augmentations import mask_electrodes

import wandb


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
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

    # Initialize the model
    enc_model = BiT(
        image_size=args['trial_size'],
        patch_size=args['patch_size'],
        num_classes=args['num_classes'],
        dim=args['dim'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dim_head=args['dim_head'],
        dropout=args['dropout']
    )

    model = MAE(
        encoder=enc_model,
        decoder_dim = enc_model.pos_embedding.shape[-1], #same shape as the encoder model outputs
        masking_ratio=args['masking_ratio'],
        decoder_depth=args['num_decoder_layers'],
        decoder_heads=args['num_decoder_heads'],
        decoder_dim_head=args['num_decoder_dim_head']
    )



    model = model.to(args['device'])
    print("model moved to device")


    # Get data loaders
    # train_loader, val_loader = get_food101_dataloader(batch_size = args.batch_size, num_workers = args.num_workers)

    train_loader, test_loader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    print("dataloaders loaded")

    # Get the Trainer
    trainer = Trainer(model = model, train_loader = train_loader, val_loader = test_loader, device = device, args = args)

    print("trainer loaded")

    # Save the configuration
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    print("configuration saved")

    # Start training
    trainer.train()


