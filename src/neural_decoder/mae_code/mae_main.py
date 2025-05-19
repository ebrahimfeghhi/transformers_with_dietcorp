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

from .mae import MAE, MAE_with_mask
from .bit import BiT_Phoneme
from .model import GRUDecoder
from .dataset import getDatasetLoaders_MAE
from .augmentations import mask_electrodes

import wandb



def trainModel(args):
    
    
    wandb.init(project="MAE", entity="skaasyap-ucla", config=dict(args), name=args['modelName'])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    # Initialize the model
    enc_model = BiT_Phoneme(
        patch_size=args['patch_size'],
        dim=args['dim'],
        dim_head=args['dim_head'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'], 
        look_ahead=0,
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'], 
        max_mask_pct=args['max_mask_pct'], 
        num_masks=args['num_masks'], 
        nClasses=args['nClasses'], 
    )

    model = MAE_with_mask(
        encoder=enc_model,
        encoder_dim = args['dim'], 
        decoder_dim = args['decoder_dim'], #same shape as the encoder model outputs
        decoder_depth=args['num_decoder_layers'],
        decoder_heads = args['num_decoder_heads'],
        decoder_dim_head = args['decoder_dim_head'], 
        gaussianSmoothWidth = args['gaussianSmoothWidth'], 
        constantOffsetSD=args['constantOffsetSD'], 
        whiteNoiseSD=args['whiteNoiseSD']
    )

    train_loader, test_loader, loadedData = getDatasetLoaders_MAE(
        args["datasetPath"],
        args["batchSize"])
    
    model = model.to(args['device'])
    
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


