# main.py

import argparse
import json
from .mae_phoneme_trainer import start_run
# DAWN and DELL optimization
import torch

import os
import pickle
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .mae import MAE, MAE_EncoderOnly
from .bit import BiT
from .dataset import getDatasetLoaders

import wandb

def trainModel(args):
    

    wandb.init(project="MAE + GRU", entity="skaasyap-ucla", config=dict(args))

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
        day_specific=False, 
        day_specific_tokens = False
    )
    
    
    checkpoint = torch.load(args['best_model_path'], map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder_only_model = MAE_EncoderOnly(model)
    
    if args['freeze_mae_encoder']:
        
        print("FREEZING MAE")
        for param in encoder_only_model.parameters():
            param.requires_grad = False
            
        encoder_only_model.eval()
        
    train_loader, test_loader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"]
    )
    
    # Get the Trainer
    start_run(args=args, mae_encoder=encoder_only_model, trainLoader=train_loader, 
                         testLoader=test_loader, loadedData=loadedData)
    
    
    breakpoint()
    


