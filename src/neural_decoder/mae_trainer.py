# training/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
from torcheval.metrics import R2Score
from .dataset import segment_data

class Trainer:
    
    def __init__(self, model, train_loader, val_loader, device, args):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.model.to(self.device)

        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args['num_epochs'])
        self.metric = R2Score()

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        self.save_folder = args['outputDir']
        os.makedirs(self.save_folder, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        chunk_number = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            
            neural_data, day_idx, X_len = batch
            
            # select trials that are longer than chunk size
            #mask = X_len >= self.model.encoder.trial_length 
            
            #neural_data  = neural_data[mask]
            
            neural_data, day_idx, X_len = (neural_data.to(self.device), 
                           day_idx.to(self.device),
                           X_len.to(self.device))
            
            self.optimizer.zero_grad()
            loss, acc = self.model(neural_data, X_len, day_idx) #MAE returns reconstruction loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_acc += acc.item()
            chunk_number += 1
                
            # _, predicted = classification_head_logits.max(1)
            # total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()

        return total_loss / chunk_number, total_acc/chunk_number

    def validate(self):
        
        self.model.eval()
        total_loss = 0
        total_acc = 0
        chunk_number = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                neural_data, day_idx, X_len = batch
            
                # select trials that are longer than chunk size
                #mask = X_len >= self.model.encoder.trial_length 
                #neural_data  = neural_data[mask]
                            
                neural_data, day_idx, X_len = (neural_data.to(self.device), 
                           day_idx.to(self.device),
                           X_len.to(self.device))
            
               
                loss, acc = self.model(neural_data, X_len, day_idx)
                total_loss += loss.item()
                total_acc += acc.item()
                chunk_number+=1
                # _, predicted = classification_head_logits.max(1)
                # total += labels.size(0)
                # correct += predicted.eq(labels).sum().item()
    
        return total_loss / chunk_number, total_acc/chunk_number

    def train(self):
        
        best_val_loss = torch.inf

        for epoch in range(self.args['num_epochs']):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{self.args['num_epochs']}:")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss, 
                    'val_acc': val_acc
                }, f'{self.save_folder}/save_best.pth')
                
            # Log the metrics to wandb
            wandb.log({
                'train_loss': train_loss, 
                'train_r2': train_acc,
                "loss": val_loss,
                'val_r2': val_acc,
            })
            

        print(f"Best Validation Loss: {best_val_loss:.2f}%")
        #self.plot_results()

        # Save final checkpoint after training completes
        #self.save_checkpoint(epoch+1, best_val_loss)
        
    def save_checkpoint(self, epoch, best_val_loss):
        """Saves the model checkpoint at the end of training."""
        checkpoint_path = os.path.join('checkpoints', f'vit_checkpoint_epoch_{epoch}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at '{checkpoint_path}'")


