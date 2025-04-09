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
        


    def segment_data(self, data: torch.Tensor, N: int):
        """
        Segments a tensor of shape (batch_size, time, num_features) into
        consecutive chunks of length N, starting from a random index.
        
        Args:
            data (torch.Tensor): Input tensor of shape (batch_size, time, num_features).
            N (int): Length of each segment.
            
        Yields:
            torch.Tensor: Segments of shape (batch_size, N, num_features).
        """
        batch_size, time, num_features = data.shape
        if N > time:
            raise ValueError("Segment length N must be less than or equal to the time dimension.")

        start = 0
        indices = list(range(start, time, N))

        for i in indices:
            if i + N <= time:
                yield data[:, i:i+N, :]
            else:
                yield data[:, -N:, :]

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        chunk_number = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            neural_data, day_idx, X_len = batch
            mask = X_len >= 100
            neural_data  = neural_data[mask]
            neural_data = neural_data.to(self.device)

            for neural_chunk in self.segment_data(neural_data, N=self.model.encoder.trial_length):
                self.optimizer.zero_grad()
                loss, acc = self.model(neural_chunk) #MAE returns reconstruction loss
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
                min_trial_time = torch.min(X_len)
                if min_trial_time < self.model.encoder.trial_length:
                    continue
                neural_data = neural_data[:, :min_trial_time, :]
                neural_data = neural_data.to(self.device)
                # classification_head_logits = self.model(images)['classification_head_logits']
                # loss = self.criterion(classification_head_logits, labels)
                for neural_chunk in self.segment_data(neural_data, N=self.model.encoder.trial_length):
                    loss, acc = self.model(neural_chunk)
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
                torch.save(self.model.state_dict(), 'save_best/best_vit_model.pth')
                
                  
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
        self.save_checkpoint(epoch+1, best_val_loss)
        
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

    def plot_results(self):
        epochs = range(1, self.args['num_epochs'] + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()



