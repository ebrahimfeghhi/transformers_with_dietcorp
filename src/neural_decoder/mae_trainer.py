# training/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex



class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.model.to(self.device)

        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        if device == "xpu":
            self.model, self.optimizer = ipex.optimize(model=self.model, optimizer = self.optimizer, dtype=torch.float32)
            print("model and optimizer correctly optimized with ipex")  

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            neural_data, labels = batch
            neural_data, labels = neural_data.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            # classification_head_logits = self.model(neural_data)['classification_head_logits']
            # loss = self.criterion(classification_head_logits, labels)
            loss = self.model(neural_data) #MAE returns reconstruction loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # _, predicted = classification_head_logits.max(1)
            # total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                neural_data, labels = batch
                neural_data, labels = neural_data.to(self.device), labels.to(self.device)

                # classification_head_logits = self.model(images)['classification_head_logits']
                # loss = self.criterion(classification_head_logits, labels)

                loss = self.model(neural_data)

                total_loss += loss.item()
                # _, predicted = classification_head_logits.max(1)
                # total += labels.size(0)
                # correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.val_loader)

    def train(self):
        best_val_acc = 0

        for epoch in range(self.args.num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1}/{self.args.num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_vit_model.pth')

        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        self.plot_results()

        # Save final checkpoint after training completes
        self.save_checkpoint(epoch+1, best_val_acc)
        
    def save_checkpoint(self, epoch, best_val_acc):
        """Saves the model checkpoint at the end of training."""
        checkpoint_path = os.path.join('checkpoints', f'vit_checkpoint_epoch_{epoch}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, checkpoint_path)
        print(f"Checkpoint saved at '{checkpoint_path}'")

    def plot_results(self):
        epochs = range(1, self.args.num_epochs + 1)

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



