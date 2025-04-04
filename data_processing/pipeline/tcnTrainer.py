import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from loss import FocalLoss
# from loss import ClassBalancedFocalLoss
from dataset import FeatureDataset
from models.tcn.tcn import TCN

class TCNTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.config = config
        self._prepare_data()
        self._build_model()
        
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.best_model_path = f"reports/models/best_model_{self.timestamp}_loss.pt"
        self.training_metrics_path = f"reports/visuals/training/curve_{self.timestamp}_{self.config['batch_size']}_{self.config['gamma']}_{self.config['beta']}.png"
        self.tsne_path = f"reports/visuals/training/tsne.png"
        
        
    def _prepare_data(self):
        print("Loading datasets...")
        self.train_dataset = FeatureDataset(self.config['train_seq_dir'], self.config['train_label_dir'], mode="training")
        self.val_dataset = FeatureDataset(self.config['val_seq_dir'], self.config['val_label_dir'])

        if self.config['sampler'] == "weighted-random":
            print("Computing class weights...") 
            
                
            # boost_factors = {
            #     2: 2.5,   
            #     9: 0.7,
            #     10: 2.5,   # Moderate boost for '1 arm placing'
            #     11: 0.8
                
            # }
            
            # boost_factors = {
            #     2: 3.5,   # Moderate + boost (was 3.0)
            #     10: 2.0,  # Slight increase (was 1.5)
            #     9: 0.6,   # Reduce placing rings 2 arms a bit more
            #     11: 0.75  # Continue to nudge down 2 arms placing
            # }
           
            # **************
            # boost_factors = {
            #     2: 3.2,
            #     3: 1.2,
            #     9: 0.7,
            #     10: 2.0,
            #     11: 0.85,
            #     12: 2.0
            # }
            
            boost_factors = {
                2: 3.5,
                3: 1.2, 
                9: 0.7,
                10: 2.0,
                11: 0.85,
                12: 2.0
            }
            
            # boost_factors = {
            #     2: 4.0,
            #     9: 1.0,   # Don’t boost it, but don’t penalize it either
            #     10: 2.5,
            #     11: 0.75,
            #     12: 2.0
            # }
    
            sequence_labels = []
            for i in tqdm(range(len(self.train_dataset))):
                _, label_seq = self.train_dataset[i]
                label_seq = label_seq.squeeze()
                label = torch.mode(label_seq).values.item()
                sequence_labels.append(label)

            # Count classes
            class_counts = Counter(sequence_labels)
            total = sum(class_counts.values())

            # Base class weights (inverse frequency)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        
            # Apply boost factors
            for cls, factor in boost_factors.items():
                if cls in class_weights:
                    class_weights[cls] *= factor

            # Compute per-sample weights for sampling
            sample_weights = [class_weights[label] for label in sequence_labels]

            # Initialize sampler
            self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            self.class_counts = class_counts
            self.total = total
                
    
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], sampler=self.sampler, num_workers=4)
            # self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4) 

        elif self.config['sampler'] == "uniform": 
            print("Uniform sampling")
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

    def _build_model(self):
        print("Initializing model...")
        self.model = TCN(input_dim=2048, num_classes=13).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])


        pretrained_path = "reports/models/best_model_20250401-154822_loss.pt" # initial path
        # pretrained_path = "reports/models/best_model_20250403-133158_loss.pt" # new path 
        # pretrained_path = "reports/models/best_model_20250403-201032_loss.pt" # initial path

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        else:
            print(f"Pretrained model not found at {pretrained_path}")


        if self.config['loss'] == "focal":
            
            samples_per_class = [self.class_counts.get(i, 1) for i in range(13)]

            # boost_factors = {
            #     2: 1.5,  # Class 2 gets 3x weight
            #     10: 1.5   # Class 10 gets 1.5x weight
            # }

            # self.criterion = ClassBalancedFocalLoss(
            #     samples_per_class=samples_per_class,
            #     beta=self.config['beta'],
            #     gamma=self.config['gamma'],
            #     reduction='mean',
            #     # boost_factors=boost_factors
            # )
            
            self.criterion = FocalLoss(gamma=1.5) 
            
            # self.criterion = ClassBalancedFocalLoss(
            #     samples_per_class=samples_per_class,
            #     beta=1.995,
            #     gamma=1.8,
            #     boost_factors=boost_factors
            # )
          
        elif self.config['loss'] == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss()

        else:
            print("Incorrect loss configuration")

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
         
    def _accuracy(self, outputs, labels):
        preds = outputs.argmax(dim=-1)
        correct = (preds == labels).float()
        return correct.mean().item()

    def _train_epoch(self):
        self.model.train()
        running_loss, running_acc = 0.0, 0.0
        loop = tqdm(self.train_loader, desc="Training")

        for sequences, labels in loop:
            sequences = sequences.to(self.device)
            labels = self._prepare_labels(labels)

            # outputs = self.model(sequences)
            # loss = self.criterion(outputs, labels)
            
            # ****TEMP***
            outputs, _ = self.model(sequences)  
            loss = self.criterion(outputs, labels)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = self._accuracy(outputs, labels)
            running_loss += loss.item()
            running_acc += acc
            loop.set_postfix(loss=loss.item(), acc=acc)

        return running_loss / len(self.train_loader), running_acc / len(self.train_loader)

    def _prepare_labels(self, labels):
        if labels.ndim == 3:
            labels = labels.squeeze(-1)
        if labels.ndim == 2:
            labels = torch.mode(labels, dim=1).values
        return labels.view(-1).to(self.device)

    # *****Temp*****
    def _validate(self):
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        all_preds, all_labels = [], []

        all_embeddings = []
        all_embedding_labels = [] 
        
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = self._prepare_labels(labels)

                outputs, embeddings = self.model(sequences)  
                preds = outputs.argmax(dim=1)

                mask = (labels != 0).squeeze()
                filtered_outputs = outputs[mask]
                filtered_labels = labels[mask]
                filtered_preds = preds[mask]
                filtered_embeddings = embeddings[mask]  

                if filtered_labels.numel() > 0:
                    loss = self.criterion(filtered_outputs, filtered_labels)
                    acc = self._accuracy(filtered_outputs, filtered_labels)
                    val_loss += loss.item()
                    val_acc += acc

                    all_preds.append(filtered_preds.cpu())
                    all_labels.append(filtered_labels.cpu())

                    all_embeddings.append(filtered_embeddings.cpu())
                    all_embedding_labels.append(filtered_labels.cpu())

        # Return embeddings along with predictions and metrics
        return (
            val_loss / len(self.val_loader),
            val_acc / len(self.val_loader),
            all_preds,
            all_labels,
            torch.cat(all_embeddings),
            torch.cat(all_embedding_labels)
        )


    def train(self):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            train_loss, train_acc = self._train_epoch()
            
            # val_loss, val_acc, all_preds, all_labels = self._validate()
            # ***TEMP***
            val_loss, val_acc, all_preds, all_labels, tsne_embeddings, tsne_labels = self._validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if epoch % 10 == 0 or epoch == (self.config['epochs'] - 1):
                self._log_per_class_metrics(all_preds, all_labels, epoch)     
                
            # if epoch % 50 == 0 or epoch == (self.config['epochs'] - 1):
            #     self._plot_tsne(tsne_embeddings, tsne_labels, epoch)

            # if epoch % 5 == 0:
            #     self._plot_confusion_matrix(all_preds, all_labels)
            
            self._save_best_model(val_loss)

            if self.patience_counter >= self.config['patience']:
                print("Early stopping triggered.")
                break
        
        self._plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, self.config)
        self._plot_confusion_matrix(all_preds, all_labels)        


    def _plot_confusion_matrix(self, all_preds, all_labels):
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        phase_names = self.config['phase_names']
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(1, len(phase_names) + 1))
        cm_normalised = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalised, annot=True, fmt='.2f', cmap='Blues', xticklabels=phase_names, yticklabels=phase_names)
        plt.xlabel("Predicted phase")
        plt.ylabel("True phase")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    def _plot_metrics(self, train_loss, val_loss, train_acc, val_acc, config):
        
        plt.figure(figsize=(10, 6))

        # Loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()

        # Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        # Config summary to show on plot
        if self.config['loss'] == "focal":
            config_text = f"Timestamp: {self.timestamp}, LR: {config['lr']}, Batch: {config['batch_size']}, Loss : {config['loss']}, Gamma: {config['gamma']}, Beta: {config['beta']}"
        
        elif self.config['loss'] == "cross-entropy":
            config_text = f"Timestamp: {self.timestamp}, LR: {config['lr']}, Batch: {config['batch_size']}, Loss: {config['loss']}"

        
        plt.figtext(0.5, 0.01, config_text, ha='center', fontsize=8)

        # Save the plot (overwrite same file)
        plt.tight_layout()
        plt.savefig(self.training_metrics_path)
        plt.close()
        
        
    def _log_per_class_metrics(self, all_preds, all_labels, epoch):
        preds = torch.cat(all_preds).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()

        report = classification_report(
            labels, preds,
            labels=list(range(1, len(self.config['phase_names']) + 1)),
            target_names=self.config['phase_names'],
            digits=4,
            output_dict=True
        )

        # Save to CSV
        df = pd.DataFrame(report).transpose()
        os.makedirs(f"reports/logs/training_logs/{self.timestamp}", exist_ok=True)
        save_path = os.path.join(f"reports/logs/training_logs/{self.timestamp}/epoch_{epoch}_class_performance")
        df.to_csv(save_path)

    def _plot_tsne(self, embeddings, labels, epoch):
        
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        reduced = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", legend='full')
        plt.title(f"t-SNE Visualization at Epoch {epoch}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()


        plt.savefig(self.tsne_path)
        plt.close()



    def _save_best_model(self, val_loss):
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0

            # Save the model
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"Best model saved to {self.best_model_path}")
        else:
            self.patience_counter += 1
            print(f"No improvement. Patience: {self.patience_counter}/{self.config['patience']}")
