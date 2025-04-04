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
from loss import ClassBalancedFocalLoss
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
        # self.best_model_path = f"reports/models/best_model_{self.timestamp}_loss.pt"
        # self.training_metrics_path = f"reports/visuals/training/curve_{self.timestamp}_{self.config['batch_size']}_{self.config['gamma']}_{self.config['beta']}.png"
        # self.tsne_path = f"reports/visuals/training/tsne.png"
        
        
    def _prepare_data(self):
        print("Loading datasets...")
        self.train_dataset = FeatureDataset(self.config['train_seq_dir'], self.config['train_label_dir'], mode="training")
        self.val_dataset = FeatureDataset(self.config['val_seq_dir'], self.config['val_label_dir'])

        if self.config['sampler'] == "weighted-random":
            print("Computing class weights...") 
                        
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
            for cls, factor in self.config['boost_factors'].items():
                if cls in class_weights:
                    class_weights[cls] *= factor

            # Compute per-sample weights for sampling
            sample_weights = [class_weights[label] for label in sequence_labels]

            # Initialize sampler
            self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            self.class_counts = class_counts
            self.total = total
                
    
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], sampler=self.sampler, num_workers=4)
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

        # *** Define loss criterion *** 
        
        if self.config['loss'] == "focal": 
            
            self.criterion = FocalLoss(gamma=self.config['gamma']) 
            
        elif self.config['loss'] == "class-balanced-focal":
            
            samples_per_class = [self.class_counts.get(i, 1) for i in range(13)]
            self.criterion = ClassBalancedFocalLoss(
                samples_per_class=samples_per_class,
                beta=self.config['beta'],
                gamma=self.config['gamma'],
                reduction='mean',
            )
                      
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
            
            val_loss, val_acc, all_preds, all_labels, tsne_embeddings, tsne_labels = self._validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Directory to store model metrics and parameters
            model_directory = os.path.join("reports", "model_evaluations", {self.timestamp})
            os.makedirs(model_directory, exist_ok=True)
            
            log_class_freq = self.config['log_per_class_freq']
            plot_confusion_freq = self.config['plot_confusion_freq']
            plot_tsne_freq = self.config['plot_tsne_freq']

            if log_class_freq and (epoch % log_class_freq == 0 or epoch == self.config['epochs'] - 1):
                os.makedirs(model_directory, "per_class_metrics", exist_ok=True)
                self._log_per_class_metrics(all_preds, all_labels, epoch, model_directory)     

            if plot_confusion_freq and (epoch % plot_confusion_freq == 0 or epoch == self.config['epochs'] - 1):
                self._plot_confusion_matrix(all_preds, all_labels, model_directory)

            if plot_tsne_freq and (epoch % plot_tsne_freq == 0 or epoch == self.config['epochs'] - 1):
                self._plot_tsne(tsne_embeddings, tsne_labels, epoch, model_directory)


            # if epoch % 10 == 0 or epoch == (self.config['epochs'] - 1):
            #     self._log_per_class_metrics(all_preds, all_labels, epoch)     
                
            # if epoch % 50 == 0 or epoch == (self.config['epochs'] - 1):
            #     self._plot_tsne(tsne_embeddings, tsne_labels, epoch)

            # if epoch % 5 == 0:
            #     self._plot_confusion_matrix(all_preds, all_labels)
            
            self._save_best_model(val_loss)

            if self.patience_counter >= self.config['patience']:
                print("Early stopping triggered.")
                break
            
        self._plot_train_val_curves(train_losses, val_losses, train_accuracies, val_accuracies, self.config, model_directory)


    def _log_per_class_metrics(self, all_preds, all_labels, epoch, model_directory):
        preds = torch.cat(all_preds).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()

        report = classification_report(
            labels, preds,
            labels=list(range(1, len(self.config['phase_names']) + 1)),
            target_names=self.config['phase_names'],
            digits=4,
            output_dict=True
        )

        df = pd.DataFrame(report).transpose()

        # Round numeric columns to 4 decimal places for readability
        df = df.round(4)

        # Create save directory
        save_dir = os.path.join(model_directory, "per_class_metrics")
        os.makedirs(save_dir, exist_ok=True)

        # Save CSV
        save_path = os.path.join(save_dir, f"epoch_{epoch}_class_performance.csv")
        df.to_csv(save_path)



    # def _log_per_class_metrics(self, all_preds, all_labels, epoch, model_directory):
    #     preds = torch.cat(all_preds).cpu().numpy()
    #     labels = torch.cat(all_labels).cpu().numpy()

    #     report = classification_report(
    #         labels, preds,
    #         labels=list(range(1, len(self.config['phase_names']) + 1)),
    #         target_names=self.config['phase_names'],
    #         digits=4,
    #         output_dict=True,
    #         zero_division=0
    #     )

    #     # Save to CSV
    #     df = pd.DataFrame(report).transpose()
        
    #     # os.makedirs(f"reports/logs/training_logs/{self.timestamp}", exist_ok=True)
    #     # save_path = os.path.join(f"reports/logs/training_logs/{self.timestamp}/epoch_{epoch}_class_performance")
        
    #     save_path = os.path.join(model_directory, "per_class_metrics", f"epoch_{epoch}_class_performance", exist_ok=True)
        
    #     df.to_csv(save_path)


    def _plot_confusion_matrix(self, all_preds, all_labels, model_directory):
        
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
        
        save_path = os.path.join(model_directory, "confusion.png")
        
        plt.savefig(save_path)
        plt.show()

    def _plot_tsne(self, embeddings, labels, epoch, model_directory):
        
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
        
        save_path = os.path.join(model_directory, "tsne.png")

        plt.savefig(save_path)
        plt.close()

    def _plot_train_val_curves(self, train_loss, val_loss, train_acc, val_acc, config, model_directory):
        
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
        
        save_path = os.path.join(model_directory, "train_val_curves.png")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        
    def _save_best_model(self, val_loss, model_directory):
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            save_path = os.path.join(model_directory, "best_model.pt")

            # Save the model
            torch.save(self.model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        else:
            self.patience_counter += 1
            print(f"No improvement. Patience: {self.patience_counter}/{self.config['patience']}")
