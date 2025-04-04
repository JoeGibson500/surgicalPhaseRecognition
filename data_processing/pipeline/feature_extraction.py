import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from data_processing.pipeline.dataset import FrameDataset, train_transforms
from models.cnn.cnn import CNNFeatureExtractor  

class FeatureExtractor:
    def __init__(self, data_split, image_dir, feature_save_path, batch_size=32, num_workers=4):
  
        self.data_split = data_split
        self.image_dir = image_dir
        self.feature_save_path = feature_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure save directory exists
        os.makedirs(self.feature_save_path, exist_ok=True)

        # Initialize dataset and DataLoader
        self.dataset = SurgicalPhaseDataset(data_split=self.data_split, image_dir=self.image_dir, transform=train_transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Load the CNN model
        self.model = CNNFeatureExtractor().to(self.device)
        self.model.eval()  # Set model to evaluation mode

        print(f"Dataset Loaded: {len(self.dataset)} frames available.")

    def extract_features(self):
        """
        Extracts features from the dataset using the CNN model and saves them.
        """
        features_list = []
        labels_list = []
        frame_names_list = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images = images.to(self.device)
                features = self.model(images)  # Extract features

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                # Get corresponding frame names
                start_idx = batch_idx * self.dataloader.batch_size
                end_idx = start_idx + len(labels)
                frame_names_list.extend(self.dataset.data.iloc[start_idx:end_idx]["file_path"].tolist())

        # Convert lists to numpy arrays
        features_array = np.vstack(features_list)
        labels_array = np.concatenate(labels_list)
        frame_names_array = np.array(frame_names_list)

        # Save extracted features
        self.save_features(features_array, labels_array, frame_names_array)

    def save_features(self, features_array, labels_array, frame_names_array):
        """
        Saves extracted features, labels, and frame names to .npy files.
        """
        np.save(os.path.join(self.feature_save_path, "cnn_training_features.npy"), features_array)
        np.save(os.path.join(self.feature_save_path, "cnn_training_labels.npy"), labels_array)
        np.save(os.path.join(self.feature_save_path, "cnn_training_frame_names.npy"), frame_names_array)

        print(f"Saved CNN features to 'cnn_training_features.npy' (Shape: {features_array.shape})")
        print(f"Saved labels to 'cnn_training_labels.npy' (Shape: {labels_array.shape})")
        print(f"Saved frame names to 'cnn_training_frame_names.npy' (Shape: {frame_names_array.shape})")


