import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Ensure correct import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from data_processing.load_dataset.data_loader import SurgicalPhaseDataset, train_transforms
from models.cnn.cnn import CNNFeatureExtractor  

# Define dataset paths
CSV_FILE = "data/splits/train_split.csv"
IMAGE_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/frames/"
FEATURE_SAVE_PATH = "data/features/"

# Ensure save directory exists
os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)

# Create dataset
dataset = SurgicalPhaseDataset(csv_file=CSV_FILE, image_dir=IMAGE_DIR, transform=train_transforms)

# Create DataLoader for batch processing
batch_size = 32 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Dataset Loaded: {len(dataset)} frames available.")

# Load the CNN feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNFeatureExtractor().to(device)  # CNN outputs spatial features
model.eval()  # Set to evaluation mode

# Storage for features, labels, and frame names
features_list = []
labels_list = []
frame_names_list = []  # Store frame file paths

# Extract features
with torch.no_grad():  
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        features = model(images) 

        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

        start_idx = batch_idx * dataloader.batch_size
        end_idx = start_idx + len(labels)
        frame_names_list.extend(dataset.data.iloc[start_idx:end_idx]["file_path"].tolist())


# Convert lists to numpy arrays
features_array = np.vstack(features_list)  
labels_array = np.concatenate(labels_list) 
frame_names_array = np.array(frame_names_list) 

# Save extracted features for later
np.save(os.path.join(FEATURE_SAVE_PATH, "cnn_features.npy"), features_array)
np.save(os.path.join(FEATURE_SAVE_PATH, "cnn_labels.npy"), labels_array)
np.save(os.path.join(FEATURE_SAVE_PATH, "cnn_frame_names.npy"), frame_names_array)

print(f"Saved CNN features to 'cnn_features.npy' (Shape: {features_array.shape})")
print(f"Saved labels to 'cnn_labels.npy' (Shape: {labels_array.shape})")
print(f"Saved frame names to 'cnn_frame_names.npy' (Shape: {frame_names_array.shape})")
