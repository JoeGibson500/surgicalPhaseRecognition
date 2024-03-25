import os
import pandas as pd
import numpy as np
import torch
import sys
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import get_phase_to_index

# Manual phase-to-index mapping
PHASE_TO_INDEX = get_phase_to_index()


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet values)
])

class FrameDataset(Dataset):
    def __init__(self, data_split, image_dir, transform=None):
      
        self.data = pd.read_csv(data_split)
        self.image_dir = image_dir
        self.transform = transform

        # Convert phase names to integers using the predefined dictionary
        self.data["phase"] = self.data["phase"].map(PHASE_TO_INDEX)

        # Handle any missing mappings
        if self.data["phase"].isna().any():
            missing_phases = self.data[self.data["phase"].isna()]["phase"].unique()
            raise ValueError(f"Error: Some phase labels are missing in PHASE_TO_INDEX mapping: {missing_phases}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]["file_path"]) # dont think this is needed - test without 
        image = Image.open(img_path).convert("RGB")

        # Original Image Info
        original_size = image.size  
        image_mode = image.mode  

        # Load label
        label = self.data.iloc[idx]["phase"]
        
        # [debug]:
        # label_name = [key for key, value in PHASE_TO_INDEX.items() if value == label][0]  
        # print(label_name)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    

class FeatureDataset(Dataset):
    
    def __init__(self, sequences_dir, labels_dir, mode="training"):
        
        self.sequences_dir = sequences_dir
        self.labels_dir = labels_dir
        self.mode = mode
        
        
         # Collect valid (sequence, label) file pairs
        self.sequence_label_pairs = []
        for video in os.listdir(self.sequences_dir):
            
            
            video_path = os.path.join(self.sequences_dir, video)
            
            for sequence_name in os.listdir(video_path):
                
                # sequence_path = os.path.join(video_path, sequence_name)
                # label_path = os.path.join(self.labels_dir,video,sequence_name) 
                                
                # self.sequence_label_pairs.append((sequence_path, label_path))  
                 
                if self.mode == "training" and "unknown" in sequence_name.lower():
                    continue
                else:
                    sequence_path = os.path.join(video_path, sequence_name)
                    label_path = os.path.join(self.labels_dir, video, sequence_name) 
                                    
                    self.sequence_label_pairs.append((sequence_path, label_path))            
   
        # Check if we found any valid pairs
        if not self.sequence_label_pairs:
            raise ValueError(f"No valid sequences found in {sequences_dir} with corresponding labels in {labels_dir}")

        # print("Valid sequence-label pairs:", self.sequence_label_pairs)



    def __len__(self):
        """ Returns the total number of valid sequence-label pairs. """
        return len(self.sequence_label_pairs)

    def __getitem__(self, idx):
        """ Loads a single sequence and its corresponding label. """
        feature_path, label_path = self.sequence_label_pairs[idx]
        
        # Load sequence features
        feature_sequence = np.load(feature_path)  # Shape: (sequence_length, feature_dim)


        # Load labelshe glob()
        label_sequence = np.load(label_path)  # Shape: (sequence_length,)

        # Convert to PyTorch tensors
        feature_sequence = torch.tensor(feature_sequence, dtype=torch.float32)
        label_sequence = torch.tensor(label_sequence, dtype=torch.long)
            
        return feature_sequence, label_sequence