import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Manual phase-to-index mapping
PHASE_TO_INDEX = {
    "unknown": 0,
    "pull through": 1,
    "placing rings": 2,
    "suture pick up": 3,
    "suture pull through": 4,
    "suture tie": 5,
    "uva pick up": 6,
    "uva pull through": 7,
    "uva tie": 8,
    "placing rings 2 arms": 9,
    "1 arm placing": 10,
    "2 arms placing": 11,
    "pull off": 12
}

class SurgicalPhaseDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            image_dir (str): Root directory containing image frames.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
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
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]["file_path"])
        image = Image.open(img_path).convert("RGB")

        # Original Image Info
        original_size = image.size  # (width, height)
        image_mode = image.mode  # e.g., 'RGB'

        # Load label
        label = self.data.iloc[idx]["phase"]
        label_name = [key for key, value in PHASE_TO_INDEX.items() if value == label][0]  # Reverse lookup for name

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Extract tensor stats
        # tensor_min = image.min().item()
        # tensor_max = image.max().item()
        # tensor_mean = image.mean().item()
        # tensor_std = image.std().item()

        return image, torch.tensor(label, dtype=torch.long)


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet values)
])
