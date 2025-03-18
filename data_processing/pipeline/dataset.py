import os
import pandas as pd
import torch
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import get_phase_to_index

# Manual phase-to-index mapping
PHASE_TO_INDEX = get_phase_to_index()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import clean_phase_name

class SurgicalPhaseDataset(Dataset):
    def __init__(self, data_split, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            image_dir (str): Root directory containing image frames.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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

        return image, torch.tensor(label, dtype=torch.long)


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet values)
])
