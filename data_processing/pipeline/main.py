from feature_extraction import FeatureExtractor
from generate_sequences import VideoSequenceGenerator, PhaseSequenceGenerator
from dataset import FrameDataset, FeatureDataset
from trainer import TCNTrainer
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm  


# # """
# # Main pipeline for training the model


# # Add more desc...

# # """


IMAGE_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/frames/"
SEQUENCE_LENGTH = 32
STRIDE = 2

# # # # """

# # # #   ***** GENERATE TRAINING  DATA ******
     
# # # # """

TRAINING_SPLIT = "data/splits/train_split.csv"
TRAINING_FEATURE_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_training_features.npy"
TRAINING_FEATURE_NAMES_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_training_frame_names.npy"
TRAINING_FEATURE_LABELS_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_training_labels.npy"
TRAINING_SEQUENCE_OUTPUT_FOLDER = "/vol/scratch/SoC/misc/2024/sc22jg/train/sequences_32/"
TRAINING_LABEL_OUTPUT_FOLDER = "/vol/scratch/SoC/misc/2024/sc22jg/train/labels_32/"

# print("Runnning feature extraction on training frames...")

# Create FeatureExtractor instance and run feature extraction
# # training_extractor = FeatureExtractor(TRAINING_SPLIT, IMAGE_DIR, TRAINING_FEATURE_FILE, TRAINING_FEATURE_NAMES_FILE, TRAINING_FEATURE_LABELS_FILE )
# # training_extractor.extract_features()

print("Runnning sequence generation on training features...")

# training_sequence_generator = PhaseSequenceGenerator(TRAINING_SPLIT,
#                                        SEQUENCE_LENGTH,  STRIDE,
#                                        TRAINING_FEATURE_FILE,
#                                        TRAINING_FEATURE_NAMES_FILE,
#                                        TRAINING_FEATURE_LABELS_FILE, 
#                                        TRAINING_SEQUENCE_OUTPUT_FOLDER,
#                                        TRAINING_LABEL_OUTPUT_FOLDER
#                                        )

# training_sequence_generator.create_sequence_folder()

# # # """

# # #   ***** GENERATE VALIDATION  DATA ******s
     
# # # """

VALIDATION_SPLIT = "data/splits/val_split.csv"
VALIDATION_FEATURE_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_val_features.npy"
VALIDATION_FEATURE_NAMES_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_val_frame_names.npy"
VALIDATION_FEATURE_LABELS_FILE = "/vol/scratch/SoC/misc/2024/sc22jg/features/cnn_val_labels.npy"
VALIDATION_SEQUENCE_OUTPUT_FOLDER = "/vol/scratch/SoC/misc/2024/sc22jg/val/sequences_32/"
VALIDATION_LABEL_OUTPUT_FOLDER = "/vol/scratch/SoC/misc/2024/sc22jg/val/labels_32/"

print("Runnning feature extraction on validation frames ...")
# validation_extractor = FeatureExtractor(VALIDATION_SPLIT, IMAGE_DIR, VALIDATION_FEATURE_FILE, VALIDATION_FEATURE_NAMES_FILE, VALIDATION_FEATURE_LABELS_FILE)
# # validation_extractor.extract_features()

print("Runnning sequence generation on validation features...")

# validation_sequence_generator = VideoSequenceGenerator(VALIDATION_SPLIT,
#                                        SEQUENCE_LENGTH,  STRIDE,
#                                        VALIDATION_FEATURE_FILE,
#                                        VALIDATION_FEATURE_NAMES_FILE,
#                                        VALIDATION_FEATURE_LABELS_FILE, 
#                                        VALIDATION_SEQUENCE_OUTPUT_FOLDER,
#                                        VALIDATION_LABEL_OUTPUT_FOLDER)

# validation_sequence_generator.create_sequence_folder()

# ==== Setup CUDA ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Paths ===
TRAIN_SEQUENCES_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/train/sequences_32/"
TRAIN_LABELS_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/train/labels_32"
VAL_SEQUENCES_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/val/sequences_32"
VAL_LABELS_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/val/labels_32"


BEST_MODEL_PATH = "reports/models/"

print("Running dataset creation...")

# === Data ===
train_dataset = FeatureDataset(TRAIN_SEQUENCES_DIR, TRAIN_LABELS_DIR, mode="training")
val_dataset = FeatureDataset(VAL_SEQUENCES_DIR, VAL_LABELS_DIR)


config = {
    'train_seq_dir': TRAIN_SEQUENCES_DIR,
    'train_label_dir': TRAIN_LABELS_DIR,
    'val_seq_dir': VAL_SEQUENCES_DIR,
    'val_label_dir': VAL_LABELS_DIR,
    'batch_size': 20,
    'lr': 1e-6,
    'epochs': 120,
    'patience': 15,
    'gamma' :  1.9,
    'beta' : 0.992,
    'model_save_path': BEST_MODEL_PATH,
    'phase_names': [
        "pull through", "placing rings", "suture pick up", "suture pull through",
        "suture tie", "uva pick up", "uva pull through", "uva tie",
        "placing rings 2 arms", "1 arm placing", "2 arms placing", "pull off"
    ]
}

trainer = TCNTrainer(config)
trainer.train()

