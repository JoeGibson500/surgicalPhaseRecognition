import pandas as pd 
import os
import sys
import json
import numpy as np
from collections import defaultdict
import random

class VideoSequenceGenerator:
    
    def __init__(self,
                 frame_info,
                 seq_len,
                 stride,
                 feature_vectors,
                 feature_names,
                 feature_labels, 
                 sequence_directory = "/vol/scratch/SoC/misc/2024/sc22jg/val/sequences/",
                 label_directory =  "/vol/scratch/SoC/misc/2024/sc22jg/val/labels"
                ):

        self.frame_info = pd.read_csv(frame_info)
        self.seq_len = seq_len 
        self.stride = stride
        
        self.feature_vectors = np.load(feature_vectors)
        self.feature_names = np.load(feature_names)
        self.feature_labels = np.load(feature_labels)
        
        self.sequence_directory = sequence_directory
        self.labels_directory = label_directory
                                        
    
    def create_sequence_folder(self):
        
        video_to_indices = defaultdict(list)
   
        for idx, frame_path in enumerate(self.feature_names):
            video_id = frame_path.split("/")[-2]
            video_to_indices[video_id].append(idx)
        
        sequence_metadata = []
        
        for video_id, indices in video_to_indices.items():
            
            num_frames = len(indices)
            
            video_sequence_directory = os.path.join(self.sequence_directory, video_id)
            video_label_directory = os.path.join(self.labels_directory, video_id)
            
            os.makedirs(video_sequence_directory, exist_ok=True)
            os.makedirs(video_label_directory , exist_ok=True)
        
            # Generate sequences with a sliding window
            for start_idx in range(0, num_frames - self.seq_len + 1, self.stride):
                sequence_indices = indices[start_idx:start_idx + self.seq_len]
                
                # Extract features and labels
                sequence_features = self.feature_vectors[sequence_indices]
                sequence_labels = self.feature_labels[sequence_indices].reshape(-1, 1)  # Ensure shape (sequence_length, 1)

                # Save features
                sequence_filename = f"{video_id}_sequence_{start_idx}.npy"
                feature_path = os.path.join(video_sequence_directory, sequence_filename)
                np.save(feature_path, sequence_features)

                # Save labels
                label_filename = f"{video_id}_sequence_{start_idx}.npy"
                label_path = os.path.join(video_label_directory, label_filename)
                np.save(label_path, sequence_labels)

                # Store metadata
                sequence_metadata.append({
                    "video_id": video_id,
                    "sequence_path": sequence_filename,
                    "label_path": label_filename,
                    "start_frame": start_idx * 5,
                    "end_frame": (start_idx + self.seq_len - 1) * 5
                })
                
        # metadata_path = "/vol/scratch/SoC/misc/2024/sc22jg/val/sequences_32/label_sequences_metadata.json"

        metadata_path = "data/sequences/val_label_sequences_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sequence_metadata, f, indent = 4)

class PhaseSequenceGenerator:
    
    def __init__(self,
                 frame_info,
                 seq_len,
                 stride,
                 feature_vectors,
                 feature_names,
                 feature_labels, 
                 sequence_directory,
                 label_directory 
                ):

        self.frame_info = pd.read_csv(frame_info)
        self.seq_len = seq_len 
        self.stride = stride
        
        self.feature_vectors = np.load(feature_vectors)
        self.feature_names = np.load(feature_names)
        self.feature_labels = np.load(feature_labels)
        
        self.sequence_directory = sequence_directory
        self.labels_directory = label_directory
        
#     def create_sequence_folder(self):
#         video_to_indices = defaultdict(list)

#         # Group videos by index -> e.g. 0002_21: 0,1,....150 =
#         for idx, frame_path in enumerate(self.feature_names):
#             video_id = frame_path.split("/")[-2]
#             video_to_indices[video_id].append(idx)
        
#         sequence_metadata = []

#         for video_id, indices in video_to_indices.items():
            
#             num_frames = len(indices)
#             video_sequence_directory = os.path.join(self.sequence_directory, video_id)
#             video_label_directory = os.path.join(self.labels_directory, video_id)
            
#             os.makedirs(video_sequence_directory, exist_ok=True)
#             os.makedirs(video_label_directory , exist_ok=True)

#             # Fetch phase boundaries for the current video
#             phases_for_video = self.frame_info[self.frame_info["video_id"] == video_id]["phase"].values
            
#             # Create sequences for each phase
#             for phase in np.unique(phases_for_video):  # Iterate over all phases
                
#                 # Find indices corresponding to the current phase
#                 phase_indices = [idx for idx, phase_label in zip(indices, phases_for_video) if phase_label == phase]
                
#                 # Generate sequences for the current phase
#                 for start_idx in range(0, len(phase_indices) - self.seq_len + 1, self.stride): # PROBLEM : phases may reoccur so you need to make contiguous block sequences
#                     sequence_indices = phase_indices[start_idx:start_idx + self.seq_len]

#                     # Extract features and labels for the phase-based sequence
#                     sequence_features = self.feature_vectors[sequence_indices]
#                     sequence_labels = self.feature_labels[sequence_indices].reshape(-1, 1)  # Ensure shape (sequence_length, 1)

#                     # Save features
#                     sequence_filename = f"{phase}_sequence_{start_idx}.npy"
#                     feature_path = os.path.join(video_sequence_directory, sequence_filename)
#                     np.save(feature_path, sequence_features)

#                     # Save labels
#                     label_filename = f"{phase}_sequence_{start_idx}.npy"
#                     label_path = os.path.join(video_label_directory, label_filename)
#                     np.save(label_path, sequence_labels)

#                     # Store metadata
#                     sequence_metadata.append({
#                         "video_id": video_id,
#                         "phase": phase,
#                         "sequence_path": sequence_filename,
#                         "label_path": label_filename,
#                         "start_frame": start_idx,
#                         "end_frame": start_idx + self.seq_len - 1
#                     })


# POSSIBLE SOLUTION : 

    def create_sequence_folder(self):
        video_to_indices = defaultdict(list)
        sequence_metadata = []

        # Group frame indices by video ID
        for idx, frame_path in enumerate(self.feature_names):
            video_id = frame_path.split("/")[-2]
            video_to_indices[video_id].append(idx)

        for video_id, indices in video_to_indices.items():
            num_frames = len(indices)

            video_sequence_directory = os.path.join(self.sequence_directory, video_id)
            video_label_directory = os.path.join(self.labels_directory, video_id)

            os.makedirs(video_sequence_directory, exist_ok=True)
            os.makedirs(video_label_directory, exist_ok=True)

            # Fetch phase labels for all frames of this video
            phases_for_video = self.frame_info[self.frame_info["video_id"] == video_id]["phase"].values

            # For each unique phase, split into contiguous chunks
            for phase in np.unique(phases_for_video):
                # Collect all indices for the current phase
                phase_indices_all = [idx for idx, phase_label in zip(indices, phases_for_video) if phase_label == phase]

                # Split into contiguous chunks (where frame indices are sequential)
                contiguous_blocks = []
                current_block = []

                for i, idx in enumerate(phase_indices_all):
                    if not current_block or idx == current_block[-1] + 1:
                        current_block.append(idx)
                    else:
                        contiguous_blocks.append(current_block)
                        current_block = [idx]
                if current_block:
                    contiguous_blocks.append(current_block)

                # Generate sequences within each contiguous block
                for block in contiguous_blocks:
                    for start_idx in range(0, len(block) - self.seq_len + 1, self.stride):
                        sequence_indices = block[start_idx:start_idx + self.seq_len]

                        # Extract features and labels
                        sequence_features = self.feature_vectors[sequence_indices]
                        sequence_labels = self.feature_labels[sequence_indices].reshape(-1, 1)

                        # Save features
                        feature_filename = f"{phase}_sequence_{sequence_indices[0]}.npy"
                        feature_path = os.path.join(video_sequence_directory, feature_filename)
                        np.save(feature_path, sequence_features)

                        # Save labels
                        label_filename = f"{phase}_sequence_{sequence_indices[0]}.npy"
                        label_path = os.path.join(video_label_directory, label_filename)
                        np.save(label_path, sequence_labels)

                        # Save metadata
                        sequence_metadata.append({
                            "video_id": video_id,
                            "phase": phase,
                            "sequence_path": feature_filename,
                            "label_path": label_filename,
                            "start_frame": sequence_indices[0] * 5 ,
                            "end_frame": sequence_indices[-1] * 5
                        })
                        
                        
            # metadata_path = "/vol/scratch/SoC/misc/2024/sc22jg/train/label_sequences_metadata.json"
            metadata_path = "data/sequences/train_label_sequences_metadata.json"

            with open(metadata_path, "w") as f:
                json.dump(sequence_metadata, f, indent = 4)

