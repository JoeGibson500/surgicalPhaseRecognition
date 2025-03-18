import pandas as pd 
import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import get_phase_to_index

PHASE_TO_INDEX = get_phase_to_index()

class PhaseFrameGrouper:
    
    def __init__(self, train_split):
        """
        Initializes the PhaseFrameGrouper with a CSV file.

        Args:
            train_split (str): Path to the CSV file containing video ID, frame numbers, and phases.
        """
        self.train_split = train_split
        self.df = None
        self.phase_dict = defaultdict(list)


    def load_data(self):
        """Loads and processes the CSV file into a DataFrame."""
        self.df = pd.read_csv(self.train_split)

        # Ensure necessary columns exist
        if not {"video_id", "frame_number", "phase", "file_path"}.issubset(self.df.columns):
            raise ValueError("CSV must contain 'video_id', 'frame_number', 'phase', and 'file_path' columns.")

        # Map phase names to numeric indexes using PHASE_TO_INDEX
        self.df["phase_id"] = self.df["phase"].map(PHASE_TO_INDEX)

        # Drop Phase 0 (which corresponds to 'unknown')
        self.df = self.df[self.df["phase_id"] != 0]

        # self.df["phase_id"] = self.df["phase_id"].astype(int)  # Convert phase IDs to integers

        # Sort by phase, video, and frame number
        self.df = self.df.sort_values(by=["phase", "video_id", "frame_number"])

    def generate_phase_ranges(self):
        """Generates frame ranges grouped by phase and video."""
        for (phase_id, video), group in self.df.groupby(["phase", "video_id"]):
            frame_numbers = sorted(group["frame_number"].tolist())
            ranges = []

            # Convert frames to continuous frame ranges
            start = frame_numbers[0]
            prev = start

            for frame in frame_numbers[1:]:
                if frame != prev + 5:  # If gap detected, finalize the previous range
                    ranges.append(f"{video}/frames {start} - frames {prev}")
                    start = frame  # Start new range
                prev = frame

            # Append the last range
            ranges.append(f"{video}/frames {start} - frames {prev}")

            # Store in dictionary
            self.phase_dict[phase_id].extend(ranges)

    def save_to_json(self, output_file="data/sequences/train_phases.json"):
        """Saves the grouped phase ranges into a JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.phase_dict, f, indent=4)
        print(f"Saved phase frame ranges to {output_file}")
        
        
        
class PhaseSequenceGenerator:
    
    def __init__(self, phases_file, seq_len, stride, output_directory = "data/sequences/"):
        
       self.phases_file =  phases_file 
       self.seq_len = seq_len
       self.stride = stride
       self.output_directory = output_directory
       self.sequences_dict = {}
       
       
    def load_phase_data(self):
        """Loads train_phases.json into memory."""
        with open(self.phases_file, "r") as f:
            self.phase_data = json.load(f)
       
    def generate_sequences(self):
        """Generates phase-pure sequences using a sliding window approach."""
        if self.phase_data is None:
            raise ValueError("Phase data is not loaded. Run `load_phase_data()` first.")

        for phase, video_ranges in self.phase_data.items():
            self.sequences_dict[phase] = []

            for range_entry in video_ranges:
                video_id, frame_range = range_entry.split("/frames ")

                # Extract start and end frame numbers
                start_frame, end_frame = map(lambda x: int(x.replace("frames ", "")), frame_range.split(" - "))

                # Generate sequences using a sliding window
                for i in range(start_frame, end_frame - self.seq_len + 1, self.stride):
                    sequence = [f"{video_id}/frames {frame}" for frame in range(i, i + self.seq_len, 5)]
                    self.sequences_dict[phase].append(sequence)

      
    def save_sequences(self, output_filename="phase_sequences.npy"):
        """Saves generated sequences to a NumPy file."""
        output_path = os.path.join(self.output_directory, output_filename)
        np.save(output_path, self.sequences_dict)
