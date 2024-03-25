
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

class DatasetSplitter:
    
    def __init__(self, metadata_path, output_folder="data/splits/", seed=None):
        self.metadata_path = metadata_path
        self.output_folder = output_folder
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        np.random.seed(self.seed)

        self.df = None
        self.train_videos = set()
        self.val_videos = set()
        self.test_videos = set()

    def load_metadata(self):
        """Load metadata from CSV file."""
        self.df = pd.read_csv(self.metadata_path)
        self.unique_videos = set(self.df["video_id"].unique())
        logging.info(f"Loaded metadata from {self.metadata_path}. Total unique videos: {len(self.unique_videos)}")

    def get_phase_video_map(self):
        """Returns a mapping of phases to videos that contain them."""
        phase_video_map = defaultdict(set)
        video_phase_map = self.df.groupby("video_id")["phase"].apply(set).to_dict()

        for video, phases in video_phase_map.items():
            for phase in phases:
                phase_video_map[phase].add(video)

        return phase_video_map

    def ensure_phase_coverage(self):
        """Ensures that each phase appears in at least one of the train/val/test splits."""
        phase_video_map = self.get_phase_video_map()
        assigned_videos = set()

        for phase, videos in phase_video_map.items():
            videos = list(videos)
            np.random.shuffle(videos)

            if len(videos) >= 3:
                self.train_videos.add(videos[0])
                self.val_videos.add(videos[1])
                self.test_videos.add(videos[2])
                assigned_videos.update(videos[:3])
            elif len(videos) == 2:
                self.train_videos.add(videos[0])
                self.val_videos.add(videos[1])
                self.test_videos.add(np.random.choice(videos))  # Randomly duplicate one for test
                assigned_videos.update(videos)
            else:
                split_choice = np.random.choice(["train", "val", "test"])
                getattr(self, f"{split_choice}_videos").add(videos[0])
                assigned_videos.add(videos[0])

        logging.info(f"Initial phase coverage ensured. Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")
        return assigned_videos

    def enforce_70_15_15_split(self, assigned_videos):
        """Ensures exactly 70% train, 15% val, and 15% test split while avoiding duplicates."""
        remaining_videos = list(self.unique_videos - assigned_videos)
        np.random.shuffle(remaining_videos)

        total_videos = len(self.unique_videos)
        train_target = int(total_videos * 0.7)
        val_target = int(total_videos * 0.15)
        test_target = total_videos - (train_target + val_target)  # Exact split

        # Remove potential duplicates between sets
        self.train_videos -= (self.val_videos | self.test_videos)
        self.val_videos -= (self.train_videos | self.test_videos)
        self.test_videos -= (self.train_videos | self.val_videos)

        # Add remaining videos to ensure exact split
        while len(self.train_videos) < train_target and remaining_videos:
            self.train_videos.add(remaining_videos.pop(0))
        while len(self.val_videos) < val_target and remaining_videos:
            self.val_videos.add(remaining_videos.pop(0))
        while len(self.test_videos) < test_target and remaining_videos:
            self.test_videos.add(remaining_videos.pop(0))

        logging.info(f"Final Split -> Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")

    def save_splits(self):
        """Save final train, validation, and test splits to CSV."""
        train_df = self.df[self.df["video_id"].isin(self.train_videos)]
        val_df = self.df[self.df["video_id"].isin(self.val_videos)]
        test_df = self.df[self.df["video_id"].isin(self.test_videos)]

        train_df.to_csv(f"{self.output_folder}/train_split.csv", index=False)
        val_df.to_csv(f"{self.output_folder}/val_split.csv", index=False)
        test_df.to_csv(f"{self.output_folder}/test_split.csv", index=False)

        logging.info("Final train/val/test splits saved successfully.")

    def run(self):
        """Main pipeline to execute the dataset splitting."""
        self.load_metadata()
        assigned_videos = self.ensure_phase_coverage()
        self.enforce_70_15_15_split(assigned_videos)
        self.save_splits()


if __name__ == "__main__":
    splitter = DatasetSplitter(metadata_path="data/frames/frames_metadata.csv")
    splitter.run()
