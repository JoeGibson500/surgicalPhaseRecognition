# import pandas as pd
# from sklearn.model_selection import train_test_split
# import logging

# logging.basicConfig(level=logging.INFO)

# class DatasetSplitter:
    
#     def __init__(self, metadata_path, output_folder="data/splits/"):
#         self.metadata_path = metadata_path
#         self.output_folder = output_folder
#         self.df = None
#         self.train_videos = []
#         self.val_videos = []
#         self.test_videos = []

#     def load_metadata(self):
#         """Load metadata from CSV file."""
#         self.df = pd.read_csv(self.metadata_path)
#         logging.info(f"Loaded metadata from {self.metadata_path}. Total frames: {len(self.df)}")

#     def ensure_phase_coverage(self):
#         """Ensure all phases appear in each split by first distributing videos with unique phases."""
#         video_phase_map = self.df.groupby("video_id")["phase"].apply(set).to_dict()
#         phase_video_map = {}
        
#         # Create a mapping of each phase to videos containing it
#         for video, phases in video_phase_map.items():
#             for phase in phases:
#                 if phase not in phase_video_map:
#                     phase_video_map[phase] = []
#                 phase_video_map[phase].append(video)

#         # Assign one video per phase to each split
#         assigned_videos = set()
#         for phase, videos in phase_video_map.items():
#             if len(videos) >= 3:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[1])
#                 self.test_videos.append(videos[2])
#                 assigned_videos.update(videos[:3])
#             elif len(videos) == 2:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[1])
#                 self.test_videos.append(videos[0])  # Duplicate one if needed
#                 assigned_videos.update(videos)
#             elif len(videos) == 1:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[0])
#                 self.test_videos.append(videos[0])
#                 assigned_videos.add(videos[0])
        
#         logging.info(f"Initial Phase Coverage Assigned. Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")

#         return assigned_videos

#     def balance_frame_distribution(self, assigned_videos):
#         """Distribute remaining videos while balancing frame counts."""
#         video_frame_counts = self.df.groupby("video_id")["frame_number"].count()
#         remaining_videos = [v for v in video_frame_counts.index if v not in assigned_videos]
        
#         # Sort videos by frame count (longest first)
#         remaining_videos = sorted(remaining_videos, key=lambda v: video_frame_counts[v], reverse=True)

#         train_frames, val_frames, test_frames = sum(video_frame_counts[v] for v in self.train_videos), \
#                                                 sum(video_frame_counts[v] for v in self.val_videos), \
#                                                 sum(video_frame_counts[v] for v in self.test_videos)

#         for video in remaining_videos:
#             video_frames = video_frame_counts[video]

#             # Assign to the split with the lowest total frame count
#             if train_frames <= val_frames and train_frames <= test_frames:
#                 self.train_videos.append(video)
#                 train_frames += video_frames
#             elif val_frames <= test_frames:
#                 self.val_videos.append(video)
#                 val_frames += video_frames
#             else:
#                 self.test_videos.append(video)
#                 test_frames += video_frames

#         logging.info(f"Balanced Frame Distribution. Train Frames: {train_frames}, Val Frames: {val_frames}, Test Frames: {test_frames}")

#     def save_splits(self):
#         """Save final train, validation, and test splits to CSV."""
#         train_df = self.df[self.df["video_id"].isin(self.train_videos)]
#         val_df = self.df[self.df["video_id"].isin(self.val_videos)]
#         test_df = self.df[self.df["video_id"].isin(self.test_videos)]

#         train_df.to_csv(f"{self.output_folder}/train_split.csv", index=False)
#         val_df.to_csv(f"{self.output_folder}/val_split.csv", index=False)
#         test_df.to_csv(f"{self.output_folder}/test_split.csv", index=False)

#         logging.info("Final train/val/test splits saved successfully.")

#     def run(self):
#         """Main pipeline to execute the dataset splitting."""
#         self.load_metadata()
#         assigned_videos = self.ensure_phase_coverage()
#         self.balance_frame_distribution(assigned_videos)
#         self.save_splits()


# if __name__ == "__main__":
#     splitter = DatasetSplitter(metadata_path="data/frames/frames_metadata.csv")
#     splitter.run()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import logging

# logging.basicConfig(level=logging.INFO)

# class DatasetSplitter:
    
#     def __init__(self, metadata_path, output_folder="data/splits/"):
#         self.metadata_path = metadata_path
#         self.output_folder = output_folder
#         self.df = None
#         self.train_videos = []
#         self.val_videos = []
#         self.test_videos = []

#     def load_metadata(self):
#         """Load metadata from CSV file."""
#         self.df = pd.read_csv(self.metadata_path)
#         logging.info(f"Loaded metadata from {self.metadata_path}. Total frames: {len(self.df)}")

#     def ensure_phase_coverage(self):
#         """Ensure all phases appear in each split by first distributing videos with unique phases."""
#         video_phase_map = self.df.groupby("video_id")["phase"].apply(set).to_dict()
#         phase_video_map = {}
        
#         # Create a mapping of each phase to videos containing it
#         for video, phases in video_phase_map.items():
#             for phase in phases:
#                 if phase not in phase_video_map:
#                     phase_video_map[phase] = []
#                 phase_video_map[phase].append(video)

#         # Assign one video per phase to each split
#         assigned_videos = set()
#         for phase, videos in phase_video_map.items():
#             if len(videos) >= 3:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[1])
#                 self.test_videos.append(videos[2])
#                 assigned_videos.update(videos[:3])
#             elif len(videos) == 2:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[1])
#                 self.test_videos.append(videos[0])  # Duplicate one if needed
#                 assigned_videos.update(videos)
#             elif len(videos) == 1:
#                 self.train_videos.append(videos[0])
#                 self.val_videos.append(videos[0])
#                 self.test_videos.append(videos[0])
#                 assigned_videos.add(videos[0])
        
#         logging.info(f"Initial Phase Coverage Assigned. Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")

#         return assigned_videos

#     def balance_frame_distribution(self, assigned_videos):
#         """Distribute remaining videos while balancing frame counts, prioritizing the training set (70%)."""
#         video_frame_counts = self.df.groupby("video_id")["frame_number"].count()
#         remaining_videos = [v for v in video_frame_counts.index if v not in assigned_videos]
        
#         # Sort videos by frame count (longest first)
#         remaining_videos = sorted(remaining_videos, key=lambda v: video_frame_counts[v], reverse=True)

#         train_frames, val_frames, test_frames = sum(video_frame_counts[v] for v in self.train_videos), \
#                                                 sum(video_frame_counts[v] for v in self.val_videos), \
#                                                 sum(video_frame_counts[v] for v in self.test_videos)

#         for video in remaining_videos:
#             video_frames = video_frame_counts[video]

#             # Prioritize placing more videos in the training set (70%)
#             if len(self.train_videos) / len(self.df["video_id"].unique()) < 0.7:
#                 self.train_videos.append(video)
#                 train_frames += video_frames
#             elif len(self.val_videos) / len(self.df["video_id"].unique()) < 0.15:
#                 self.val_videos.append(video)
#                 val_frames += video_frames
#             else:
#                 self.test_videos.append(video)
#                 test_frames += video_frames

#         logging.info(f"Balanced Frame Distribution. Train Frames: {train_frames}, Val Frames: {val_frames}, Test Frames: {test_frames}")

#     def save_splits(self):
#         """Save final train, validation, and test splits to CSV."""
#         train_df = self.df[self.df["video_id"].isin(self.train_videos)]
#         val_df = self.df[self.df["video_id"].isin(self.val_videos)]
#         test_df = self.df[self.df["video_id"].isin(self.test_videos)]

#         train_df.to_csv(f"{self.output_folder}/train_split.csv", index=False)
#         val_df.to_csv(f"{self.output_folder}/val_split.csv", index=False)
#         test_df.to_csv(f"{self.output_folder}/test_split.csv", index=False)

#         logging.info("Final train/val/test splits saved successfully.")

#     def run(self):
#         """Main pipeline to execute the dataset splitting."""
#         self.load_metadata()
#         assigned_videos = self.ensure_phase_coverage()
#         self.balance_frame_distribution(assigned_videos)
#         self.save_splits()


# if __name__ == "__main__":
#     splitter = DatasetSplitter(metadata_path="data/frames/frames_metadata.csv")
#     splitter.run()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

class DatasetSplitter:
    
    def __init__(self, metadata_path, output_folder="data/splits/", seed=None):
        self.metadata_path = metadata_path
        self.output_folder = output_folder
        self.df = None
        self.train_videos = []
        self.val_videos = []
        self.test_videos = []
        self.seed = seed if seed is not None else np.random.randint(0, 10000)  # Ensures different splits each run
        np.random.seed(self.seed)  # Set random seed

    def load_metadata(self):
        """Load metadata from CSV file."""
        self.df = pd.read_csv(self.metadata_path)
        logging.info(f"Loaded metadata from {self.metadata_path}. Total frames: {len(self.df)}")

    def ensure_phase_coverage(self):
        """Ensure all phases appear in each split by randomly distributing videos with unique phases."""
        video_phase_map = self.df.groupby("video_id")["phase"].apply(set).to_dict()
        phase_video_map = {}

        # Create a mapping of each phase to videos containing it
        for video, phases in video_phase_map.items():
            for phase in phases:
                if phase not in phase_video_map:
                    phase_video_map[phase] = []
                phase_video_map[phase].append(video)

        # Shuffle videos within each phase for random selection
        for phase in phase_video_map:
            np.random.shuffle(phase_video_map[phase])

        assigned_videos = set()

        # Assign one video per phase to each split, ensuring all phases appear
        for phase, videos in phase_video_map.items():
            num_videos = len(videos)
            if num_videos >= 3:
                self.train_videos.append(videos[0])
                self.val_videos.append(videos[1])
                self.test_videos.append(videos[2])
                assigned_videos.update(videos[:3])
            elif num_videos == 2:
                self.train_videos.append(videos[0])
                self.val_videos.append(videos[1])
                self.test_videos.append(np.random.choice(videos))  # Random duplicate
                assigned_videos.update(videos)
            elif num_videos == 1:
                self.train_videos.append(videos[0])
                self.val_videos.append(videos[0])
                self.test_videos.append(videos[0])
                assigned_videos.add(videos[0])

        logging.info(f"Initial Phase Coverage Assigned. Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")

        return assigned_videos

    def balance_frame_distribution(self, assigned_videos):
        """Distribute remaining videos while balancing frame counts, prioritizing the training set (70%)."""
        video_frame_counts = self.df.groupby("video_id")["frame_number"].count()
        remaining_videos = [v for v in video_frame_counts.index if v not in assigned_videos]

        # Shuffle remaining videos for random selection
        np.random.shuffle(remaining_videos)

        total_videos = len(self.df["video_id"].unique())

        for video in remaining_videos:
            if len(self.train_videos) / total_videos < 0.7:
                self.train_videos.append(video)
            elif len(self.val_videos) / total_videos < 0.15:
                self.val_videos.append(video)
            else:
                self.test_videos.append(video)

        logging.info(f"Final Split Sizes -> Train: {len(self.train_videos)}, Val: {len(self.val_videos)}, Test: {len(self.test_videos)}")

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
        self.balance_frame_distribution(assigned_videos)
        self.save_splits()


if __name__ == "__main__":
    splitter = DatasetSplitter(metadata_path="data/frames/frames_metadata.csv")
    splitter.run()
