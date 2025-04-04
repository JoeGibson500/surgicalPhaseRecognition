import pandas as pd
import multiprocessing 
import os
import cv2
import logging
from tqdm import tqdm


class FrameExtractionManager:
    
    def __init__(self, video_folder, timestamp_folder, output_folder, num_workers, frame_skip ):
        self.video_folder = video_folder
        self.timestamp_folder = timestamp_folder
        self.output_folder = output_folder
        self.num_workers = num_workers
        self.frame_skip = frame_skip
        
        
    def process_frame_extraction(self, video_filename):
        
        video_path = os.path.join(self.video_folder, video_filename)
        timestamp_file = os.path.join(self.timestamp_folder, video_filename.replace(".mkv", ".xlsx"))

        print(f"Processing {video_filename}...")
        extractor = FrameExtractor(video_path, timestamp_file, self.output_folder, self.frame_skip)
        extractor.extract_frames()
        
        
    def run(self):
        
        """
        Starts multiprocessing frame extraction for all videos.
        """
        
        video_files = [f for f in os.listdir(self.video_folder) if f.endswith(".mkv")]

        if not video_files:
            print("No videos found in the folder. Exiting.")
            exit(1)

        print(f"Running frame extraction on {len(video_files)} videos...")

        # Set up multiprocessing pool
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            list(tqdm(pool.imap(self.process_frame_extraction, video_files), total=len(video_files), desc="Processing Videos"))

        print("All videos processed successfully!")
        
        
class FrameExtractor:
    
    def __init__(self, video_path, timestamp_file, output_folder, frame_skip=5):
        self.video_path = video_path
        self.timestamp_file = timestamp_file
        self.output_folder = output_folder
        self.frame_skip = frame_skip
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.video_output_folder = os.path.join(output_folder, self.video_name)
        os.makedirs(self.video_output_folder, exist_ok=True)


    def extract_frames(self):
        """
        Extracts frames from a video based on timestamps.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        df = pd.read_excel(self.timestamp_file)
        df["Action"] = df["Action"].dropna()

        phases = df["Action"].to_list()
        start_times = df["Start Time (s)"].to_list()
        end_times = df["End Time (s)"].to_list()

        frame_count = 0
        extracted_count =0 
        
        
        with tqdm(total=total_frames, desc=f"Extracting {self.video_name}") as pbar:            
            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break

                current_phase = self.get_current_phase(frame_count, phases, start_times, end_times, fps)

                frame_filename = os.path.join(self.video_output_folder, f"frame_{frame_count:06d}_{current_phase}.jpg")
                cv2.imwrite(frame_filename, frame)

                frame_count += self.frame_skip
                extracted_count += 1
                pbar.update(self.frame_skip)

        cap.release()
        print(f"Extracted frames saved in {self.video_output_folder}")

    def get_current_phase(self, frame_count, phases, start_times, end_times, fps):
        """
        Determines the phase of the current frame.
        """
        for i, phase in enumerate(phases):
            start_frame = int(start_times[i] * fps)
            end_frame = int(end_times[i] * fps)
            if start_frame <= frame_count < end_frame:
                return phase
        return "unknown"
        
        
 