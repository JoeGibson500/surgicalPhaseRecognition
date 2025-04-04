 import os
import pandas as pd
import glob
import logging

logging.basicConfig(level=print)

class MetadataGenerator:
    
    def __init__(self, frame_folder, output_csv="data/frames/frames_metadata.csv"):
        self.frame_folder = frame_folder
        self.output_csv = output_csv
        
    def extract_frame_metadata(self):
        """
        Extracts metadata from extracted frames and saves it as a CSV file.
        """
        video_dirs = sorted(os.listdir(self.frame_folder))
        data = []
    
        for video_id in video_dirs: 
            video_path = os.path.join(self.frame_folder, video_id)
            if not os.path.isdir(video_path):
                continue  # Skip if not a directory
            
            frame_files = glob.glob(os.path.join(video_path, "*.jpg"))

            for frame_file in frame_files:
                filename = os.path.basename(frame_file)
                
                # Extract frame number and phase
                parts = filename.split("_")
                frame_number = parts[1]
                phase = filename.split("_", 2)[2].replace(".jpg", "").strip()  # Get phase label
                
                # Store data
                data.append([video_id, frame_number, phase, frame_file])

        df = pd.DataFrame(data, columns=["video_id", "frame_number", "phase", "file_path"])

        df.to_csv(self.output_csv, index=False)
        print(f"Metadata extraction complete. CSV saved at {self.output_csv}")
