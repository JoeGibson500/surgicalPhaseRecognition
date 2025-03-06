import pandas as pd
import cv2
import os
import sys
import os

# Get the absolute path of the project root for helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import clean_phase_name 

def extract_frames_per_phase(timestamp_folder, video_folder, output_folder):
    """
    Extracts frames from surgical videos per phase based on timestamps from Excel files.

    Args:
    - timestamp_folder (str): Path to folder containing Excel files with phase timestamps.
    - video_folder (str): Path to folder containing surgical videos.
    - output_folder (str): Directory to save extracted frames.

    Output:
    - Saves extracted frames per phase inside `output_folder/video_name/phase_name/`
    """
    
    # Get list of all timestamp files
    timestamp_files = [f for f in os.listdir(timestamp_folder) if f.endswith(".xlsx")]

    for timestamp_file in timestamp_files: 
        # Extract video name from timestamp file
        video_name = timestamp_file.replace(".xlsx", ".mkv")  # Fix the replacement
        video_path = os.path.join(video_folder, video_name)
        
        print(f"Extracting phases from {video_name}...")
        
        # Ensure the video file exists
        if not os.path.exists(video_path):
            print(f"Warning: No matching video found for {timestamp_file}")
            continue  # Skip this file if no matching video is found
        
        # Create video output folder
        video_output_folder = os.path.join(output_folder, video_name.replace(".mkv", ""))
        os.makedirs(video_output_folder, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read timestamp data
        df = pd.read_excel(os.path.join(timestamp_folder, timestamp_file))
        df["Action"] = df["Action"].dropna().apply(clean_phase_name) # apply clean phase name

        phases = df["Action"].to_list()
        start_times = df["Start Time (s)"].to_list()
        end_times = df["End Time (s)"].to_list()

        for i, phase in enumerate(phases): 
            # Create phase folder inside the correct video folder
            phase_folder = os.path.join(video_output_folder, phase)
            os.makedirs(phase_folder, exist_ok=True)

            start_frame = int(start_times[i] * fps)
            end_frame = int(end_times[i] * fps)
            
            # Extract every frame within this phase's time window
            for frame_id in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not extract frame {frame_id} in {video_name}, phase: {phase}")
                    break  # Stop if we can't read a frame
                frame_name = os.path.join(phase_folder, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_name, frame)

        cap.release()  # Release the video after processing all phases

    print(f"Frames successfully extracted in {output_folder}")
