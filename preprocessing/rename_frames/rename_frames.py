import os
import sys
import re

# Define the path to the frames folder
FRAMES_FOLDER = "/vol/scratch/SoC/misc/2024/sc22jg/frames/"

# Add project path (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import clean_phase_name  # Import the cleaning function

# Iterate through all video folders inside FRAMES_FOLDER
for video in os.listdir(FRAMES_FOLDER):
    video_path = os.path.join(FRAMES_FOLDER, video)

    # Ensure it's a directory
    if not os.path.isdir(video_path):
        continue  

    # Iterate through all frame files in the video folder
    for filename in os.listdir(video_path):
        file_path = os.path.join(video_path, filename)

        # Skip non-image files
        if not filename.lower().endswith((".jpg")):
            continue

        # Extract phase name (everything after the second underscore)
        phase_name = filename.split('_', 2)[-1].replace(".jpg", "")

        # Clean the phase name
        cleaned_phase = clean_phase_name(phase_name)

        # Construct the new filename
        new_filename = filename.replace(phase_name, cleaned_phase)
        new_file_path = os.path.join(video_path, new_filename)

        # Rename file if necessary
        if new_file_path != file_path:
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} → {new_filename}")
