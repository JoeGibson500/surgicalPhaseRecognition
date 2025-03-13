import os
import pandas as pd
import glob

# Path to your dataset
DATASET_PATH = "/vol/scratch/SoC/misc/2024/sc22jg/frames/"

# List all subdirectories (video IDs)
video_dirs = sorted(os.listdir(DATASET_PATH))

# Initialize metadata storage
data = []

for video_id in video_dirs:
    video_path = os.path.join(DATASET_PATH, video_id)
    if not os.path.isdir(video_path):
        continue  # Skip if not a directory

    # List all frame images in the video folder
    frame_files = glob.glob(os.path.join(video_path, "*.jpg"))

    for frame_file in frame_files:
        filename = os.path.basename(frame_file)
        
        # Extract frame number and phase
        parts = filename.split("_")
        frame_number = parts[1]
        phase = filename.split("_", 2)[2].replace(".jpg", "").strip()  # Get phase label
        
        # Store data
        data.append([video_id, frame_number, phase, frame_file])

# Convert to Pandas DataFrame
df = pd.DataFrame(data, columns=["video_id", "frame_number", "phase", "file_path"])

# Save metadata
df.to_csv("data/frames/frames_metadata.csv", index=False)

print("Metadata extraction complete. CSV saved.")
