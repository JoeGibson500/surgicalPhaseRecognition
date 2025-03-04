import os

directories = ["data/full_videos", "data/video_timestamps"]

# Repeat for video data and timestamp data
for directory in directories:
    
    # Iterate over all files in the directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)  # Full path to the file

        # Check if "Task" is in the filename
        if "Task" in file:
            new_name = file.replace("Task", "")  # Remove "Task"
            new_path = os.path.join(directory, new_name)  # New file path
            
            os.rename(file_path, new_path)  # Rename the file
            print(f"Renamed: {file} -> {new_name}")