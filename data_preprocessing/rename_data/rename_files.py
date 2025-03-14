import os

directories = ["data/full_videos", "data/video_timestamps"]

# Repeat for video data and timestamp data
for directory in directories:
    
    # Iterate over all files in the directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)  # Full path to the file
        
        new_name = file

        # Check if "Task" is in the filename
        if "Task" in file:
            new_name = file.replace("Task", "")  # Remove "Task"
            
            print(f"Renamed: {file} -> {new_name}")
            
        if "_Clip_List" in file:
            
            new_name = file.replace("_Clip_List", "")  # Remove "_Clip_List"
            
            print(f"Renamed: {file} -> {new_name}")
            
            
        if new_name != file:
            
            new_path = os.path.join(directory, new_name)
            os.rename(file_path, new_path)