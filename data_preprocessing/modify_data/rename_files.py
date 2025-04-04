import os

directories = ["data/full_videos", "data/video_timestamps"]


def rename_files():
    """Function to remove inconsistencies in filenames"""
    for directory in directories:
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file) 

            if "Task" in file:
                new_name = file.replace("Task", "")  
                new_path = os.path.join(directory, new_name) 
                
                os.rename(file_path, new_path)  
                print(f"Renamed: {file} -> {new_name}")
                
                
                
if __name__ == "__main__":
    rename_files()  