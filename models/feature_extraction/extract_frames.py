from frame_extraction import extract_frames_per_phase

TIMESTAMP_FOLDER = "data/video_timestamps/"
VIDEO_FOLDER = "data/full_videos"
OUTPUT_FOLDER_PHASE = "data/frames_per_phase/"
OUTPUT_FOLDER_VIDEO = "data/frames_per_video"

def main(): 
    
    print("Running frame extraction per phase...")
    extract_frames_per_phase(video_folder=VIDEO_FOLDER, timestamp_folder=TIMESTAMP_FOLDER, output_folder= OUTPUT_FOLDER_PHASE)
 
    print("Running frame extraction per video...")
    extract_frames_per_video(video_folder=VIDEO_FOLDER, timestamp_folder=TIMESTAMP_FOLDER, output_folder= OUTPUT_FOLDER)

if __name__ == "__main__":
    main()