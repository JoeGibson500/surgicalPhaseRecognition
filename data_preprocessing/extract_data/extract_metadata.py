# from metadata_extractor import process_all_videos, save_metadata_as_csv, VIDEO_METADATA_FILE, VIDEO_DIR
from metadata_extractor import VideoMetadataExtractor


VIDEO_METADATA_FILE = "reports/video_metadata.csv"
VIDEO_DIR = "data/full_videos"
TIMESTAMP_DIR = "data/video_timestamps"

def extract_metadata():
    
    metadata_extractor = VideoMetadataExtractor(VIDEO_DIR, TIMESTAMP_DIR, VIDEO_METADATA_FILE)
    
    print("Extracting video metadata...\n")
    video_metadata = metadata_extractor.extract_data()

    print("Saving metadata results...\n")
    metadata_extractor.save_metadata_as_csv(video_metadata, VIDEO_METADATA_FILE)
    print(f"Video metadata saved to {VIDEO_METADATA_FILE}")

    print("Video metadata extraction completed.")


if __name__ == "__main__":
    extract_metadata()