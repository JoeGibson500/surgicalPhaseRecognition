from metadata_extractor import process_all_videos, save_metadata_as_csv, VIDEO_METADATA_FILE, VIDEO_DIR

if __name__ == "__main__":
    print("\nExtracting video metadata...\n")
    video_metadata = process_all_videos(VIDEO_DIR)

    print("\nSaving metadata results...\n")
    save_metadata_as_csv(video_metadata, VIDEO_METADATA_FILE)

    print("\nVideo metadata extraction completed.")
