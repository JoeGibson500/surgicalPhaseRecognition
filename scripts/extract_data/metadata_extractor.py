import os
import cv2
import pandas as pd

# Ensure reports directory exists
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

VIDEO_METADATA_FILE = os.path.join(REPORTS_DIR, "video_metadata.csv")
VIDEO_DIR = "data/full_videos"
TIMESTAMP_DIR = "data/video_timestamps"

def extract_video_metadata(video_path):
    """Extracts basic metadata from a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "Video Name": os.path.basename(video_path),
        "Duration (s)": round(duration, 2),
        "Resolution": f"{width}x{height}",
        "FPS": round(fps, 2),
        "Frame Count": frame_count,
    }

def extract_timestamp_metadata(video_name):
    """Extracts metadata related to surgical phases from the corresponding timestamp file."""
    timestamp_file = os.path.join(TIMESTAMP_DIR, f"{video_name}_Clip_List.xlsx")

    if not os.path.exists(timestamp_file):
        print(f"Timestamp file missing for: {video_name}")
        return {
            "Number of Phases": None,
            "Total Phase Transitions": None
        }

    df = pd.read_excel(timestamp_file)

    if "Action" not in df.columns:
        print(f"Timestamp file format issue: {video_name}")
        return {
            "Number of Phases": None,
            "Total Phase Transitions": None
        }

    # Extract unique phase count
    unique_phases = df["Action"].dropna().str.strip().str.lower().unique()
    num_phases = len(unique_phases)

    # Count total phase transitions
    total_transitions = (df["Action"].shift() != df["Action"]).sum() - 1  # Ignore first entry

    return {
        "Number of Phases": num_phases,
        "Total Phase Transitions": total_transitions
    }

def process_all_videos(video_dir):
    """Processes all videos and extracts both video and timestamp metadata."""
    video_metadata = []

    for video in os.listdir(video_dir):
        if video.endswith((".mkv", ".mp4", ".avi", ".mov", ".flv")):
            video_name = os.path.splitext(video)[0]
            video_path = os.path.join(video_dir, video)

            # Extract metadata from video and timestamps
            video_info = extract_video_metadata(video_path)
            timestamp_info = extract_timestamp_metadata(video_name)

            if video_info:
                # Combine video and timestamp metadata
                video_info.update(timestamp_info)
                video_metadata.append(video_info)
                print(f"Processed {video}")

    return video_metadata

def save_metadata_as_csv(video_metadata, output_file):
    """Saves metadata as a CSV file."""
    if not video_metadata:
        print("No metadata extracted. Skipping file write.")
        return

    df = pd.DataFrame(video_metadata)
    df.to_csv(output_file, index=False)

    print(f"Video metadata saved to: {output_file}")
