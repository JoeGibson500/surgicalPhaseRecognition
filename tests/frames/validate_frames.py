import os
import cv2

def check_frame_integrity(frames_folder):
    """
    Checks if frames are corrupt or missing.
    """
    corrupt_files = []
    
    for phase in os.listdir(frames_folder):
        phase_path = os.path.join(frames_folder, phase)
        if not os.path.isdir(phase_path):
            continue

        for frame_file in os.listdir(phase_path):
            if not frame_file.endswith(".jpg"):
                continue

            frame_path = os.path.join(phase_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                corrupt_files.append(frame_path)

    if corrupt_files:
        print("\nCorrupt Frame Files Found:")
        for file in corrupt_files[:10]:  # Display first 10
            print(file)
    else:
        print("\nAll frames are valid!")

import os
import pandas as pd

def validate_frame_extraction(video_name, timestamp_file, frames_folder):
    """
    Validates extracted frames to check if they are stored in the correct phase folder.
    """
    df = pd.read_excel(timestamp_file)
    df["Action"] = df["Action"].dropna()

    phases = df["Action"].to_list()
    start_times = df["Start Time (s)"].to_list()
    end_times = df["End Time (s)"].to_list()

    incorrect_frames = []

    for phase in phases:
        phase_folder = os.path.join(frames_folder, f"{phase}_frames")
        if not os.path.exists(phase_folder):
            print(f"Warning: Phase folder missing - {phase_folder}")
            continue

        for frame_file in os.listdir(phase_folder):
            if not frame_file.endswith(".jpg"):
                continue

            frame_name = os.path.splitext(frame_file)[0]
            frame_id = int(frame_name.split("_")[1])

            # Check if the frame belongs to the correct phase
            found_correct_phase = False
            for i in range(len(phases)):
                start_frame = int(start_times[i] * 30)  # Assuming 30 FPS
                end_frame = int(end_times[i] * 30)

                if start_frame <= frame_id < end_frame:
                    expected_phase = phases[i]
                    if expected_phase == phase:
                        found_correct_phase = True
                    break

            if not found_correct_phase:
                incorrect_frames.append((frame_file, phase, expected_phase))

    if incorrect_frames:
        print("\nIncorrectly Assigned Frames:")
        for frame, current_phase, correct_phase in incorrect_frames[:10]:
            print(f"Frame: {frame} → Incorrect Phase: {current_phase}, Should be: {correct_phase}")
    else:
        print("\nAll frames are correctly assigned!")

# Example Usage
if __name__ == "__main__":
    video_name = "1002_21"
    timestamp_file = "../data/video_timestamps/1002_21.xlsx"
    frames_folder = "../data/frames_per_phase/1002_21"
    
    validate_frame_extraction(video_name, timestamp_file, frames_folder)
    check_frame_integrity(frames_folder)
