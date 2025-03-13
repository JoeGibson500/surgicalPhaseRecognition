# import os
# import cv2
# import pandas as pd

# def extract_all_frames_with_phases(video_path, timestamp_file, output_folder, frame_skip=5):
#     """
#     Extracts every `frame_skip` frames from a video and stores them under their video name folder with phase and frame number.

#     Args:
#     - video_path (str): Path to the surgical video file.
#     - timestamp_file (str): Path to the Excel file containing phase timestamps.
#     - output_folder (str): Directory where extracted frames will be saved.
#     - frame_skip (int): Number of frames to skip between extractions.

#     Output:
#     - Saves frames inside `output_folder/video_name/` with filenames including phase and frame number.
#     """
#     video_name = os.path.splitext(os.path.basename(video_path))[0]  # Remove extension
#     video_output_folder = os.path.join(output_folder, video_name)
#     os.makedirs(video_output_folder, exist_ok=True)  # Ensure video folder exists

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown

#     df = pd.read_excel(timestamp_file)
#     df["Action"] = df["Action"].dropna()

#     # Store phase information
#     phases = df["Action"].to_list()
#     start_times = df["Start Time (s)"].to_list()
#     end_times = df["End Time (s)"].to_list()

#     frame_count = 0
#     while cap.isOpened():
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Move to correct frame
#         ret, frame = cap.read()
#         if not ret:
#             break  # Stop if the video ends

#         # Determine the phase for the current frame
#         current_phase = "unknown"
#         for i, phase in enumerate(phases):
#             start_frame = int(start_times[i] * fps)
#             end_frame = int(end_times[i] * fps)

#             if start_frame <= frame_count < end_frame:
#                 current_phase = phase
#                 break

#         # Save the frame under video_name/ with phase and frame number in filename
#         frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:06d}_{current_phase}.jpg")
#         cv2.imwrite(frame_filename, frame)

#         frame_count += frame_skip  # Skip frames to reduce storage usage

#     cap.release()
#     print(f"Extracted every {frame_skip}th frame and stored in {video_output_folder}")
import os
import cv2
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FrameExtractor:
    
    def __init__(self, video_path, timestamp_file, output_folder, frame_skip=5):
        self.video_path = video_path
        self.timestamp_file = timestamp_file
        self.output_folder = output_folder
        self.frame_skip = frame_skip
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.video_output_folder = os.path.join(output_folder, self.video_name)

        # Ensure the output folder exists
        os.makedirs(self.video_output_folder, exist_ok=True)

    def extract_frames(self):
        """
        Extracts frames from a video based on timestamps, ensuring correct frame skipping.
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        df = pd.read_excel(self.timestamp_file)
        df["Action"] = df["Action"].dropna()

        phases = df["Action"].to_list()
        start_times = df["Start Time (s)"].to_list()
        end_times = df["End Time (s)"].to_list()

        logger.info(f"Extracting frames from {self.video_name} (Total Frames: {total_frames}, FPS: {fps}, Frame Skip: {self.frame_skip})")

        saved_frames = 0
        frame_count = 0

        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Move to the correct frame
            ret, frame = cap.read()  # Read only the selected frame

            if not ret or frame is None:
                logger.warning(f"Failed to read frame at {frame_count} from {self.video_name}")
                break  # Stop if the video ends or OpenCV fails to read the frame

            # Determine phase for this frame
            current_phase = self.get_current_phase(frame_count, phases, start_times, end_times, fps)

            # Ensure frame is saved in the correct directory
            frame_filename = os.path.join(self.video_output_folder, f"frame_{frame_count:06d}_{current_phase}.jpg")
            cv2.imwrite(frame_filename, frame)

            if not os.path.exists(frame_filename):  # Verify if frame was actually saved
                logger.error(f"Frame was not saved: {frame_filename}")

            saved_frames += 1

            # Move forward by `frame_skip` frames
            frame_count += self.frame_skip

        cap.release()
        logger.info(f"Extracted {saved_frames} frames from {self.video_name}, saved to {self.video_output_folder}")

    def get_current_phase(self, frame_count, phases, start_times, end_times, fps):
        """
        Determines the phase of the current frame.
        """
        for i, phase in enumerate(phases):
            start_frame = int(start_times[i] * fps)
            end_frame = int(end_times[i] * fps)
            if start_frame <= frame_count < end_frame:
                return phase
        return "unknown"
