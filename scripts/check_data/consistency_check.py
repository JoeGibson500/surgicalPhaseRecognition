import os
import pandas as pd

# Paths
video_dir = "data/full_videos"
timestamp_dir = "data/video_timestamps"
log_file = "reports/logs/data_consistency.log"

# Ensure reports directory exists
os.makedirs("reports/logs", exist_ok=True)

def check_missing_files():
    """ Checks for missing timestamp files and extra timestamp files without corresponding videos. """
    video_files = {f.replace(".mkv", "") for f in os.listdir(video_dir) if f.endswith(".mkv")}
    timestamp_files = {f.replace("_Clip_List.xlsx", "") for f in os.listdir(timestamp_dir) if f.endswith("_Clip_List.xlsx")}

    missing_timestamps = video_files - timestamp_files
    extra_timestamps = timestamp_files - video_files

    with open(log_file, "w") as log:
        log.write("=== Missing and Extra Timestamp Files Check ===\n")
        if missing_timestamps:
            log.write(f"Missing timestamp files for videos: {missing_timestamps}\n")
        else:
            log.write("All videos have a corresponding timestamp file .\n")

        if extra_timestamps:
            log.write(f"Extra timestamp files without corresponding videos: {extra_timestamps}\n")
        else:
            log.write("All timestamp files have a corresponding video file.\n")

def check_column_consistency():
    """ Ensures all timestamp files contain the correct columns. """
    expected_columns = {"Start Time (s)", "End Time (s)", "Action", "Clip"}
    column_issues = {}

    for file in os.listdir(timestamp_dir):
        if file.endswith("_Clip_List.xlsx"):
            file_path = os.path.join(timestamp_dir, file)
            df = pd.read_excel(file_path)

            missing_columns = expected_columns - set(df.columns)
            extra_columns = set(df.columns) - expected_columns

            if missing_columns or extra_columns:
                column_issues[file] = {"missing": missing_columns, "extra": extra_columns}

    with open(log_file, "a") as log:
        log.write("\n=== Column Consistency Check ===\n")
        if column_issues:
            log.write("Column inconsistencies detected:\n")
            for file, issues in column_issues.items():
                log.write(f"- {file}: Missing {issues['missing']}, Extra {issues['extra']}\n")
        else:
            log.write("All timestamp files have the correct columns.\n")


def extract_unique_phase_labels():
    """ Extracts all unique surgical phase labels from the timestamp files. """
    all_phases = set()

    for file in os.listdir(timestamp_dir):
        if file.endswith("_Clip_List.xlsx"):
            file_path = os.path.join(timestamp_dir, file)
            df = pd.read_excel(file_path)

            if "Action" in df.columns:
                all_phases.update(df["Action"].dropna().str.strip().str.lower())

    with open(log_file, "a") as log:
        log.write("\n=== Unique Phase Labels Extracted ===\n")
        if all_phases:
            for phase in sorted(all_phases):
                log.write(f"- {phase}\n")
        else:
            log.write("No phase labels found in the timestamp files.\n")
