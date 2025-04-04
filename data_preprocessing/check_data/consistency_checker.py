import os
import pandas as pd

class ConsistencyChecker:
    """Performs consistency checks between video files and timestamp metadata."""

    def __init__(self, video_dir, timestamp_dir, log_file):
        self.video_dir = video_dir
        self.timestamp_dir = timestamp_dir
        self.log_file = log_file

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def check_missing_files(self):
        """Checks for missing timestamp files and extra timestamp files without corresponding videos."""
        video_files = {f.replace(".mkv", "") for f in os.listdir(self.video_dir) if f.endswith(".mkv")}
        timestamp_files = {f.replace("_Clip_List.xlsx", "") for f in os.listdir(self.timestamp_dir) if f.endswith("_Clip_List.xlsx")}

        missing_timestamps = video_files - timestamp_files
        extra_timestamps = timestamp_files - video_files

        with open(self.log_file, "w") as log:
            log.write("=== Missing and Extra Timestamp Files Check ===\n")
            if missing_timestamps:
                log.write(f"Missing timestamp files for videos: {missing_timestamps}\n")
            else:
                log.write("All videos have a corresponding timestamp file.\n")

            if extra_timestamps:
                log.write(f"Extra timestamp files without corresponding videos: {extra_timestamps}\n")
            else:
                log.write("All timestamp files have a corresponding video file.\n")

    def check_column_consistency(self):
        """Ensures all timestamp files contain the correct columns."""
        expected_columns = {"Start Time (s)", "End Time (s)", "Action", "Clip"}
        column_issues = {}

        for file in os.listdir(self.timestamp_dir):
            if file.endswith("_Clip_List.xlsx"):
                file_path = os.path.join(self.timestamp_dir, file)
                df = pd.read_excel(file_path)

                missing_columns = expected_columns - set(df.columns)
                extra_columns = set(df.columns) - expected_columns

                if missing_columns or extra_columns:
                    column_issues[file] = {"missing": missing_columns, "extra": extra_columns}

        with open(self.log_file, "a") as log:
            log.write("\n=== Column Consistency Check ===\n")
            if column_issues:
                log.write("Column inconsistencies detected:\n")
                for file, issues in column_issues.items():
                    log.write(f"- {file}: Missing {issues['missing']}, Extra {issues['extra']}\n")
            else:
                log.write("All timestamp files have the correct columns.\n")

    def extract_unique_phase_labels(self):
        """Extracts all unique surgical phase labels from the timestamp files."""
        all_phases = set()

        for file in os.listdir(self.timestamp_dir):
            if file.endswith("_Clip_List.xlsx"):
                file_path = os.path.join(self.timestamp_dir, file)
                df = pd.read_excel(file_path)

                if "Action" in df.columns:
                    all_phases.update(df["Action"].dropna().str.strip().str.lower())

        with open(self.log_file, "a") as log:
            log.write("\n=== Unique Phase Labels Extracted ===\n")
            if all_phases:
                for phase in sorted(all_phases):
                    log.write(f"- {phase}\n")
            else:
                log.write("No phase labels found in the timestamp files.\n")
