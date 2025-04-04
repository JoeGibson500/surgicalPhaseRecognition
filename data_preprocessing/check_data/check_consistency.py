from scripts.check_data.consistency_checker import check_missing_files, check_column_consistency, extract_unique_phase_labels
from consistency_checker import ConsistencyChecker

VIDEO_DIR = "data/full_videos"
TIMESTAMP_DIR = "data/video_timestamps"
LOG_FILE = "reports/logs/data_consistency.log"


def check_data_consistency():
    
    # Define consistency checker
    consistency_checker = ConsistencyChecker(VIDEO_DIR, TIMESTAMP_DIR, LOG_FILE)
    
    print("Running missing files check...\n")
    consistency_checker.check_missing_files()
    
    print("Running column consistency check...\n")
    consistency_checker.check_column_consistency()
    
    print("Running unique phase labels...\n")
    consistency_checker.extract_unique_phase_labels()
    
    print(f"Consistency check complete - results saved in {LOG_FILE}")


if __name__ == "__main__":
    check_data_consistency()

