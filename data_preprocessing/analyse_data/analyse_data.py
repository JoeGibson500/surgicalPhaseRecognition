from phase_analyser import PhaseAnalyser
from video_analyser import VideoAnalyser

# === Paths for phase analysis === 
TIMESTAMP_FOLDER = "data/video_timestamps/"
FREQ_GRAPH_FILE = "reports/visuals/phases/phase_frequency_graphs.png"
TRANSITION_GRAPH_FILE = "reports/visuals/phases/phase_transition_heatmap.png"
PHASE_LENGTHS_GRAPH = "reports/visuals/phases/phase_lengths.png"

# === Paths for video analysis === 
VIDEO_METADATA_PATH = "reports/video_metadata.csv"
VIDEO_STATS_SAVE_PATH = "reports/visuals/videos/video_durations_boxplot_labeled.png"


def analyse_videos_and_phases():
    
    # Define phase analyser 
    phase_analyser = PhaseAnalyser(TIMESTAMP_FOLDER, FREQ_GRAPH_FILE, TRANSITION_GRAPH_FILE, PHASE_LENGTHS_GRAPH)
    
    # === Phase frequency ===
    print("Running Phase Frequency Analysis...\n")
    phase_analyser.visualise_phase_frequencies()
    print(f"Phase frequencies saved at: {FREQ_GRAPH_FILE}\n")
    
    # === Phase transitions ===
    print("Running Phase Transition Analysis...\n")
    phase_analyser.visualise_phase_transitions()
    print(f"Phase transitions saved at: {TRANSITION_GRAPH_FILE}\n")
    
    # === Phase lengths ===
    print("Running Phase Length Analysis...\n")
    phase_analyser.visualise_average_phase_lengths()
    
    # Define video analyser
    video__analyser = VideoAnalyser(VIDEO_METADATA_PATH, VIDEO_STATS_SAVE_PATH)

    # === Video durations ===
    print("Running Video Duration Analysis...\n")
    video__analyser.visualise_video_durations()
    print(f"Video durations saved at: {VIDEO_STATS_SAVE_PATH}\n")
    
    print("Analysis complete.")
    
if __name__ == "__main__":
    analyse_videos_and_phases()
