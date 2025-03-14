from phase_analysis import visualise_phase_frequencies, visualise_phase_transitions, visualise_average_phase_lengths
from video_analysis import visualise_video_durations

def main():
    print("\nRunning Phase Frequency Analysis...")
    visualise_phase_frequencies()

    print("\nRunning Phase Transition Analysis...")
    visualise_phase_transitions()
    
    print("\nRunning Phase Length Analysis...")
    visualise_average_phase_lengths()
       
    print("\nRunning Video Duration Analysis")
    visualise_video_durations()
    
    print("\nAll analysis completed!")
    
if __name__ == "__main__":
    main()
