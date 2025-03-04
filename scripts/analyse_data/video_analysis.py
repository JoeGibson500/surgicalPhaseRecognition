import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure reports directory exists
os.makedirs("reports/visuals/videos", exist_ok=True)

def load_video_metadata(file_path="reports/video_metadata.csv"):
    """Loads the video metadata CSV file."""
    return pd.read_csv(file_path, delimiter=",")

def compute_statistics(df):
    """Computes basic statistics from the video durations."""
    num_videos = df.shape[0]
    video_durations = df["Duration (s)"]

    stats = {
        "num_videos": num_videos,
        "min_duration": video_durations.min(),
        "max_duration": video_durations.max(),
        "median_duration": video_durations.median(),
        "mean_duration": video_durations.mean()
    }
    
    return stats, video_durations

def plot_video_durations(video_durations, stats, save_path="reports/visuals/videos/video_durations_boxplot_labeled.png"):
    """Creates and saves a boxplot of video durations with labeled statistics."""
    plt.figure(figsize=(8, 5))
    plt.boxplot(video_durations, patch_artist=True, whis=2.5, boxprops=dict(facecolor="royalblue"))
    
    # Remove x-axis label "1"
    plt.xticks([])

    # Add statistics as text labels
    plt.text(1.1, stats["min_duration"], f"Min: {stats['min_duration']}s", fontsize=10, color="black", ha="left")
    plt.text(1.1, stats["max_duration"], f"Max: {stats['max_duration']}s", fontsize=10, color="black", ha="left")
    plt.text(1.1, stats["median_duration"], f"Median: {stats['median_duration']}s", fontsize=10, color="red", ha="left")
    plt.text(1.1, stats["mean_duration"], f"Mean: {stats['mean_duration']:.2f}s", fontsize=10, color="green", ha="left")
    plt.text(1.1, stats["max_duration"] * 0.85, f"Total Videos: {stats['num_videos']}", fontsize=10, color="blue", ha="left")

    # Styling
    plt.title("Distribution of Video Durations")
    plt.ylabel("Duration (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and display the plot
    plt.savefig(save_path, dpi=300)

    
def visualise_video_durations():
    """Main function to execute the full analysis workflow."""
    df = load_video_metadata()
    stats, video_durations = compute_statistics(df)
    plot_video_durations(video_durations, stats)
    
    print("Video duration graph saved at: reports/visuals/videos/video_durations.png")
