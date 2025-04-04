import pandas as pd
import matplotlib.pyplot as plt
import os


class VideoAnalyser:
    """Analyzes video metadata to compute statistics and generate visuals."""

    def __init__(self, video_metadata_path, video_stats_save_path):
        self.video_metadata_path = video_metadata_path
        self.video_stats_save_path = video_stats_save_path

        os.makedirs("reports/visuals/videos", exist_ok=True)

    def load_video_metadata(self):
        """Load the video metadata CSV file."""
        return pd.read_csv(self.video_metadata_path, delimiter=",")

    @staticmethod
    def compute_statistics(video_metadata):
        """Compute basic statistics from the video durations."""
        num_videos = video_metadata.shape[0]
        video_durations = video_metadata["Duration (s)"]

        stats = {
            "num_videos": num_videos,
            "min_duration": video_durations.min(),
            "max_duration": video_durations.max(),
            "median_duration": video_durations.median(),
            "mean_duration": video_durations.mean()
        }

        return stats, video_durations

    def plot_video_durations(self, stats, video_durations):
        """Create and save a boxplot of video durations with annotated statistics."""
        plt.figure(figsize=(8, 5))
        plt.boxplot(video_durations, patch_artist=True, whis=2.5, boxprops=dict(facecolor="royalblue"))

        plt.xticks([])

        # Annotate key statistics on the plot
        plt.text(1.1, stats["min_duration"], f"Min: {stats['min_duration']}s", fontsize=10, color="black", ha="left")
        plt.text(1.1, stats["max_duration"], f"Max: {stats['max_duration']}s", fontsize=10, color="black", ha="left")
        plt.text(1.1, stats["median_duration"], f"Median: {stats['median_duration']}s", fontsize=10, color="red", ha="left")
        plt.text(1.1, stats["mean_duration"], f"Mean: {stats['mean_duration']:.2f}s", fontsize=10, color="green", ha="left")
        plt.text(1.1, stats["max_duration"] * 0.85, f"Total Videos: {stats['num_videos']}", fontsize=10, color="blue", ha="left")

        plt.title("Distribution of Video Durations")
        plt.ylabel("Duration (seconds)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.savefig(self.video_stats_save_path, dpi=300)

    def visualise_video_durations(self):
        """Run the full analysis and save the duration plot."""
        video_metadata = self.load_video_metadata()
        stats, video_durations = self.compute_statistics(video_metadata)
        self.plot_video_durations(stats, video_durations)

        print(f"Video duration graph saved at: {self.video_stats_save_path}")
