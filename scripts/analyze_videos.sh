#!/bin/bash
# Analyze video metadata (FPS, duration, resolution) for all videos

mkdir -p reports  # Ensure reports folder exists

# Loop through all MKV videos in dataset
for video in data/full_videos/*.mkv; do
    filename=$(basename "$video")
    echo "Extracting $filename data ..."
    ffmpeg -i "$video" 2>&1 | grep -E "Duration|Stream"
done | tee reports/video_metadata.txt

echo "Analysis complete. Results saved to reports/video_metadata.txt"
