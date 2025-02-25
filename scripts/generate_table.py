import re

# Define input and output files
input_file = "reports/video_metadata.txt"
output_file = "reports/video_metadata.md"

# Read the metadata file
with open(input_file, "r") as file:
    lines = file.readlines()

videos = []
current_video = {}

# Process the metadata
for line in lines:
    if line.strip().startswith("Extracting") and "data ..." in line:
        if current_video:  # Save the last processed video
            videos.append(current_video)
        filename = line.strip().split(" ")[1]  # Extract filename
        current_video = {"Video Name": filename}  # Start new entry

    elif "Duration:" in line:
        match = re.search(r"Duration: (\d+:\d+:\d+\.\d+)", line)
        if match:
            current_video["Duration"] = match.group(1)

    elif "Stream" in line and "Video: h264" in line:
        resolution_match = re.search(r"(\d+x\d+)", line)
        fps_match = re.search(r"(\d+\.\d+) fps", line)
        current_video["Resolution"] = resolution_match.group(1) if resolution_match else "N/A"
        current_video["FPS"] = fps_match.group(1) if fps_match else "N/A"

if current_video:  # Add the last video entry
    videos.append(current_video)

# Write output to Markdown
with open(output_file, "w") as file:
    file.write("## Metadata Table\n\n")
    file.write("| Video Name | Duration | Resolution | FPS |\n")
    file.write("|------------|----------|------------|------|\n")

    for video in videos:
        file.write(f"| {video.get('Video Name', 'N/A')} | {video.get('Duration', 'N/A')} | {video.get('Resolution', 'N/A')} | {video.get('FPS', 'N/A')} |\n")

print(f"Markdown table saved to {output_file}")
