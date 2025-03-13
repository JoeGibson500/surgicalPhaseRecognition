import pandas as pd
from sklearn.model_selection import train_test_split

# Load metadata
df = pd.read_csv("data/frames/frames_metadata.csv")

# Get unique videos and their corresponding phases
video_phase_map = df.groupby("video_id")["phase"].apply(set).to_dict()

# Convert video_id to DataFrame for stratified splitting
video_df = pd.DataFrame({"video_id": list(video_phase_map.keys()), "phases": list(video_phase_map.values())})

# Initial split (70% train, 30% temp)
train_videos, temp_videos = train_test_split(video_df["video_id"], test_size=0.3, random_state=42)

# Split temp further into validation (15%) and test (15%)
val_videos, test_videos = train_test_split(temp_videos, test_size=0.5, random_state=42)

# Ensure every phase exists in all splits
def check_phase_coverage(video_list, df):
    present_phases = set(df[df["video_id"].isin(video_list)]["phase"])
    return present_phases

train_phases = check_phase_coverage(train_videos, df)
val_phases = check_phase_coverage(val_videos, df)
test_phases = check_phase_coverage(test_videos, df)

missing_val_phases = train_phases - val_phases
missing_test_phases = train_phases - test_phases

# Fix missing phases by moving a few examples to val/test
for phase in missing_val_phases:
    sample_video = df[df["phase"] == phase]["video_id"].iloc[0]  # Get a video with the missing phase
    val_videos = pd.concat([val_videos, pd.Series([sample_video])], ignore_index=True)

for phase in missing_test_phases:
    sample_video = df[df["phase"] == phase]["video_id"].iloc[0]  # Get a video with the missing phase
    test_videos = pd.concat([test_videos, pd.Series([sample_video])], ignore_index=True)

# Save final splits
train_df = df[df["video_id"].isin(train_videos)]
val_df = df[df["video_id"].isin(val_videos)]
test_df = df[df["video_id"].isin(test_videos)]

train_df.to_csv("feature_extraction/split_dataset/csv_splits/train_split.csv", index=False)
val_df.to_csv("feature_extraction/split_dataset/csv_splits/val_split.csv", index=False)
test_df.to_csv("feature_extraction/split_dataset/csv_splits/test_split.csv", index=False)

print("Final train/val/test splits created successfully.")
