import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

#  Add project root to access helper functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.phase_utils import clean_phase_name

# Paths for data and output visuals
TIMESTAMP_FOLDER = "data/video_timestamps/"
FREQ_GRAPH_FILE = "reports/visuals/phases/phase_frequency_graphs.png"
TRANSITION_GRAPH_FILE = "reports/visuals/phases/phase_transition_heatmap.png"
PHASE_LENGTHS_GRAPH = "reports/visuals/phases/phase_lengths.png"

# Ensure required directories exist
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/visuals/phases", exist_ok=True)

# Standardized graph settings
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.alpha": 0.5,
    "axes.grid": True
})

# --- Helper Function ---
def get_unique_phases():
    """Extract unique surgical phases from all timestamp files."""
    unique_phases = set()
    for file in os.listdir(TIMESTAMP_FOLDER):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(TIMESTAMP_FOLDER, file))
            if "Action" in df.columns:
                df["Action"] = df["Action"].dropna().apply(clean_phase_name)
                unique_phases.update(df["Action"].tolist())
    return list(unique_phases)

# --- Phase Frequency Analysis ---
def compute_phase_frequencies():
    """Compute occurrence frequency of surgical phases."""
    all_phases = []
    for file in os.listdir(TIMESTAMP_FOLDER):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(TIMESTAMP_FOLDER, file))
            if "Action" in df.columns:
                cleaned_phases = df["Action"].dropna().apply(clean_phase_name).tolist()
                all_phases.extend(cleaned_phases)
    return pd.Series(all_phases).value_counts()

def visualise_phase_frequencies():
    """Plot and save a bar chart of phase frequencies."""
    phase_counts = compute_phase_frequencies()
    avg_count = np.mean(phase_counts)
    
    plt.figure()
    sns.barplot(x=phase_counts.index, y=phase_counts.values, color="royalblue", edgecolor="black")
    plt.axhline(avg_count, color="red", linestyle="dashed", linewidth=1.5, label=f"Avg Frequency: {avg_count:.2f}s")
    plt.xlabel("Surgical Phase")
    plt.ylabel("Frequency")
    plt.title("Distribution of Surgical Phases")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FREQ_GRAPH_FILE, dpi=300)
    print(f"Phase frequency graph saved at: {FREQ_GRAPH_FILE}")

# --- Phase Transition Analysis ---
def extract_phase_transitions():
    """Extract phase transition data from timestamp files."""
    phase_durations = {}
    for file in os.listdir(TIMESTAMP_FOLDER):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(TIMESTAMP_FOLDER, file))
            if "Action" in df.columns:
                df["Action"] = df["Action"].dropna().apply(clean_phase_name)
                sequence = df["Action"].tolist()
                for i in range(len(sequence) - 1):
                    phase_durations.setdefault(sequence[i], []).append(sequence[i + 1])
    return phase_durations

def compute_phase_transition_matrix():
    """Create transition matrix representing phase transitions."""
    phase_durations = extract_phase_transitions()
    unique_phases = get_unique_phases()
    phase_to_idx = {phase: i for i, phase in enumerate(unique_phases)}
    transition_matrix = np.zeros((len(unique_phases), len(unique_phases)))
    
    for phase, next_phases in phase_durations.items():
        for next_phase in next_phases:
            transition_matrix[phase_to_idx[phase], phase_to_idx[next_phase]] += 1
    
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    mask = row_sums.ravel() > 0  
    transition_matrix[mask, :] /= row_sums[mask]
    return transition_matrix, unique_phases

def visualise_phase_transitions():
    """Plot and save a heatmap of phase transitions."""
    transition_matrix, phases = compute_phase_transition_matrix()
    phase_counts = compute_phase_frequencies()
    sorted_phases = [phase for phase in phase_counts.index if phase in phases]
    
    phase_to_idx = {phase: i for i, phase in enumerate(sorted_phases)}
    reordered_matrix = np.zeros((len(sorted_phases), len(sorted_phases)))
    for i, phase in enumerate(sorted_phases):
        for j, next_phase in enumerate(sorted_phases):
            if phase in phases and next_phase in phases:
                reordered_matrix[i, j] = transition_matrix[phases.index(phase), phases.index(next_phase)]
    
    plt.figure()
    sns.heatmap(reordered_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sorted_phases, yticklabels=sorted_phases, linewidths=0.5, linecolor="black")
    plt.xlabel("Next Phase")
    plt.ylabel("Current Phase")
    plt.title("Structured Phase Transition Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(TRANSITION_GRAPH_FILE, dpi=300)
    print(f"Phase transition heatmap saved at: {TRANSITION_GRAPH_FILE}")

# --- Phase Length Analysis ---
def extract_phase_lengths():
    """Extract phase duration from timestamp files."""
    phase_lengths = {}
    for file in os.listdir(TIMESTAMP_FOLDER):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(TIMESTAMP_FOLDER, file))
            if "Action" in df.columns and "Start Time (s)" in df and "End Time (s)" in df:
                df["Action"] = df["Action"].dropna().apply(clean_phase_name)
                for i in range(len(df)):
                    phase = df.loc[i, "Action"]
                    duration = df.loc[i, "End Time (s)"] - df.loc[i, "Start Time (s)"]
                    phase_lengths.setdefault(phase, []).append(duration)
    return phase_lengths

def compute_average_phase_length():
    """Compute average duration of each phase."""
    return {phase: np.mean(lengths) for phase, lengths in extract_phase_lengths().items()}

def visualise_average_phase_lengths():
    """Plot and save bar chart of average phase lengths."""
    
    # To maintain consistency across charts order phases by most frequent phase
    phase_lengths = compute_average_phase_length()
    sorted_phases = compute_phase_frequencies().index.tolist()
    sorted_lengths = [phase_lengths.get(phase, np.nan) for phase in sorted_phases] 
    
    plt.figure()
    sns.barplot(x=sorted_phases, y=sorted_lengths, color="royalblue", edgecolor="black")
    plt.xlabel("Surgical Phase")
    plt.ylabel("Length (s)")
    plt.title("Distribution of Surgical Lengths (Ordered by Frequency)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PHASE_LENGTHS_GRAPH, dpi=300)
    print(f"Phase length graph saved at: {PHASE_LENGTHS_GRAPH}")
