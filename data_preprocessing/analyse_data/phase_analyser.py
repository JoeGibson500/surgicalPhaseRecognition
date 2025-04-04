import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Apply consistent visual settings for all plots
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.alpha": 0.5,
    "axes.grid": True
})


class PhaseAnalyser:
    """Analyzes surgical phase data from timestamp files and generates visuals."""

    def __init__(self, timestamp_folder, frequency_graph_file, transition_graph_file, phase_lengths_graph):
        self.timestamp_folder = timestamp_folder
        self.frequency_graph_file = frequency_graph_file
        self.transition_graph_file = transition_graph_file
        self.phase_lengths_graph = phase_lengths_graph

        os.makedirs("reports", exist_ok=True)
        os.makedirs("reports/visuals/phases", exist_ok=True)

    def __repr__(self):
        return f"<PhaseAnalyser at '{self.timestamp_folder}'>"

    @staticmethod
    def clean_phase_name(phase):
        """Normalize phase names by removing unwanted suffixes and formatting."""
        phase = phase.strip().lower()
        phase = re.sub(r"\s*\(attempt\)|\s*\(partial\)", "", phase)
        return phase

    def _get_excel_files(self):
        """Return list of .xlsx files in the timestamp folder."""
        return [f for f in os.listdir(self.timestamp_folder) if f.endswith(".xlsx")]

    def _save_plot(self, path, title):
        """Save a plot to disk and confirm the output."""
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"{title} saved at: {path}")

    def get_unique_phases(self):
        """Extract and return all unique surgical phases across all timestamp files."""
        unique_phases = set()
        for file in self._get_excel_files():
            df = pd.read_excel(os.path.join(self.timestamp_folder, file))
            if "Action" in df.columns:
                df["Action"] = df["Action"].dropna().apply(self.clean_phase_name)
                unique_phases.update(df["Action"].tolist())
        return list(unique_phases)

    def compute_phase_frequencies(self):
        """Count how many times each phase appears across all timestamp files."""
        all_phases = []
        for file in self._get_excel_files():
            df = pd.read_excel(os.path.join(self.timestamp_folder, file))
            if "Action" in df.columns:
                cleaned_phases = df["Action"].dropna().apply(self.clean_phase_name).tolist()
                all_phases.extend(cleaned_phases)
        return pd.Series(all_phases).value_counts()

    def visualise_phase_frequencies(self):
        """Plot and save a bar chart showing how frequently each phase appears."""
        phase_counts = self.compute_phase_frequencies()
        avg_count = np.mean(phase_counts)

        plt.figure()
        sns.barplot(x=phase_counts.index, y=phase_counts.values, color="royalblue", edgecolor="black")
        plt.axhline(avg_count, color="red", linestyle="dashed", linewidth=1.5, label=f"Avg Frequency: {avg_count:.2f}s")
        plt.xlabel("Surgical Phase")
        plt.ylabel("Frequency")
        plt.title("Distribution of Surgical Phases")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        self._save_plot(self.frequency_graph_file, "Phase frequency graph")

    def extract_phase_transitions(self):
        """Extract sequences of phase-to-phase transitions from each file."""
        phase_durations = {}
        for file in self._get_excel_files():
            df = pd.read_excel(os.path.join(self.timestamp_folder, file))
            if "Action" in df.columns:
                df["Action"] = df["Action"].dropna().apply(self.clean_phase_name)
                sequence = df["Action"].tolist()
                for i in range(len(sequence) - 1):
                    phase_durations.setdefault(sequence[i], []).append(sequence[i + 1])
        return phase_durations

    def compute_phase_transition_matrix(self):
        """Build a normalized matrix showing probabilities of transitioning between phases."""
        phase_durations = self.extract_phase_transitions()
        unique_phases = self.get_unique_phases()
        phase_to_idx = {phase: i for i, phase in enumerate(unique_phases)}
        transition_matrix = np.zeros((len(unique_phases), len(unique_phases)))

        for phase, next_phases in phase_durations.items():
            for next_phase in next_phases:
                transition_matrix[phase_to_idx[phase], phase_to_idx[next_phase]] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        mask = row_sums.ravel() > 0 # remove non zero rows
        transition_matrix[mask, :] /= row_sums[mask]
        return transition_matrix, unique_phases

    def visualise_phase_transitions(self):
        """Visualise phase-to-phase transitions as a heatmap."""
        transition_matrix, phases = self.compute_phase_transition_matrix()
        phase_counts = self.compute_phase_frequencies()
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
        self._save_plot(self.transition_graph_file, "Phase transition heatmap")

    def extract_phase_lengths(self):
        """Extract the length (in seconds) of each phase from each file."""
        phase_lengths = {}
        for file in self._get_excel_files():
            df = pd.read_excel(os.path.join(self.timestamp_folder, file))
            if "Action" in df.columns and "Start Time (s)" in df and "End Time (s)" in df:
                df["Action"] = df["Action"].dropna().apply(self.clean_phase_name)
                for i in range(len(df)):
                    phase = df.loc[i, "Action"]
                    duration = df.loc[i, "End Time (s)"] - df.loc[i, "Start Time (s)"]
                    phase_lengths.setdefault(phase, []).append(duration)
        return phase_lengths

    def compute_average_phase_length(self):
        """Calculate the average duration for each surgical phase."""
        return {phase: np.mean(lengths) for phase, lengths in self.extract_phase_lengths().items()}

    def visualise_average_phase_lengths(self):
        """Plot and save a bar chart of average phase durations, ordered by frequency."""
        phase_lengths = self.compute_average_phase_length()
        sorted_phases = self.compute_phase_frequencies().index.tolist()
        sorted_lengths = [phase_lengths.get(phase, np.nan) for phase in sorted_phases]

        plt.figure()
        sns.barplot(x=sorted_phases, y=sorted_lengths, color="royalblue", edgecolor="black")
        plt.xlabel("Surgical Phase")
        plt.ylabel("Length (s)")
        plt.title("Distribution of Surgical Lengths (Ordered by Frequency)")
        plt.xticks(rotation=45, ha="right")
        self._save_plot(self.phase_lengths_graph, "Phase length graph")
