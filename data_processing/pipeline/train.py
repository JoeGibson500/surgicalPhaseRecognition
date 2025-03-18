from feature_extraction import FeatureExtractor
from generate_sequences import PhaseFrameGrouper, PhaseSequenceGenerator

"""
Main pipeline for training the model


Add more desc...

"""

TRAINING_SPLIT = "data/splits/train_split.csv"
IMAGE_DIR = "/vol/scratch/SoC/misc/2024/sc22jg/frames/"
FEATURE_SAVE_PATH = "data/features/"

print("Runnning feature extraction...")

# # Create FeatureExtractor instance and run feature extraction
# extractor = FeatureExtractor(TRAINING_SPLIT, IMAGE_DIR, FEATURE_SAVE_PATH)
# extractor.extract_features()

print("Running phase frame grouper...")

# phase_grouper = PhaseFrameGrouper(TRAINING_SPLIT)
# phase_grouper.load_data()
# phase_grouper.generate_phase_ranges()
# phase_grouper.save_to_json()

GROUPED_PHASES_FILE = "data/sequences/train_phases.json"
SEQUENCE_LENGTH = 20
STRIDE = 5

print("Runnning sequence generation...")

sequence_generator = PhaseSequenceGenerator(GROUPED_PHASES_FILE, SEQUENCE_LENGTH, STRIDE)
sequence_generator.load_phase_data()
sequence_generator.generate_sequences()
sequence_generator.save_sequences()