import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load extracted features and labels
features = np.load("data/features/cnn_features.npy")
labels = np.load("data/features/cnn_labels.npy")
frame_names = np.load("data/features/cnn_frame_names.npy")

# Print dataset details
print(f"Feature Shape: {features.shape}")  # Expected: (num_samples, 2048) for ResNet50
print(f"Labels Shape: {labels.shape}")  # Expected: (num_samples,)
print(f"Example Frame Name: {frame_names[1]}")
print(f"Example Feature Vector: {features[1]}")

print(f"Feature Mean: {features.mean():.4f}")
print(f"Feature Std: {features.std():.4f}")
print(f"Min Value: {features.min():.4f}, Max Value: {features.max():.4f}")


# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2D = tsne.fit_transform(features[:10000])  # Use a subset for speed

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(features_2D[:, 0], features_2D[:, 1], c=labels[:10000], cmap="coolwarm", alpha=0.5)
plt.colorbar(label="Phase Label")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of CNN Features")
plt.show()
