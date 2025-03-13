import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Load a pretrained ResNet model
resnet = models.resnet50(pretrained=True)  # You can use resnet18 for faster inference

# Remove the final classification layer (fully connected layer)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer

# Set model to evaluation mode (no gradients needed)
feature_extractor.eval()

# Define image transform (match what ResNet expects)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Function to extract feature vector from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image
    image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension

    with torch.no_grad():  # No gradient calculation
        features = feature_extractor(image)  # Pass through ResNet
        features = features.view(features.shape[0], -1)  # Flatten to 1D vector

    return features.numpy()  # Convert to NumPy for further processing

# Example: Extract features from a single image
feature_vector = extract_features("example_frame.jpg")
print("Feature Vector Shape:", feature_vector.shape)  # Should be (1, 2048) for ResNet-50
