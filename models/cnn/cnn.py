import torch 
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights



class CNNFeatureExtractor(nn.Module):
    
    
    def __init__(self):
        
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        self.cnn.fc = nn.Identity() # remove last layer as we are not classifying just extracting features
        
    def forward(self, x): 
        x = self.cnn(x)  
        return x