import torch 
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class TCNBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        
        super(TCNBlock, self).__init__()
        
        
        self.basicLayer1 = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size, padding = (kernel_size-1) * dilation, dilation=dilation))
        
        self.basicLayer2 = weight_norm(nn.Conv1d(output_channels, output_channels, kernel_size, padding = (kernel_size-1) * dilation, dilation=dilation))
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()()
        
        
    def forward(self, x):  
        
        residual = self.residual(x)
        
        
        # Layer 1
        x = self.basicLayer1(x)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # Layer 2 
        x = self.basicLayer2(x)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        return x + residual
    
    
    
class TCN(nn.Module):
    
    def __init__(self, input_dim, num_classes, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        
        super(TCN, self).__init__()

        layers = []
        num_layers = len(num_channels)

        for i in range(num_layers):
            
            input_channels = input_dim if i == 0 else num_channels[i-1]
            output_channels = num_channels[i]
            dilation = 2 ** i  # Expand receptive field exponentially

            layers.append(TCNBlock(input_channels, output_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        
        self.fc = nn.Linear(num_channels[-1], num_classes)  # Final classification layer

    def forward(self, x):
        
        # x: Shape (batch, seq_len, feature_dim)
        
        x = x.transpose(1, 2)  # Convert to (batch, feature_dim, seq_len) for Conv1d
        
        x = self.network(x)
        
        x = x.mean(dim=-1)  # Global average pooling over time
        
        x = self.fc(x)  # Classification layer
        return x