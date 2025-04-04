import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels,
                                          kernel_size,
                                          padding=0,  # we pad manually
                                          dilation=dilation))

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))  # left padding only
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual  # residual connection


class TCN(nn.Module):
    def __init__(self, input_dim, num_classes,
                #  num_channels=[64, 64, 64, 64],
                num_channels=[32,64,64,128],
                 kernel_size=3,
                 dropout=0.3):
        super(TCN, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i  # exponentially increasing
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    # def forward(self, x):
    #     # x shape: (batch, seq_len, features)
    #     x = x.transpose(1, 2)  # → (batch, features, seq_len)
    #     x = self.network(x)
    #     x = x.mean(dim=-1)    # global average over time
    #     return self.fc(x)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        embedding = x.mean(dim=-1)
        logits = self.fc(embedding)
        return logits, embedding

