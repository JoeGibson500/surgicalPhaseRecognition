import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)  

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.999, gamma=1.0, reduction='mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)

        self.alpha = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1))
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        alpha_t = self.alpha.to(inputs.device).gather(0, targets.squeeze())
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

