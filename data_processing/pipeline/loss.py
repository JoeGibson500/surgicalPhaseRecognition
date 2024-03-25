import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         """
#         alpha: List or tensor of class weights (for imbalance), or None
#         gamma: Focusing parameter
#         reduction: 'mean', 'sum', or 'none'
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         if isinstance(alpha, list):
#             self.alpha = torch.tensor(alpha, dtype=torch.float32)
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         """
#         inputs: logits of shape [B, C]
#         targets: ground truth labels of shape [B]
#         """
#         if inputs.dim() > 2:
#             inputs = inputs.view(inputs.size(0), inputs.size(1))  # flatten if needed

#         # Convert logits to probabilities
#         log_probs = F.log_softmax(inputs, dim=1)
#         probs = torch.exp(log_probs)

#         targets = targets.view(-1, 1)
#         log_pt = log_probs.gather(1, targets).squeeze(1)   # log(p_t)
#         pt = probs.gather(1, targets).squeeze(1)           # p_t

#         # Alpha weighting (optional)
#         if self.alpha is not None:
#             if self.alpha.device != inputs.device:
#                 self.alpha = self.alpha.to(inputs.device)
#             at = self.alpha.gather(0, targets.squeeze())
#             log_pt = log_pt * at

#         loss = - (1 - pt) ** self.gamma * log_pt

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         return loss



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

