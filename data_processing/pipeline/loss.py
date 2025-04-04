


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class ClassBalancedFocalLoss(nn.Module):
#     def __init__(self, samples_per_class, beta=0.999, gamma=1.0, reduction='mean', boost_factors=None):
#         super(ClassBalancedFocalLoss, self).__init__()
#         effective_num = 1.0 - np.power(beta, samples_per_class)
#         weights = (1.0 - beta) / np.array(effective_num)
#         weights = weights / np.sum(weights) * len(samples_per_class)
        
#         if boost_factors:
#             for cls_idx, factor in boost_factors.items():
#                 if 0 <= cls_idx < len(weights):
#                     weights[cls_idx] *= factor
#             # Re-normalize after boosting
#             weights = weights / np.sum(weights) * len(weights)

#         self.alpha = torch.tensor(weights, dtype=torch.float32)
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         if inputs.dim() > 2:
#             inputs = inputs.view(inputs.size(0), inputs.size(1))
#         log_probs = F.log_softmax(inputs, dim=1)
#         probs = torch.exp(log_probs)

#         targets = targets.view(-1, 1)
#         log_pt = log_probs.gather(1, targets).squeeze(1)
#         pt = probs.gather(1, targets).squeeze(1)

#         alpha_t = self.alpha.to(inputs.device).gather(0, targets.squeeze())
#         loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         return loss



# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np


# # class ClassBalancedFocalLoss(nn.Module):
# #     def __init__(self, samples_per_class, beta=0.999, gamma=1.0, reduction='mean', boost_factors=None):
# #         """
# #         samples_per_class: dict (Counter) or list containing class sample counts, including class 0
# #         beta: parameter for effective number computation (close to 1.0 = more smoothing)
# #         gamma: focusing parameter for focal loss
# #         boost_factors: dict {class_idx: boost_multiplier} to manually adjust per-class weights
# #         """
# #         super(ClassBalancedFocalLoss, self).__init__()

# #         # Convert input to numpy array (handle Counter or list)
# #         # if isinstance(samples_per_class, dict):
# #         #     max_class = max(samples_per_class.keys())
# #         #     samples_array = np.array([samples_per_class.get(i, 1) for i in range(max_class + 1)])
# #         # else:
# #         #     samples_array = np.array(samples_per_class)

# #         # Compute class-balanced weights
# #         effective_num = 1.0 - np.power(beta, samples_per_class)
# #         weights = (1.0 - beta) / effective_num
# #         weights = weights / np.sum(weights) * len(samples_per_class)  # normalize to num classes

# #         # Apply per-class boost factors if given
# #         if boost_factors:
# #             for cls_idx, factor in boost_factors.items():
# #                 if 0 <= cls_idx < len(weights):
# #                     weights[cls_idx] *= factor
# #             # Re-normalize after boosting
# #             weights = weights / np.sum(weights) * len(weights)

# #         self.alpha = torch.tensor(weights, dtype=torch.float32)
# #         self.gamma = gamma
# #         self.reduction = reduction

# #     def forward(self, inputs, targets):
# #         """
# #         inputs: logits of shape [B, C]
# #         targets: ground truth labels of shape [B]
# #         """
# #         if inputs.dim() > 2:
# #             inputs = inputs.view(inputs.size(0), inputs.size(1))

# #         log_probs = F.log_softmax(inputs, dim=1)
# #         probs = torch.exp(log_probs)

# #         targets = targets.view(-1, 1)
# #         log_pt = log_probs.gather(1, targets).squeeze(1)  # log(p_t)
# #         pt = probs.gather(1, targets).squeeze(1)          # p_t

# #         alpha_t = self.alpha.to(inputs.device).gather(0, targets.squeeze())
# #         loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt

# #         if self.reduction == 'mean':
# #             return loss.mean()
# #         elif self.reduction == 'sum':
# #             return loss.sum()
# #         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten if needed

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
