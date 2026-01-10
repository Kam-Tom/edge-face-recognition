import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        # Normalize both features and weights to unit sphere
        features = F.normalize(features)
        weights = F.normalize(self.weight)
        
        # Cosine similarity via dot product
        cos_theta = F.linear(features, weights).clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Add angular margin to target class
        theta = torch.acos(cos_theta)
        theta.scatter_add_(1, targets.unsqueeze(1), torch.full_like(targets.unsqueeze(1), self.m, dtype=theta.dtype))
        
        return torch.cos(theta) * self.s
    
    def get_logits(self, features):
        return F.linear(F.normalize(features), F.normalize(self.weight)) * self.s