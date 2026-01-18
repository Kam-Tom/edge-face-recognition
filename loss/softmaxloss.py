import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(SoftmaxLoss, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        # Skalowany Softmax dla lepszej zbieżności
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        return logits * 64.0

    def get_logits(self, features):
        return F.linear(F.normalize(features), F.normalize(self.weight)) * 6