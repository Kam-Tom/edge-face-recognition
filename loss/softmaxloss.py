import torch.nn as nn

class SoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, out_classes)
        
    def forward(self, features, labels):
        logits = self.fc(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    
    def get_logits(self, features):
        return self.fc(features)