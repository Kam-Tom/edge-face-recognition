import torch
import torch.nn as nn

INFO = "Constant CNN: 8 layers, all 64 channels. Tests information bottleneck."

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            # All stages: 64 channels (no increase)
            Block(3, 64, stride=2),     # Layer 1 -> 56x56
            Block(64, 64),              # Layer 2
            
            Block(64, 64, stride=2),    # Layer 3 -> 28x28
            Block(64, 64),              # Layer 4
            
            Block(64, 64, stride=2),    # Layer 5 -> 14x14
            Block(64, 64),              # Layer 6
            Block(64, 64),              # Layer 7
            Block(64, 64),              # Layer 8
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))