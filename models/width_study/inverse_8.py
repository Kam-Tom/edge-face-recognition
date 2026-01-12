import torch
import torch.nn as nn

INFO = "Inverse CNN: 8 layers [256->128->64]. Channels decrease (opposite of normal)."

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
            # Stage 1: 256 channels (start wide)
            Block(3, 256, stride=2),    # Layer 1 -> 56x56
            Block(256, 256),            # Layer 2
            
            # Stage 2: 128 channels (decrease)
            Block(256, 128, stride=2),  # Layer 3 -> 28x28
            Block(128, 128),            # Layer 4
            
            # Stage 3: 64 channels (decrease more)
            Block(128, 64, stride=2),   # Layer 5 -> 14x14
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