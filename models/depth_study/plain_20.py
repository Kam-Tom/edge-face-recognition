import torch
import torch.nn as nn

INFO = "Plain CNN: 20 layers [64->128->256->512]. Very deep - gradient problems expected."

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
            # Stage 1: 64 channels (5 layers)
            Block(3, 64, stride=2),     # Layer 1 -> 56x56
            Block(64, 64),              # Layer 2
            Block(64, 64),              # Layer 3
            Block(64, 64),              # Layer 4
            Block(64, 64),              # Layer 5
            
            # Stage 2: 128 channels (5 layers)
            Block(64, 128, stride=2),   # Layer 6 -> 28x28
            Block(128, 128),            # Layer 7
            Block(128, 128),            # Layer 8
            Block(128, 128),            # Layer 9
            Block(128, 128),            # Layer 10
            
            # Stage 3: 256 channels (5 layers)
            Block(128, 256, stride=2),  # Layer 11 -> 14x14
            Block(256, 256),            # Layer 12
            Block(256, 256),            # Layer 13
            Block(256, 256),            # Layer 14
            Block(256, 256),            # Layer 15
            
            # Stage 4: 512 channels (5 layers)
            Block(256, 512, stride=2),  # Layer 16 -> 7x7
            Block(512, 512),            # Layer 17
            Block(512, 512),            # Layer 18
            Block(512, 512),            # Layer 19
            Block(512, 512),            # Layer 20
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))