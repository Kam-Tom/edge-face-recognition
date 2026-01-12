import torch
import torch.nn as nn

INFO = "Plain CNN 8 layers with MaxPool for downsampling. Compare to depth_study/plain_8 (stride)."

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # NO stride - always stride=1
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stage 1: 64 channels
            nn.MaxPool2d(2),            # -> 56x56 (maxpool downsample)
            Block(3, 64),
            Block(64, 64),
            
            # Stage 2: 128 channels
            nn.MaxPool2d(2),            # -> 28x28
            Block(64, 128),
            Block(128, 128),
            
            # Stage 3: 256 channels
            nn.MaxPool2d(2),            # -> 14x14
            Block(128, 256),
            Block(256, 256),
            Block(256, 256),
            Block(256, 256),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))