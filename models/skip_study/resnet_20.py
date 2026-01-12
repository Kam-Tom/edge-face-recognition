import torch
import torch.nn as nn

INFO = "ResNet: 20 layers with Skip Connections. Compare to plain_20."

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.PReLU(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        
        self.act2 = nn.PReLU(out_c)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act2(out)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        # 10 ResBlocks x 2 layers = 20 layers
        self.features = nn.Sequential(
            # Stage 1: 64 channels (2 blocks = 4 layers)
            ResBlock(3, 64, stride=2),    # Layers 1-2 -> 56x56
            ResBlock(64, 64),             # Layers 3-4
            
            # Stage 2: 128 channels (3 blocks = 6 layers)
            ResBlock(64, 128, stride=2),  # Layers 5-6 -> 28x28
            ResBlock(128, 128),           # Layers 7-8
            ResBlock(128, 128),           # Layers 9-10
            
            # Stage 3: 256 channels (3 blocks = 6 layers)
            ResBlock(128, 256, stride=2), # Layers 11-12 -> 14x14
            ResBlock(256, 256),           # Layers 13-14
            ResBlock(256, 256),           # Layers 15-16
            
            # Stage 4: 512 channels (2 blocks = 4 layers)
            ResBlock(256, 512, stride=2), # Layers 17-18 -> 7x7
            ResBlock(512, 512),           # Layers 19-20
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))