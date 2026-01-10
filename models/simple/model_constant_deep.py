import torch
import torch.nn as nn

INFO = "Constant Deep CNN: 6 blocks, Width fixed at 64 channels. Tests information bottleneck when spatial size drops."

class Block(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            Block(3, 64),
            Block(64, 64, pool=True),    # -> 56x56
            
            Block(64, 64),
            Block(64, 64, pool=True),    # -> 28x28
            
            Block(64, 64),
            Block(64, 64, pool=True),    # -> 14x14
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(64, embedding_size) 
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))