import torch
import torch.nn as nn

INFO = "Bottleneck 14 layers WITHOUT skip. Prove skips matter for bottleneck too."

class BottleneckBlock(nn.Module):
    """Bottleneck without skip connection"""
    def __init__(self, in_c, out_c, stride=1, reduction=4):
        super().__init__()
        
        mid_c = max(out_c // reduction, 16)
        
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act1 = nn.PReLU(mid_c)
        
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.act2 = nn.PReLU(mid_c)
        
        self.conv3 = nn.Conv2d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act3 = nn.PReLU(out_c)
        
        # NO SHORTCUT

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        # NO SKIP
        return out

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            BottleneckBlock(3, 64, stride=2),
            BottleneckBlock(64, 64),
            BottleneckBlock(64, 128, stride=2),
            BottleneckBlock(128, 128),
            BottleneckBlock(128, 256, stride=2),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))