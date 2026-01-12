import torch
import torch.nn as nn

INFO = "GhostNet-style 20 layers. Ghost modules + depthwise. Efficient for Pi."

class GhostModule(nn.Module):
    """Ghost: 1x1 for half, depthwise 3x3 for other half"""
    def __init__(self, in_c, out_c):
        super().__init__()
        
        half_c = out_c // 2
        
        # Primary: 1x1 conv
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, half_c, 1, bias=False),
            nn.BatchNorm2d(half_c),
            nn.PReLU(half_c),
        )
        
        # Cheap: depthwise 3x3
        self.cheap = nn.Sequential(
            nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
            nn.BatchNorm2d(half_c),
            nn.PReLU(half_c),
        )

    def forward(self, x):
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)

class GhostBottleneck(nn.Module):
    """Ghost bottleneck with skip connection"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        mid_c = out_c // 2
        
        self.ghost1 = GhostModule(in_c, mid_c)
        
        # Downsample with depthwise if needed
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
                nn.BatchNorm2d(mid_c),
            )
        else:
            self.downsample = nn.Identity()
        
        self.ghost2 = GhostModule(mid_c, out_c)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.ghost1(x)
        out = self.downsample(out)
        out = self.ghost2(out)
        return out + self.shortcut(x)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        
        self.features = nn.Sequential(
            # Stage 1: 64 channels
            GhostBottleneck(64, 64),
            
            # Stage 2: 128 channels
            GhostBottleneck(64, 128, stride=2),
            GhostBottleneck(128, 128),
            GhostBottleneck(128, 128),
            
            # Stage 3: 256 channels
            GhostBottleneck(128, 256, stride=2),
            GhostBottleneck(256, 256),
            GhostBottleneck(256, 256),
            
            # Stage 4: 512 channels
            GhostBottleneck(256, 512, stride=2),
            GhostBottleneck(512, 512),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))