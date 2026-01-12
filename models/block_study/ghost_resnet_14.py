import torch
import torch.nn as nn

INFO = "Ghost ResNet 14 layers. Half conv + cheap transform. Compare to skip_study/resnet_14."

class GhostModule(nn.Module):
    """Ghost: half channels from conv, half from cheap 3x3 transform"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        half_c = out_c // 2
        
        # Primary: regular conv for half channels
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, half_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(half_c),
            nn.PReLU(half_c),
        )
        
        # Cheap: 3x3 conv on primary output (simple, no groups)
        self.cheap = nn.Sequential(
            nn.Conv2d(half_c, half_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(half_c),
            nn.PReLU(half_c),
        )

    def forward(self, x):
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)

class GhostResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.ghost1 = GhostModule(in_c, out_c, stride=stride)
        self.ghost2 = GhostModule(out_c, out_c, stride=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.ghost1(x)
        out = self.ghost2(out)
        out += self.shortcut(x)
        return out

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            GhostResBlock(3, 64, stride=2),
            GhostResBlock(64, 64),
            GhostResBlock(64, 128, stride=2),
            GhostResBlock(128, 128),
            GhostResBlock(128, 256, stride=2),
            GhostResBlock(256, 256),
            GhostResBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))