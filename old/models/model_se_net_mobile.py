import torch
import torch.nn as nn

INFO = "SE-Net Mobile: SE blocks + Separable Conv + GAP"

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class SeparableSEBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.se = SEBlock(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        x = self.se(self.act(self.bn(x)))
        return self.pool(x)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2)
        )
        self.features = nn.Sequential(
            SeparableSEBlock(64, 128, pool=True),
            SeparableSEBlock(128, 256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))