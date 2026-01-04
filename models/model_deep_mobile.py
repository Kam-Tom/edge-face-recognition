import torch
import torch.nn as nn

INFO = "Deep CNN Mobile: 6 blocks with Separable Conv + GAP"

class SeparableBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.pointwise(self.depthwise(x)))))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.features = nn.Sequential(
            SeparableBlock(64, 64, pool=True),
            SeparableBlock(64, 128),
            SeparableBlock(128, 128, pool=True),
            SeparableBlock(128, 256),
            SeparableBlock(256, 256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))