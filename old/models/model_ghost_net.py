import torch
import torch.nn as nn

INFO = "GhostNet: Generate features cheaply with linear ops"

class GhostModule(nn.Module):
    def __init__(self, in_c, out_c, ratio=2):
        super().__init__()
        init_c = out_c // ratio
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_c, 1, bias=False),
            nn.BatchNorm2d(init_c),
            nn.PReLU(init_c)
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_c, init_c, 3, padding=1, groups=init_c, bias=False),
            nn.BatchNorm2d(init_c),
            nn.PReLU(init_c)
        )

    def forward(self, x):
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)

class GhostBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.ghost = GhostModule(in_c, out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.ghost(x))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            GhostBlock(3, 64, pool=True),
            GhostBlock(64, 128, pool=True),
            GhostBlock(128, 256, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))