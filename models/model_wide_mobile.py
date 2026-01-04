import torch
import torch.nn as nn

INFO = "Wide CNN Mobile: Wide channels [128, 256, 512] + Separable + GAP"

class SeparableBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.pw(self.dw(x)))))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.MaxPool2d(2)
        )
        self.features = nn.Sequential(
            SeparableBlock(128, 256),
            SeparableBlock(256, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))