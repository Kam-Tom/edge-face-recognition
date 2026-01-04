import torch
import torch.nn as nn

INFO = "ResNet Mobile: Skip connections + Separable Conv + GAP"

class SeparableResBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.dw1 = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dw2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.pw2 = nn.Conv2d(out_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        
        self.skip = nn.Identity()
        if downsample or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.pw1(self.dw1(x))))
        out = self.bn2(self.pw2(self.dw2(out)))
        return self.act(out + self.skip(x))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            SeparableResBlock(64, 128, downsample=True),
            SeparableResBlock(128, 256, downsample=True),
            SeparableResBlock(256, 256, downsample=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))