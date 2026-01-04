import torch
import torch.nn as nn

INFO = "ResNet-IR-SE: Improved Residual + SE blocks"

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
        return x * self.fc(w).view(b, c, 1, 1)

class IRBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        
        self.skip = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.act(self.bn2(self.conv1(out)))
        out = self.se(self.bn3(self.conv2(out)))
        return out + self.skip(x)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.features = nn.Sequential(
            IRBlock(64, 64, stride=2),
            IRBlock(64, 64),
            IRBlock(64, 128, stride=2),
            IRBlock(128, 128),
            IRBlock(128, 256, stride=2),
            IRBlock(256, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))