import torch
import torch.nn as nn

INFO = "ResNet Deep: 12 conv layers with skip connections"


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

        self.skip = nn.Identity()
        if downsample or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2),                      # 112 -> 56
            
            # Stage 1: 56x56
            ResBlock(64, 64),
            ResBlock(64, 128, downsample=True),   # -> 28
            
            # Stage 2: 28x28
            ResBlock(128, 128),
            ResBlock(128, 256, downsample=True),  # -> 14
            
            # Stage 3: 14x14
            ResBlock(256, 256),
            ResBlock(256, 512, downsample=True),  # -> 7
            
            # Stage 4: 7x7
            ResBlock(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))