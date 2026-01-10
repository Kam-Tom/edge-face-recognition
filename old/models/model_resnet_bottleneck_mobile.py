import torch
import torch.nn as nn

INFO = "Bottleneck Mobile: 1x1->3x3dw->1x1 (MobileNetV2 style) + GAP"

class InvertedBottleneck(nn.Module):
    def __init__(self, in_c, out_c, expand=4, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        mid_c = in_c * expand
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.dw = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(mid_c)
        
        self.use_skip = (stride == 1 and in_c == out_c)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.bn3(self.conv3(out))
        return out + x if self.use_skip else out

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            InvertedBottleneck(32, 64, downsample=True),
            InvertedBottleneck(64, 64),
            InvertedBottleneck(64, 128, downsample=True),
            InvertedBottleneck(128, 128),
            InvertedBottleneck(128, 256, downsample=True),
            InvertedBottleneck(256, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))