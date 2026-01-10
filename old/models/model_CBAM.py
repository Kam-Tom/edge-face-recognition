import torch
import torch.nn as nn

INFO = "CBAM: Channel + Spatial attention (more than just SE)"

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x)
        x = self.sa(x)
        return self.pool(x)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            CBAMBlock(3, 64, pool=True),
            CBAMBlock(64, 128, pool=True),
            CBAMBlock(128, 256, pool=True),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.fc(self.features(x)))