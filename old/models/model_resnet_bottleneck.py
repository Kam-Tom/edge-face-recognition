import torch
import torch.nn as nn

INFO = "Bottleneck: 1x1->3x3->1x1 squeeze-expand pattern (like ResNet-50)"

class Bottleneck(nn.Module):
    def __init__(self, in_c, mid_c, out_c, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        
        self.skip = nn.Identity()
        if downsample or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + self.skip(x))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            Bottleneck(64, 32, 128, downsample=True),
            Bottleneck(128, 64, 256, downsample=True),
            Bottleneck(256, 128, 256, downsample=True),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.fc(self.features(x)))