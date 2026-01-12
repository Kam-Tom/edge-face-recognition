import torch
import torch.nn as nn

INFO = "Depthwise Separable ResNet 14. Test if depthwise is efficient. Compare to skip_study/resnet_14."

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable: depthwise 3x3 + pointwise 1x1"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        # Depthwise: 3x3 conv per channel separately
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.act1 = nn.PReLU(in_c)
        
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.PReLU(out_c)

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x

class DepthwiseResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_c, out_c, stride)
        self.conv2 = DepthwiseSeparableConv(out_c, out_c, 1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        # First conv is regular (3 input channels - can't do depthwise well)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        
        self.features = nn.Sequential(
            DepthwiseResBlock(64, 64),
            DepthwiseResBlock(64, 128, stride=2),
            DepthwiseResBlock(128, 128),
            DepthwiseResBlock(128, 256, stride=2),
            DepthwiseResBlock(256, 256),
            DepthwiseResBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))