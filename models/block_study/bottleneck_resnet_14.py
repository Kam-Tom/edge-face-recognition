import torch
import torch.nn as nn

INFO = "Bottleneck ResNet 14 layers [1x1->3x3->1x1]. Compare to skip_study/resnet_14."

class BottleneckBlock(nn.Module):
    """Bottleneck: squeeze channels, process, expand back"""
    def __init__(self, in_c, out_c, stride=1, reduction=4):
        super().__init__()
        
        mid_c = max(out_c // reduction, 16)
        
        # 1x1 squeeze
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act1 = nn.PReLU(mid_c)
        
        # 3x3 process
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.act2 = nn.PReLU(mid_c)
        
        # 1x1 expand
        self.conv3 = nn.Conv2d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        
        self.act3 = nn.PReLU(out_c)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.act3(out)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            BottleneckBlock(3, 64, stride=2),
            BottleneckBlock(64, 64),
            BottleneckBlock(64, 128, stride=2),
            BottleneckBlock(128, 128),
            BottleneckBlock(128, 256, stride=2),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))