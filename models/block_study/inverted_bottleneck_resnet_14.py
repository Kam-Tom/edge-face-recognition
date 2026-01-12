import torch
import torch.nn as nn

INFO = "Inverted Bottleneck 14 layers [1x1 expand->3x3->1x1 squeeze]. Compare to bottleneck_resnet_14."

class InvertedBlock(nn.Module):
    """Inverted: expand first, then squeeze (opposite of bottleneck)"""
    def __init__(self, in_c, out_c, stride=1, expand=4):
        super().__init__()
        
        mid_c = in_c * expand
        
        # 1x1 expand
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act1 = nn.PReLU(mid_c)
        
        # 3x3 process (regular conv, not depthwise)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.act2 = nn.PReLU(mid_c)
        
        # 1x1 squeeze
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
            InvertedBlock(3, 64, stride=2, expand=1),  # First: can't expand 3 channels much
            InvertedBlock(64, 64),
            InvertedBlock(64, 128, stride=2),
            InvertedBlock(128, 128),
            InvertedBlock(128, 256, stride=2),
            InvertedBlock(256, 256),
            InvertedBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))