import torch
import torch.nn as nn

INFO = "MobileNetV2-style 20 layers. Inverted bottleneck + depthwise. Efficient for Pi."

class InvertedResidual(nn.Module):
    """MobileNetV2 block: expand -> depthwise -> squeeze"""
    def __init__(self, in_c, out_c, stride=1, expand=4):
        super().__init__()
        
        mid_c = in_c * expand
        self.use_skip = (stride == 1 and in_c == out_c)
        
        self.block = nn.Sequential(
            # 1x1 expand
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            
            # 3x3 depthwise
            nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            
            # 1x1 squeeze (no activation - linear)
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_skip:
            return x + self.block(x)
        return self.block(x)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        # Stem: regular conv (can't depthwise 3 channels well)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        
        self.features = nn.Sequential(
            # Stage 1: 64 channels
            InvertedResidual(64, 64, expand=2),
            
            # Stage 2: 128 channels
            InvertedResidual(64, 128, stride=2),
            InvertedResidual(128, 128),
            InvertedResidual(128, 128),
            
            # Stage 3: 256 channels
            InvertedResidual(128, 256, stride=2),
            InvertedResidual(256, 256),
            InvertedResidual(256, 256),
            
            # Stage 4: 512 channels
            InvertedResidual(256, 512, stride=2),
            InvertedResidual(512, 512),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))