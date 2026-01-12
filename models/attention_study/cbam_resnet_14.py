import torch
import torch.nn as nn

INFO = "CBAM-ResNet 14 layers. Channel + Spatial attention. Compare to skip_study/resnet_14."

class ChannelAttention(nn.Module):
    """Channel attention: what features are important"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Both avg and max pool
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        # Shared MLP
        avg_out = self.fc2(torch.relu(self.fc1(avg_pool)))
        max_out = self.fc2(torch.relu(self.fc1(max_pool)))
        # Combine
        w = torch.sigmoid(avg_out + max_out)
        return x * w.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """Spatial attention: where to focus"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Channel-wise avg and max
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        # Concat and conv
        combined = torch.cat([avg_out, max_out], dim=1)
        w = torch.sigmoid(self.conv(combined))
        return x * w

class CBAMResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.PReLU(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        
        self.act2 = nn.PReLU(out_c)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)  # Channel attention first
        out = self.sa(out)  # Then spatial attention
        out += self.shortcut(x)
        return self.act2(out)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            CBAMResBlock(3, 64, stride=2),
            CBAMResBlock(64, 64),
            CBAMResBlock(64, 128, stride=2),
            CBAMResBlock(128, 128),
            CBAMResBlock(128, 256, stride=2),
            CBAMResBlock(256, 256),
            CBAMResBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))