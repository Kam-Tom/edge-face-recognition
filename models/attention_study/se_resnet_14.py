import torch
import torch.nn as nn

INFO = "SE-ResNet 14 layers. Squeeze-Excitation attention. Compare to skip_study/resnet_14."

class SEModule(nn.Module):
    """Squeeze-Excitation: channel attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pool
        w = x.mean(dim=(2, 3))
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Scale
        return x * w.view(b, c, 1, 1)

class SEResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.PReLU(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.se = SEModule(out_c)  # SE after conv2
        
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
        out = self.se(out)  # Apply SE before skip
        out += self.shortcut(x)
        return self.act2(out)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            SEResBlock(3, 64, stride=2),
            SEResBlock(64, 64),
            SEResBlock(64, 128, stride=2),
            SEResBlock(128, 128),
            SEResBlock(128, 256, stride=2),
            SEResBlock(256, 256),
            SEResBlock(256, 512, stride=2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.bn(self.fc(x))