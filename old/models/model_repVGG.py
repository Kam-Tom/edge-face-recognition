import torch
import torch.nn as nn

INFO = "RepVGG: Multi-branch training, single-branch inference (reparameterizable)"

class RepVGGBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        
        self.conv3x3 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_c)
        self.conv1x1 = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_c)
        self.bn_identity = nn.BatchNorm2d(out_c) if in_c == out_c and stride == 1 else None
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.bn_identity is not None:
            out = out + self.bn_identity(x)
        return self.act(out)

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            RepVGGBlock(3, 64, stride=2),
            RepVGGBlock(64, 64),
            RepVGGBlock(64, 128, stride=2),
            RepVGGBlock(128, 128),
            RepVGGBlock(128, 256, stride=2),
            RepVGGBlock(256, 256),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.fc(self.features(x)))