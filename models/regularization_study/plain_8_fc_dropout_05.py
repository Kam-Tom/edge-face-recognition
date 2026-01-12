import torch
import torch.nn as nn

INFO = "Plain CNN 8 layers + Dropout(0.5) before FC only. Compare to depth_study/plain_8."

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            Block(3, 64, stride=2),
            Block(64, 64),
            Block(64, 128, stride=2),
            Block(128, 128),
            Block(128, 256, stride=2),
            Block(256, 256),
            Block(256, 256),
            Block(256, 256),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.bn(self.fc(x))