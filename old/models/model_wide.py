import torch
import torch.nn as nn

INFO = "Wide CNN: 3 blocks [128, 256, 512] - tests width vs depth"

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            Block(3, 128),
            Block(128, 256),
            Block(256, 512),
            nn.Flatten()
        )
        self.fc = nn.Linear(512 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.fc(self.features(x)))