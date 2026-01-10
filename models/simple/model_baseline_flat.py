import torch
import torch.nn as nn

INFO = "Baseline CNN (No GAP): 3 stages [64->128->256]. Flatten -> Large FC (50k inputs)"

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
            Block(3, 64),
            Block(64, 128),
            Block(128, 256),
            nn.Flatten() # 14x14x256 -> 50176
        )
        # 256 channels * 14 * 14 spatial size = 50,176 input features
        self.fc = nn.Linear(50176, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.fc(self.features(x)))