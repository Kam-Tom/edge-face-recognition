import torch
import torch.nn as nn

INFO = "Inverse Pyramid: Channels decrease [256->128->64]. Flatten used to save spatial info."

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
            Block(3, 256),   # -> 56x56
            Block(256, 128), # -> 28x28
            Block(128, 64),  # -> 14x14
            nn.Flatten()
        )
        
        self.fc = nn.Linear(12544, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return self.bn(x)