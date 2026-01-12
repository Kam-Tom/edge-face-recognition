import torch
import torch.nn as nn

INFO = "Plain CNN 8 layers with Flatten (no GAP). Compare to depth_study/plain_8."

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
        
        # Same architecture as depth_study/plain_8
        self.features = nn.Sequential(
            Block(3, 64, stride=2),     # Layer 1 -> 56x56
            Block(64, 64),              # Layer 2
            Block(64, 128, stride=2),   # Layer 3 -> 28x28
            Block(128, 128),            # Layer 4
            Block(128, 256, stride=2),  # Layer 5 -> 14x14
            Block(256, 256),            # Layer 6
            Block(256, 256),            # Layer 7
            Block(256, 256),            # Layer 8
        )
        
        # DIFFERENCE: Flatten instead of GAP
        # 14x14x256 = 50176 inputs
        self.fc = nn.Linear(256 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)  # [B, 50176] instead of [B, 256]
        return self.bn(self.fc(x))