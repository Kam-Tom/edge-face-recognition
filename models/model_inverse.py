import torch
import torch.nn as nn

INFO = "Model Inverse: Inverse Pyramid [256, 128, 64]."

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        
        self.features = nn.Sequential(
            BasicBlock(3, 256),   # -> 56x56
            BasicBlock(256, 128), # -> 28x28
            BasicBlock(128, 64),  # -> 14x14
            nn.Flatten()
        )
        
        # 64 channels * 14 * 14 pixels
        flat_dim = 64 * 14 * 14
        
        self.fc = nn.Linear(flat_dim, embedding_size)
        self.bn_out = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.bn_out(x)
        return x