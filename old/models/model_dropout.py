import torch
import torch.nn as nn

INFO = "Model Dropout: Deep 5 Layers with Dropout(0.5)."

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
            BasicBlock(3, 32),    # -> 56x56
            BasicBlock(32, 64),   # -> 28x28
            BasicBlock(64, 128),  # -> 14x14
            BasicBlock(128, 256), # -> 7x7
            BasicBlock(256, 512), # -> 3x3
            nn.Flatten()
        )
        
        self.dropout = nn.Dropout(p=0.5)
        
        # 512 channels * 3 * 3 pixels
        flat_dim = 512 * 3 * 3
        
        self.fc = nn.Linear(flat_dim, embedding_size)
        self.bn_out = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x) # Apply dropout here
        x = self.fc(x)
        x = self.bn_out(x)
        return x