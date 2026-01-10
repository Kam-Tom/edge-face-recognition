import torch
import torch.nn as nn

INFO = "Dropout CNN: Same as Baseline Flatten (3 blocks) + Dropout(0.5)"

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
            Block(3, 64),    # -> 56x56
            Block(64, 128),  # -> 28x28
            Block(128, 256), # -> 14x14
            nn.Flatten()
        )
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(50176, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x) 
        x = self.fc(x)
        return self.bn(x)