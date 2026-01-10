import torch
import torch.nn as nn

INFO = "Dropout GAP: Same as Baseline GAP + Dropout(0.5). Tests regularization on small embeddings."

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
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(p=0.5)
        
        # FC have only 256 channels (No GAP)
        self.fc = nn.Linear(256, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.bn(x)