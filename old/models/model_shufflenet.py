import torch
import torch.nn as nn

INFO = "ShuffleNet: Channel shuffle + grouped convolutions for efficiency"

def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=4):
        super().__init__()
        self.stride = stride
        mid_c = out_c // 4
        self.groups = groups
        
        self.compress = nn.Conv2d(in_c, mid_c, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.dw = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.expand = nn.Conv2d(mid_c, out_c, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        
        self.shortcut = nn.Sequential()
        if stride == 2 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.compress(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.dw(out))
        out = self.bn3(self.expand(out))
        return self.act(out + self.shortcut(x))

class Net(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(24)
        )
        self.features = nn.Sequential(
            ShuffleBlock(24, 64, stride=2),
            ShuffleBlock(64, 64),
            ShuffleBlock(64, 128, stride=2),
            ShuffleBlock(128, 128),
            ShuffleBlock(128, 256),
            ShuffleBlock(256, 256),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 14 * 14, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.stem(x)
        return self.bn(self.fc(self.features(x)))