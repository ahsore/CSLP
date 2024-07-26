import torch.nn as nn
import torch.nn.functional as func
import torch
from tools.attention import CBAM

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.max_split_size_mb=64

class ResBlock(nn.Module):

    def __init__(self, n_feats, kernel_size, bias=True, pad='same', pad_mode='reflect', bn=False, act=nn.GELU()):

        super(ResBlock, self).__init__()
        m = []

        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=pad, padding_mode=pad_mode))

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class CSLP(nn.Module):
    def __init__(self, n_channels, n_features=64, kernel_size=3, pad='same', pad_mode='reflect', bias_flag=True):
        super(CSLP, self).__init__()


        sensing_rate=0.5
        self.measurement = int(sensing_rate * 1024)
        self.base = 64
        self.sample = nn.Conv2d(n_channels, self.measurement, kernel_size=16, padding=0, stride=16, bias=False)
        self.initial = nn.Conv2d(self.measurement,256, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=False)
        self.res1 = ResBlock(self.base, 3)
        self.res2 = ResBlock(self.base, 3)
        self.res3 = ResBlock(self.base, 3)
        self.conv2 = nn.Conv2d(self.base, n_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv_1 = nn.Conv2d(n_channels, n_features, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)
        self.conv_2 = nn.Conv2d(n_features, n_features, kernel_size, bias=bias_flag, padding=pad,
                                padding_mode=pad_mode)
        self.CBAM_1 = CBAM(n_features, reduction_ratio=4, spatial=True)
        self.res_block_1 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.res_block_2 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.CBAM_2 = CBAM(n_features, reduction_ratio=4, spatial=True)
        self.conv_3 = nn.Conv2d(n_features, n_channels-1, 5, bias=bias_flag, padding=pad, padding_mode=pad_mode)

    def forward(self, inp):

     y = self.sample(inp)
     y = self.initial(y)
     c = nn.PixelShuffle(16)(y)
     out = self.relu(self.conv1(c))
     out = self.res1(out)
     out = self.res2(out)
     out = self.res3(out)
     out = self.conv2(out)
     SUMS= out + inp

     x = func.relu(self.conv_1(SUMS))
     x = func.relu(self.conv_2(x))
     x = self.CBAM_1(x) + x
     x = self.res_block_1(x)
     x = self.res_block_2(x)
     x = self.CBAM_2(x) + x
     x = self.conv_3(x)

     x = x +inp[:,:-1,:,:]

     return x

