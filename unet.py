import torch
from torch import nn
import torch.nn.functional as F
from Conv2d_padding_torch import Conv2d as Conv2dWithPadding


class InBlock(nn.Module):
    def __init__(self, in_channels, block_channels, out_channels):
        super(InBlock, self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, padding=1)
        # self.batch_norm1 = nn.BatchNorm2d(block_channels)
        self.conv2 = nn.Conv2d(block_channels, out_channels, 3, padding=1)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.batch_norm1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        # out = self.batch_norm2(out)
        out = self.act_func(out)
        return out


class Downblock(nn.Module):
    def __init__(self, in_channels, block_channels, out_channels, pool_kernel_size=2, drop_=False, drop_p=0.5):
        super(Downblock, self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.drop_ = drop_
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, padding=1)
        # self.batch_norm1 = nn.BatchNorm2d(block_channels)
        self.conv2 = nn.Conv2d(block_channels, out_channels, 3, padding=1)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(drop_p)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        # out = self.batch_norm1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        # out = self.batch_norm2(out)
        out = self.act_func(out)
        if self.drop_:
            out = self.dropout(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, block_channels, out_channels):
        super(UpBlock, self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = Conv2dWithPadding(
            in_channels, block_channels, 2, padding='SAME')
        # self.conv1 = nn.Conv2d(in_channels,block_channels,3,padding=1)
        self.conv2 = nn.Conv2d(in_channels, block_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(block_channels, out_channels, 3, padding=1)

    def forward(self, x1, x2):
        out = self.upsample(x1)
        out = self.conv1(out)
        out = self.act_func(out)
        out = torch.cat([out, x2], dim=1)
        out = self.conv2(out)
        out = self.act_func(out)
        out = self.conv3(out)
        out = self.act_func(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_channels, block_channels, out_channels):
        super(OutBlock, self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(block_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act_func(out)
        out = self.conv2(out)
        return out


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.down1 = InBlock(n_channels, 64, 64)
        self.down2 = Downblock(64, 128, 128, 2, drop_=False)
        self.down3 = Downblock(128, 256, 256, 2, drop_=False)
        self.down4 = Downblock(256, 512, 512, 2, drop_=True)
        self.down5 = Downblock(512, 1024, 1024, 2, drop_=True)

        self.up6 = UpBlock(1024, 512, 512)
        self.up7 = UpBlock(512, 256, 256)
        self.up8 = UpBlock(256, 128, 128)
        self.up9 = UpBlock(128, 64, 64)

        self.out10 = OutBlock(64, 2, n_classes)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.up6(x5, x4)
        x7 = self.up7(x6, x3)
        x8 = self.up8(x7, x2)
        x9 = self.up9(x8, x1)
        out = self.out10(x9)
        return out
