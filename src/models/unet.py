"""
Standard encoder-decoder U-Net with skip connections (Ronneberger et al., 2015),
adapted for arbitrary input channel counts (multi-spectral + index stacks) and
an arbitrary number of output classes.

This is a from-scratch implementation (no external backbone) so it works with
any `in_channels` without needing ImageNet-pretrained weights. For a
pretrained-encoder variant, see `models/deeplabv3plus.py`, which wraps
`segmentation_models_pytorch`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad in case of odd input sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Parameters
    ----------
    in_channels : total input channels (spectral bands + appended indices)
    num_classes : number of land-cover categories
    base_channels : width of the first encoder stage (doubled at each downsample)
    """

    def __init__(self, in_channels: int = 15, num_classes: int = 7, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    # quick shape sanity check (no training)
    model = UNet(in_channels=15, num_classes=7)
    dummy = torch.randn(2, 15, 256, 256)
    out = model(dummy)
    assert out.shape == (2, 7, 256, 256), out.shape
    print("UNet OK:", out.shape)
