"""
Siamese U-Net for bitemporal change detection (Section 4.4 / Paper 3 in the
literature review). Two images (t1, t2) are passed through a weight-shared
encoder; encoder feature maps at every scale are absolutely-differenced and
fed to a shared decoder that predicts a single-channel binary change map.

Optionally, a DWT-based change-intensity map (see
`preprocessing.transforms.dwt_change_score`) can be concatenated as an extra
input channel to each image before encoding, per `config.change_detection.use_dwt_features`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import DoubleConv, Down


class SiameseEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class DiffUp(nn.Module):
    """Decoder upsample block operating on absolute feature differences."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip_diff):
        x = self.up(x)
        diff_y = skip_diff.size(2) - x.size(2)
        diff_x = skip_diff.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip_diff, x], dim=1)
        return self.conv(x)


class SiameseUNetChangeDetector(nn.Module):
    def __init__(self, in_channels: int = 15, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.encoder = SiameseEncoder(in_channels, c)

        self.up1 = DiffUp(c * 16, c * 8, c * 8)
        self.up2 = DiffUp(c * 8, c * 4, c * 4)
        self.up3 = DiffUp(c * 4, c * 2, c * 2)
        self.up4 = DiffUp(c * 2, c, c)

        self.outc = nn.Conv2d(c, 1, kernel_size=1)   # binary change map logit

    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> torch.Tensor:
        feats1 = self.encoder(img_t1)
        feats2 = self.encoder(img_t2)
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]
        d1, d2, d3, d4, d5 = diffs

        x = self.up1(d5, d4)
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)
        return self.outc(x)   # (B, 1, H, W) logits -> sigmoid for change probability


if __name__ == "__main__":
    model = SiameseUNetChangeDetector(in_channels=15)
    t1 = torch.randn(2, 15, 256, 256)
    t2 = torch.randn(2, 15, 256, 256)
    out = model(t1, t2)
    assert out.shape == (2, 1, 256, 256), out.shape
    print("Siamese U-Net OK:", out.shape)
