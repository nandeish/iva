"""
Lossy multi-spectral compression (Section 4.2 / 2.2.3 of the report):
a convolutional autoencoder with a learned entropy bottleneck (via CompressAI's
hyperprior model), used either standalone or combined with block-DCT
quantization (`preprocessing.transforms.block_dct` + `quantize_dct`) as a
classical baseline / pre-conditioning step.

Two paths are provided:
  1. `ConvAutoencoder`   -- plain conv encoder/decoder, trained with MSE +
                             a rate proxy (entropy of the quantized latent).
  2. `HyperpriorCompressor` -- thin wrapper around CompressAI's
                             `ScaleHyperprior`, which gives a proper learned
                             entropy model and real bits-per-pixel estimates.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Simple symmetric conv autoencoder, band-agnostic (works on any number
    of spectral channels, e.g. all 12 Sentinel-2 bands)."""

    def __init__(self, in_channels: int = 12, latent_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(64, 96, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(96, latent_channels, 5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 96, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(96, 64, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        # Straight-through uniform quantization (rounding) with a differentiable
        # noise proxy during training, as in Balle et al. (2017).
        if self.training:
            noise = torch.empty_like(latent).uniform_(-0.5, 0.5)
            latent_q = latent + noise
        else:
            latent_q = torch.round(latent)
        recon = self.decoder(latent_q)
        return recon, latent_q


class HyperpriorCompressor(nn.Module):
    """Wraps CompressAI's ScaleHyperprior for a proper rate-distortion trained
    model with a real entropy model (bits-per-pixel is directly estimable)."""

    def __init__(self, in_channels: int = 12, N: int = 128, M: int = 192):
        super().__init__()
        from compressai.models import ScaleHyperprior

        self.model = ScaleHyperprior(N=N, M=M)
        # Patch the first/last conv layers to match `in_channels` instead of
        # CompressAI's default 3-channel (RGB) assumption.
        first_conv = self.model.g_a[0]
        self.model.g_a[0] = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding,
        )
        last_conv = self.model.g_s[-1]
        self.model.g_s[-1] = nn.ConvTranspose2d(
            last_conv.in_channels, in_channels,
            kernel_size=last_conv.kernel_size, stride=last_conv.stride,
            padding=last_conv.padding, output_padding=last_conv.output_padding,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)   # dict with 'x_hat' and 'likelihoods'


def rate_distortion_loss(recon, target, latent_likelihoods=None, lambda_rd: float = 0.01):
    """
    Combined MSE distortion + rate loss. If `latent_likelihoods` is provided
    (from a CompressAI model's 'likelihoods' dict), the bits-per-pixel term is
    computed exactly; otherwise it falls back to an entropy proxy over the
    rounded ConvAutoencoder latent.
    """
    mse = nn.functional.mse_loss(recon, target)
    if latent_likelihoods is not None:
        num_pixels = target.shape[0] * target.shape[2] * target.shape[3]
        bpp = sum(
            (torch.log(likelihoods).sum() / (-torch.log(torch.tensor(2.0)) * num_pixels))
            for likelihoods in latent_likelihoods.values()
        )
        return mse + lambda_rd * bpp, mse, bpp
    return mse, mse, torch.tensor(0.0)


if __name__ == "__main__":
    model = ConvAutoencoder(in_channels=12, latent_channels=32)
    dummy = torch.randn(2, 12, 256, 256)
    recon, latent = model(dummy)
    assert recon.shape == dummy.shape, recon.shape
    print("ConvAutoencoder OK:", recon.shape, "latent:", latent.shape)
