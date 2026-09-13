"""
DeepLabV3+ with Atrous Spatial Pyramid Pooling (Chen et al., 2018), wrapped via
`segmentation_models_pytorch` (SMP) for a pretrained ResNet-50 encoder.

SMP handles arbitrary `in_channels` by re-initialising the first conv layer,
so ImageNet weights still provide useful low-level features even though our
input is a 15-channel spectral+index stack rather than RGB.
"""
from __future__ import annotations

import torch.nn as nn


def build_deeplabv3plus(
    in_channels: int = 15,
    num_classes: int = 7,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,   # raw logits; loss functions apply softmax/sigmoid
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_deeplabv3plus(in_channels=15, num_classes=7, encoder_weights=None)
    dummy = torch.randn(2, 15, 256, 256)
    out = model(dummy)
    assert out.shape == (2, 7, 256, 256), out.shape
    print("DeepLabV3+ OK:", out.shape)
