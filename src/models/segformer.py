"""
SegFormer-B3 (Xie et al., 2021), wrapped via HuggingFace `transformers`.

The Mix Transformer (MiT-B3) hierarchical encoder is pretrained on ImageNet
for RGB input; since our pipeline feeds a 15-channel spectral+index stack, the
patch-embedding conv is re-initialised for the new channel count (pretrained
weights are kept for every other layer).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SegFormerLULC(nn.Module):
    def __init__(
        self,
        in_channels: int = 15,
        num_classes: int = 7,
        pretrained: str = "nvidia/mit-b3",
    ):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        config = SegformerConfig.from_pretrained(pretrained, num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, config=config, ignore_mismatched_sizes=True
        )

        # Re-initialise the first patch-embedding conv to accept `in_channels`
        # instead of the default 3 (RGB), copying pretrained RGB weights into
        # the first 3 channels and initialising the rest from a scaled mean.
        old_proj = self.model.segformer.stages[0].patch_embeddings.proj
        if in_channels != old_proj.in_channels:
            new_proj = nn.Conv2d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
            )
            with torch.no_grad():
                mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
                new_proj.weight[:, :3] = old_proj.weight
                new_proj.weight[:, 3:] = mean_weight.repeat(1, in_channels - 3, 1, 1)
                new_proj.bias = old_proj.bias
            self.model.segformer.stages[0].patch_embeddings.proj = new_proj

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.model(pixel_values=x).logits          # (B, num_classes, H/4, W/4)
        return nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = SegFormerLULC(in_channels=15, num_classes=7)
    dummy = torch.randn(1, 15, 256, 256)
    out = model(dummy)
    assert out.shape == (1, 7, 256, 256), out.shape
    print("SegFormer OK:", out.shape)
