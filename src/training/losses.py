"""
Loss functions for multi-class segmentation and binary change detection.

`ComboLoss` = Dice + Focal, which handles the class imbalance typical of
land-cover datasets (e.g. water/wetland classes being rare relative to
cropland/forest) better than plain cross-entropy alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(logits, target)


class BinaryDiceBCELoss(nn.Module):
    """Used for the Siamese change-detection head (single-channel logits)."""

    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)
