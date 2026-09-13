"""
PyTorch Dataset for pixel-level land-cover segmentation.

Expects pre-tiled `.npy` files on disk (produced by `scripts/preprocess_data.py`):

    data/processed/segmentation/
        train/
            images/tile_00001.npy   -> (C, H, W) float32, spectral bands + indices
            masks/tile_00001.npy    -> (H, W) int64 class ids
        val/...
        test/...
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LULCSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root) / split
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"{self.image_dir} not found. Run scripts/preprocess_data.py first."
            )

        self.tile_ids = sorted(p.stem for p in self.image_dir.glob("*.npy"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        tile_id = self.tile_ids[idx]
        image = np.load(self.image_dir / f"{tile_id}.npy")   # (C, H, W)
        mask = np.load(self.mask_dir / f"{tile_id}.npy")     # (H, W)

        if self.transform is not None:
            # albumentations-style transform expects HWC images
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image, mask=mask)
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"]

        image_t = torch.from_numpy(image.astype(np.float32))
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return image_t, mask_t, tile_id
