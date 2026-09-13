"""
PyTorch Dataset for bitemporal (Siamese) change detection.

Expects:
    data/processed/change/
        train/
            t1/tile_00001.npy   -> (C, H, W) image at time t1
            t2/tile_00001.npy   -> (C, H, W) image at time t2
            labels/tile_00001.npy -> (H, W) binary change mask {0,1}
        val/...
        test/...
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root) / split
        self.t1_dir = self.root / "t1"
        self.t2_dir = self.root / "t2"
        self.label_dir = self.root / "labels"
        if not self.t1_dir.exists():
            raise FileNotFoundError(
                f"{self.t1_dir} not found. Run scripts/preprocess_data.py --task change first."
            )

        self.tile_ids = sorted(p.stem for p in self.t1_dir.glob("*.npy"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        tile_id = self.tile_ids[idx]
        img1 = np.load(self.t1_dir / f"{tile_id}.npy")
        img2 = np.load(self.t2_dir / f"{tile_id}.npy")
        label = np.load(self.label_dir / f"{tile_id}.npy")

        if self.transform is not None:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            augmented = self.transform(image=img1, image2=img2, mask=label)
            img1 = np.transpose(augmented["image"], (2, 0, 1))
            img2 = np.transpose(augmented["image2"], (2, 0, 1))
            label = augmented["mask"]

        return (
            torch.from_numpy(img1.astype(np.float32)),
            torch.from_numpy(img2.astype(np.float32)),
            torch.from_numpy(label.astype(np.float32)),
            tile_id,
        )
