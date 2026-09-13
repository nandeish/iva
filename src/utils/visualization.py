"""
Small plotting helpers for sanity-checking pipeline outputs: RGB composites,
categorical LULC maps with a fixed class colour palette, binary change masks,
and NDVI-style anomaly Z-score maps.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Fixed palette matching `configs/config.yaml -> data.class_names`
LULC_COLORS = [
    "#1a9850",  # forest
    "#fee08b",  # cropland
    "#4575b4",  # water
    "#d73027",  # urban
    "#66c2a5",  # wetland
    "#bdbdbd",  # barren
    "#a6d96a",  # grassland
]


def plot_rgb_composite(stack_hwc: np.ndarray, band_indices=(2, 1, 0), ax=None, title: str = "RGB composite"):
    """`stack_hwc` is (H, W, C) reflectance in [0,1]; `band_indices` picks
    (R, G, B) channels, defaulting to Sentinel-2 (B04, B03, B02) order."""
    ax = ax or plt.gca()
    rgb = np.clip(stack_hwc[:, :, band_indices] * 2.5, 0, 1)   # simple brightness stretch
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_lulc_map(class_map: np.ndarray, class_names: Sequence[str], ax=None, title: str = "LULC classification"):
    ax = ax or plt.gca()
    cmap = ListedColormap(LULC_COLORS[: len(class_names)])
    im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=len(class_names) - 1)
    ax.set_title(title)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(class_names)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(class_names)
    return ax


def plot_change_map(change_mask: np.ndarray, ax=None, title: str = "Change detection"):
    ax = ax or plt.gca()
    ax.imshow(change_mask, cmap="Reds", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_anomaly_zmap(z_map: np.ndarray, ax=None, title: str = "Anomaly Z-score"):
    ax = ax or plt.gca()
    im = ax.imshow(z_map, cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
