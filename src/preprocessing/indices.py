"""
Spectral vegetation / water / burn indices computed from Sentinel-2 surface
reflectance bands, as described in Section 4.3 of the project report.

All functions expect reflectance already scaled to [0, 1] (see
`preprocessing.loader.normalize_reflectance`) and operate on 2-D (H, W)
band arrays. A small epsilon avoids division-by-zero on masked / no-data pixels.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Sensitive to green vegetation density."""
    return (nir - red) / (nir + red + EPS)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
        g: float = 2.5, c1: float = 6.0, c2: float = 7.5, l: float = 1.0) -> np.ndarray:
    """Enhanced Vegetation Index. Reduces atmospheric and soil background effects,
    performs better than NDVI in dense canopy."""
    return g * (nir - red) / (nir + c1 * red - c2 * blue + l + EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters). Delineates open water /
    surface moisture."""
    return (green - nir) / (green + nir + EPS)


def savi(nir: np.ndarray, red: np.ndarray, l: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index. `l` is a soil-brightness correction factor
    (0 = dense vegetation, 1 = bare soil)."""
    return ((nir - red) / (nir + red + l + EPS)) * (1 + l)


def bai(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Burned Area Index. Highlights post-fire burn scars — large positive values
    indicate charcoal/ash signatures."""
    return 1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + EPS)


def compute_index_stack(
    bands: dict[str, np.ndarray],
    which: list[str] = ("NDVI", "EVI", "NDWI", "SAVI", "BAI"),
) -> np.ndarray:
    """
    Compute a set of indices from a dict of named reflectance bands, e.g.
        bands = {"B02": blue, "B03": green, "B04": red, "B08": nir}
    and stack them into a (H, W, len(which)) array in the requested order.
    """
    lookup = {
        "NDVI": lambda b: ndvi(b["B08"], b["B04"]),
        "EVI": lambda b: evi(b["B08"], b["B04"], b["B02"]),
        "NDWI": lambda b: ndwi(b["B03"], b["B08"]),
        "SAVI": lambda b: savi(b["B08"], b["B04"]),
        "BAI": lambda b: bai(b["B04"], b["B08"]),
    }
    missing = [name for name in which if name not in lookup]
    if missing:
        raise ValueError(f"Unknown indices requested: {missing}")

    layers = [np.clip(lookup[name](bands), -5, 5) for name in which]
    return np.stack(layers, axis=-1).astype(np.float32)
