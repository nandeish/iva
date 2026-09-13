"""
Satellite raster I/O: reading multi-band Sentinel-2 / Landsat scenes with rasterio,
stacking selected bands, and cutting large scenes into training tiles.

Expected raw data layout (per scene):
    data/raw/<scene_id>/B02.tif
    data/raw/<scene_id>/B03.tif
    ...
    data/raw/<scene_id>/label.tif   (optional, for supervised training)

A scene can also be a single stacked GeoTIFF with bands in a known order --
`load_stacked_scene` handles that case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover - documented dependency, not vendored
    rasterio = None


def _require_rasterio():
    if rasterio is None:
        raise ImportError(
            "rasterio/GDAL is required for raster I/O. Install with "
            "`pip install rasterio` (requires GDAL system libraries)."
        )


def load_scene_bands(scene_dir: str | Path, bands: Sequence[str]) -> np.ndarray:
    """
    Load a set of single-band GeoTIFFs (one per Sentinel-2 band) from `scene_dir`
    and stack them into a (H, W, C) float32 array, resampling every band onto the
    grid of the first (assumed finest resolution) band via nearest-neighbour.

    Returns
    -------
    np.ndarray of shape (H, W, len(bands))
    """
    _require_rasterio()
    scene_dir = Path(scene_dir)

    arrays: List[np.ndarray] = []
    ref_shape: Optional[Tuple[int, int]] = None

    for band in bands:
        band_path = scene_dir / f"{band}.tif"
        if not band_path.exists():
            raise FileNotFoundError(f"Missing band file: {band_path}")
        with rasterio.open(band_path) as src:
            arr = src.read(1).astype(np.float32)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            arr = _resample_to_shape(arr, ref_shape)
        arrays.append(arr)

    stacked = np.stack(arrays, axis=-1)
    return stacked


def _resample_to_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Cheap nearest-neighbour resample used when bands have mismatched resolutions
    (e.g. Sentinel-2 20m/60m bands vs 10m bands). For production use, prefer
    rasterio's warp/reproject with bilinear resampling."""
    from scipy.ndimage import zoom

    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=0)


def load_stacked_scene(path: str | Path) -> Tuple[np.ndarray, Dict]:
    """Load a single multi-band GeoTIFF, returning (H, W, C) array and geo metadata."""
    _require_rasterio()
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
        meta = src.meta.copy()
    return np.transpose(arr, (1, 2, 0)), meta


def load_label_mask(path: str | Path) -> np.ndarray:
    """Load a single-band categorical label raster (H, W) of class indices."""
    _require_rasterio()
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.int64)
    return mask


def normalize_reflectance(arr: np.ndarray, scale: float = 10000.0, clip: bool = True) -> np.ndarray:
    """Convert Sentinel-2 reflectance DN values to [0, 1] and safely handle
    no-data / invalid pixels."""
    out = arr.astype(np.float32) / scale

    # Replace invalid pixels so NaNs cannot propagate into the model/loss.
    out = np.nan_to_num(
        out,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    if clip:
        out = np.clip(out, 0.0, 1.0)

    return out


def tile_scene(
    arr: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
    label: Optional[np.ndarray] = None,
):
    """
    Slide a (tile_size x tile_size) window with the given overlap across a large
    (H, W, C) scene array, yielding tiles (and matching label tiles if given).
    Tiles smaller than tile_size at the scene border are dropped.

    Yields
    ------
    (tile, label_tile, (row_off, col_off))
    """
    h, w = arr.shape[:2]
    stride = tile_size - overlap
    for row in range(0, h - tile_size + 1, stride):
        for col in range(0, w - tile_size + 1, stride):
            tile = arr[row : row + tile_size, col : col + tile_size]
            lbl_tile = None
            if label is not None:
                lbl_tile = label[row : row + tile_size, col : col + tile_size]
            yield tile, lbl_tile, (row, col)


def write_geotiff(
    path: str | Path,
    arr: np.ndarray,
    meta: Dict,
    dtype: str = "float32",
) -> None:
    """Write a (H, W, C) or (H, W) array to disk as a GeoTIFF, reusing metadata
    (transform/crs) captured from an existing scene."""
    _require_rasterio()
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape

    out_meta = meta.copy()
    out_meta.update(count=c, dtype=dtype, height=h, width=w)

    with rasterio.open(path, "w", **out_meta) as dst:
        for i in range(c):
            dst.write(arr[:, :, i].astype(dtype), i + 1)
