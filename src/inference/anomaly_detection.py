"""
Near-real-time disaster/anomaly detection (Section 2.2.4 / 5.2 of the report):
a Fourier-detrended Z-score approach applied to a per-pixel vegetation-index
time series (default NDVI). The idea:

  1. For each pixel, fit and subtract a low-order Fourier series (a handful
     of seasonal harmonics) from its historical index time series -- this
     removes normal phenological cyclicity (green-up/senescence) without
     needing a full growing-season stack for every prediction.
  2. Compute a Z-score of the latest detrended residual against the
     historical residual distribution.
  3. Flag pixels whose |Z| exceeds `z_threshold` as anomalies (e.g. flood
     inundation, wildfire burn scar, drought stress), then filter tiny
     specks below `min_alert_area_px`.

This directly targets the research gap identified in Section 5.2: existing
NDVI-based monitoring (papers 12, 13) runs in batch mode without an explicit
seasonality/anomaly separation suited to sub-30-minute alerting.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


def fit_fourier_seasonality(time_series: np.ndarray, timestamps_doy: np.ndarray, harmonics: int = 3) -> np.ndarray:
    """
    Fit a truncated Fourier series (annual period) to a 1-D pixel time series
    and return the fitted seasonal curve evaluated at the same timestamps.

    Parameters
    ----------
    time_series : (T,) array of index values (e.g. NDVI) for one pixel across T acquisitions
    timestamps_doy : (T,) array of day-of-year for each acquisition
    harmonics : number of sine/cosine harmonic pairs to fit
    """
    t = 2 * np.pi * timestamps_doy / 365.25
    design = [np.ones_like(t)]
    for k in range(1, harmonics + 1):
        design.append(np.cos(k * t))
        design.append(np.sin(k * t))
    design = np.stack(design, axis=1)   # (T, 1 + 2*harmonics)

    coeffs, *_ = np.linalg.lstsq(design, time_series, rcond=None)
    return design @ coeffs


def detrended_zscore_map(
    index_stack: np.ndarray,
    timestamps_doy: Sequence[int],
    latest_index: int = -1,
    harmonics: int = 3,
) -> np.ndarray:
    """
    Vectorised Fourier-detrended Z-score over a full spatial stack.

    Parameters
    ----------
    index_stack : (T, H, W) array of a vegetation index (e.g. NDVI) across T
                  historical acquisitions, most recent last.
    timestamps_doy : length-T sequence of day-of-year values.
    latest_index : which time step to score as the "current" observation
                   (default: the most recent).
    harmonics : Fourier harmonics used to model seasonality.

    Returns
    -------
    (H, W) array of Z-scores for the `latest_index` acquisition.
    """
    t_arr = np.asarray(timestamps_doy, dtype=np.float32)
    n_t, h, w = index_stack.shape
    flat = index_stack.reshape(n_t, -1)   # (T, H*W)

    theta = 2 * np.pi * t_arr / 365.25
    design = [np.ones_like(theta)]
    for k in range(1, harmonics + 1):
        design.append(np.cos(k * theta))
        design.append(np.sin(k * theta))
    design = np.stack(design, axis=1)     # (T, 1+2H)

    # Least-squares fit for every pixel simultaneously: coeffs shape (1+2H, H*W)
    coeffs, *_ = np.linalg.lstsq(design, flat, rcond=None)
    fitted = design @ coeffs              # (T, H*W)
    residuals = flat - fitted             # (T, H*W)

    resid_std = residuals.std(axis=0) + 1e-6
    resid_mean = residuals.mean(axis=0)
    z = (residuals[latest_index] - resid_mean) / resid_std

    return z.reshape(h, w)


def flag_anomalies(z_map: np.ndarray, z_threshold: float = 2.5, min_area_px: int = 25) -> np.ndarray:
    """
    Threshold a Z-score map and remove connected components smaller than
    `min_area_px` to suppress speckle noise. Returns a boolean anomaly mask.
    """
    raw_mask = np.abs(z_map) > z_threshold
    labeled, n_features = ndimage.label(raw_mask)
    if n_features == 0:
        return raw_mask

    sizes = ndimage.sum(raw_mask, labeled, range(1, n_features + 1))
    small_labels = np.where(sizes < min_area_px)[0] + 1
    cleaned = raw_mask.copy()
    for lbl in small_labels:
        cleaned[labeled == lbl] = False
    return cleaned


def detect_anomalies(
    index_stack: np.ndarray,
    timestamps_doy: Sequence[int],
    harmonics: int = 3,
    z_threshold: float = 2.5,
    min_area_px: int = 25,
) -> dict:
    """End-to-end convenience wrapper: fit seasonality, score the latest date,
    threshold, and return both the continuous Z-map and the boolean alert mask."""
    z_map = detrended_zscore_map(index_stack, timestamps_doy, harmonics=harmonics)
    alert_mask = flag_anomalies(z_map, z_threshold=z_threshold, min_area_px=min_area_px)
    return {
        "z_map": z_map,
        "alert_mask": alert_mask,
        "alert_pixel_count": int(alert_mask.sum()),
        "alert_area_fraction": float(alert_mask.mean()),
    }
