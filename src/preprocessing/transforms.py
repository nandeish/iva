"""
Classical signal-processing transforms used throughout the pipeline
(Section 4.2 of the report):

  * DFT  -- frequency-domain filtering / periodic pattern detection
  * DWT  -- multi-resolution sub-band decomposition (db4), used as auxiliary
            features for both segmentation and change detection
  * DCT  -- block-based transform underpinning the lossy compression module

Each transform is implemented per single-band 2-D array; callers loop over
channels for multi-spectral cubes.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pywt
from scipy.fftpack import dct, idct
from scipy.fft import fft2, ifft2, fftshift, ifftshift


# ----------------------------------------------------------------------
# Discrete Fourier Transform
# ----------------------------------------------------------------------
def dft2(band: np.ndarray) -> np.ndarray:
    """2-D FFT of a single band, shifted so the DC component is centred."""
    return fftshift(fft2(band))


def idft2(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of `dft2`. Returns the real-valued spatial-domain band."""
    return np.real(ifft2(ifftshift(spectrum)))


def bandpass_filter(band: np.ndarray, low_cutoff: float = 0.0, high_cutoff: float = 0.1) -> np.ndarray:
    """
    Suppress frequency components outside [low_cutoff, high_cutoff] of the
    Nyquist frequency (0-1 range). Useful for isolating periodic structures
    such as agricultural field rows or urban street grids, and for noise
    suppression prior to segmentation.
    """
    h, w = band.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    norm_dist = dist / (max_dist + 1e-9)

    mask = (norm_dist >= low_cutoff) & (norm_dist <= high_cutoff)
    spectrum = dft2(band)
    filtered_spectrum = spectrum * mask
    return idft2(filtered_spectrum)


# ----------------------------------------------------------------------
# Discrete Wavelet Transform (Daubechies-4)
# ----------------------------------------------------------------------
def dwt2_decompose(band: np.ndarray, wavelet: str = "db4", levels: int = 2):
    """
    Multi-level 2-D DWT decomposition. Returns the full coefficient list from
    `pywt.wavedec2`:
        [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    where cA is the approximation sub-band and cH/cV/cD are horizontal /
    vertical / diagonal detail sub-bands.
    """
    return pywt.wavedec2(band, wavelet=wavelet, level=levels)


def dwt2_feature_map(band: np.ndarray, wavelet: str = "db4", levels: int = 2) -> np.ndarray:
    """
    Collapse a multi-level DWT decomposition into a single feature map the same
    size as the input by upsampling each sub-band back to the original
    resolution and stacking energy (magnitude) per level. Useful as an
    auxiliary input channel concatenated onto the raw spectral bands, or as a
    compact descriptor for bitemporal change detection (Section 5.2).

    Returns
    -------
    np.ndarray of shape (H, W, levels) -- one energy map per decomposition level.
    """
    from scipy.ndimage import zoom

    coeffs = dwt2_decompose(band, wavelet=wavelet, levels=levels)
    h, w = band.shape
    feature_maps = []

    # coeffs[1:] holds (cH, cV, cD) tuples from coarsest to finest level
    for detail in coeffs[1:]:
        cH, cV, cD = detail
        energy = np.sqrt(cH**2 + cV**2 + cD**2)
        zoom_factors = (h / energy.shape[0], w / energy.shape[1])
        feature_maps.append(zoom(energy, zoom_factors, order=1))

    return np.stack(feature_maps, axis=-1).astype(np.float32)


def dwt_change_score(band_t1: np.ndarray, band_t2: np.ndarray, wavelet: str = "db4", levels: int = 2) -> np.ndarray:
    """
    Bitemporal DWT differencing: decompose both dates, take the absolute
    difference of detail-sub-band energies, and recombine into a single
    change-intensity map (same spatial size as input). High values indicate
    likely structural change between t1 and t2.
    """
    f1 = dwt2_feature_map(band_t1, wavelet, levels)
    f2 = dwt2_feature_map(band_t2, wavelet, levels)
    diff = np.abs(f1 - f2).mean(axis=-1)
    return diff


# ----------------------------------------------------------------------
# Discrete Cosine Transform (block-based, JPEG-style)
# ----------------------------------------------------------------------
def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def block_dct(band: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Apply an 8x8-block 2-D DCT across a single band, zero-padding if the band's
    dimensions aren't a multiple of `block_size`. Returns coefficients in the
    same (padded) spatial layout, block-by-block.
    """
    h, w = band.shape
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    padded = np.pad(band, ((0, pad_h), (0, pad_w)), mode="reflect")

    out = np.zeros_like(padded)
    for i in range(0, padded.shape[0], block_size):
        for j in range(0, padded.shape[1], block_size):
            block = padded[i : i + block_size, j : j + block_size]
            out[i : i + block_size, j : j + block_size] = _dct2(block)
    return out


def block_idct(coeffs: np.ndarray, block_size: int = 8, original_shape: Tuple[int, int] | None = None) -> np.ndarray:
    """Inverse of `block_dct`. Crops back to `original_shape` if provided."""
    out = np.zeros_like(coeffs)
    for i in range(0, coeffs.shape[0], block_size):
        for j in range(0, coeffs.shape[1], block_size):
            block = coeffs[i : i + block_size, j : j + block_size]
            out[i : i + block_size, j : j + block_size] = _idct2(block)
    if original_shape is not None:
        out = out[: original_shape[0], : original_shape[1]]
    return out


def quantize_dct(coeffs: np.ndarray, block_size: int = 8, quality: int = 50, data_range: float = 1.0) -> np.ndarray:
    """
    Apply a JPEG-style luminance quantization matrix (scaled by `quality`,
    1-100) block-by-block to DCT coefficients, zeroing high-frequency terms and
    achieving energy compaction / lossy compression. Higher `quality` -> less
    aggressive quantization -> higher fidelity, lower compression ratio.

    `data_range` is the dynamic range of the *input pixel values* the DCT was
    computed from (e.g. 1.0 for normalised [0,1] reflectance, 255.0 for 8-bit
    imagery). The standard JPEG quantization table below is calibrated for a
    0-255 range, so it is rescaled by `data_range / 255` -- without this, the
    default table zeroes out virtually every coefficient of [0,1]-scaled
    reflectance data (step sizes of 10-100 against pixel values of ~0-1).
    """
    base_qtable = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.float32)

    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    qtable = np.clip((base_qtable * scale + 50) / 100, 1, 255)
    qtable = qtable * (data_range / 255.0)

    # tile qtable to arbitrary block_size via nearest-neighbour resize if needed
    if block_size != 8:
        from scipy.ndimage import zoom
        qtable = zoom(qtable, (block_size / 8, block_size / 8), order=0)

    out = np.zeros_like(coeffs)
    for i in range(0, coeffs.shape[0], block_size):
        for j in range(0, coeffs.shape[1], block_size):
            block = coeffs[i : i + block_size, j : j + block_size]
            out[i : i + block_size, j : j + block_size] = np.round(block / qtable) * qtable
    return out
