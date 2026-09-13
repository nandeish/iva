"""
Evaluation metrics referenced in Section 2.2.6 / Table 4.4 of the report:
Overall Accuracy, Precision, Recall, F1, mean IoU, Cohen's Kappa for
segmentation and change detection; PSNR, SSIM and Spectral Angle Mapper (SAM)
for compression fidelity.
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------
# Segmentation / change-detection (classification-style) metrics
# ----------------------------------------------------------------------
def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Standard (num_classes x num_classes) confusion matrix from flattened
    integer-label prediction/target arrays."""
    mask = (target >= 0) & (target < num_classes)
    idx = num_classes * target[mask].astype(int) + pred[mask].astype(int)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def overall_accuracy(cm: np.ndarray) -> float:
    return float(np.diag(cm).sum() / (cm.sum() + 1e-9))


def per_class_precision_recall_f1(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def mean_iou(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = tp / (union + 1e-9)
    return float(np.nanmean(iou))


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    po = np.diag(cm).sum() / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n**2)
    return float((po - pe) / (1 - pe + 1e-9))


def segmentation_report(pred: np.ndarray, target: np.ndarray, num_classes: int, class_names=None) -> dict:
    cm = confusion_matrix(pred.flatten(), target.flatten(), num_classes)
    precision, recall, f1 = per_class_precision_recall_f1(cm)
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    return {
        "overall_accuracy": overall_accuracy(cm),
        "mean_iou": mean_iou(cm),
        "kappa": cohen_kappa(cm),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for name, p, r, f in zip(names, precision, recall, f1)
        },
        "confusion_matrix": cm.tolist(),
    }


# ----------------------------------------------------------------------
# Compression fidelity metrics
# ----------------------------------------------------------------------
def psnr(original: np.ndarray, reconstructed: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim_metric(original: np.ndarray, reconstructed: np.ndarray, data_range: float = 1.0) -> float:
    """Per-band SSIM averaged over channels, using scikit-image."""
    from skimage.metrics import structural_similarity as ssim

    if original.ndim == 2:
        return float(ssim(original, reconstructed, data_range=data_range))
    scores = [
        ssim(original[..., c], reconstructed[..., c], data_range=data_range)
        for c in range(original.shape[-1])
    ]
    return float(np.mean(scores))


def spectral_angle_mapper(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean Spectral Angle Mapper (radians) between original and reconstructed
    multi-band pixels -- measures spectral-shape fidelity independent of
    brightness, complementing PSNR/SSIM for multi-spectral compression
    evaluation (addresses the research gap noted in Section 5.2).

    Inputs expected as (H, W, C).
    """
    o = original.reshape(-1, original.shape[-1])
    r = reconstructed.reshape(-1, reconstructed.shape[-1])
    dot = np.sum(o * r, axis=1)
    norm_o = np.linalg.norm(o, axis=1)
    norm_r = np.linalg.norm(r, axis=1)
    cos_angle = np.clip(dot / (norm_o * norm_r + 1e-9), -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.mean(angles))


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return float(original_bytes / max(compressed_bytes, 1))
