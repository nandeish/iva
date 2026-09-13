"""
Run a trained segmentation model over a full (large) scene using sliding-window
tiling with overlap-averaging, and write the resulting class-index map as a
GeoTIFF.

Usage
-----
    python -m src.inference.predict_segmentation \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/segmentation/best_model.pt \\
        --scene data/raw/scene_001 \\
        --out outputs/scene_001_lulc.tif
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.utils.config import load_config, get_device
from src.preprocessing.loader import load_scene_bands, normalize_reflectance, write_geotiff
from src.preprocessing.indices import compute_index_stack


def build_model_for_inference(cfg: dict, checkpoint_path: str, device):
    from src.training.train_segmentation import build_model

    model = build_model(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_input_stack(scene_dir: str, cfg: dict) -> np.ndarray:
    """Load raw bands, normalise, compute indices, and concatenate into the
    (C, H, W) tensor the model expects."""
    bands_list = cfg["data"]["bands"]
    raw = load_scene_bands(scene_dir, bands_list)          # (H, W, C)
    raw = normalize_reflectance(raw)

    band_lookup = {name: raw[:, :, i] for i, name in enumerate(bands_list)}
    indices = compute_index_stack(band_lookup, cfg["indices"]["compute"])  # (H, W, K)

    stacked = np.concatenate([raw, indices], axis=-1)      # (H, W, C+K)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)  # (C, H, W)


def sliding_window_predict(model, image: np.ndarray, tile_size: int, overlap: int, num_classes: int, device):
    """Predict class probabilities over a full scene via a sliding window,
    averaging overlapping predictions before taking the argmax."""
    c, h, w = image.shape
    stride = tile_size - overlap
    prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)
    count_accum = np.zeros((h, w), dtype=np.float32)

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        h, w = image.shape[1:]
        prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)
        count_accum = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for row in range(0, h - tile_size + 1, stride):
            for col in range(0, w - tile_size + 1, stride):
                tile = image[:, row : row + tile_size, col : col + tile_size]
                tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)
                logits = model(tile_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob_accum[:, row : row + tile_size, col : col + tile_size] += probs
                count_accum[row : row + tile_size, col : col + tile_size] += 1

    count_accum = np.maximum(count_accum, 1e-6)
    prob_accum /= count_accum[None, :, :]
    pred = np.argmax(prob_accum, axis=0)
    return pred[: image.shape[1] - pad_h, : image.shape[2] - pad_w] if (pad_h or pad_w) else pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True, help="directory with per-band GeoTIFFs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-band", default=None, help="band file to copy geo-metadata from; defaults to first band")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])

    model = build_model_for_inference(cfg, args.checkpoint, device)
    image_stack = prepare_input_stack(args.scene, cfg)

    pred = sliding_window_predict(
        model, image_stack,
        tile_size=cfg["data"]["tile_size"],
        overlap=cfg["data"]["tile_overlap"],
        num_classes=cfg["data"]["num_classes"],
        device=device,
    )

    import rasterio
    from pathlib import Path

    ref_band = args.reference_band or f"{cfg['data']['bands'][0]}.tif"
    with rasterio.open(Path(args.scene) / ref_band) as src:
        meta = src.meta.copy()

    write_geotiff(args.out, pred.astype(np.uint8), meta, dtype="uint8")
    print(f"Wrote LULC prediction to {args.out}")


if __name__ == "__main__":
    main()
