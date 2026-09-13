"""
Run the trained Siamese change-detection model on a pair of scenes (t1, t2)
covering the same area, and write a binary change map GeoTIFF.

Usage
-----
    python -m src.inference.predict_change \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/change_detection/best_model.pt \\
        --scene-t1 data/raw/scene_2023 \\
        --scene-t2 data/raw/scene_2024 \\
        --out outputs/change_2023_2024.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import rasterio

from src.utils.config import load_config, get_device
from src.preprocessing.loader import write_geotiff
from src.preprocessing.transforms import dwt_change_score
from src.inference.predict_segmentation import prepare_input_stack
from src.models.siamese_change import SiameseUNetChangeDetector


def maybe_add_dwt_channel(stack_t1: np.ndarray, stack_t2: np.ndarray, use_dwt: bool):
    """Optionally append a single-channel DWT bitemporal change-intensity map
    (computed from the NIR band, index 6 in the default 10-band ordering) to
    both input stacks, matching the `in_channels` used at training time."""
    if not use_dwt:
        return stack_t1, stack_t2

    nir_idx = 6   # B08 position in configs/config.yaml `data.bands`
    change_map = dwt_change_score(stack_t1[nir_idx], stack_t2[nir_idx])
    change_map = change_map[None, :, :].astype(np.float32)

    stack_t1 = np.concatenate([stack_t1, change_map], axis=0)
    stack_t2 = np.concatenate([stack_t2, change_map], axis=0)
    return stack_t1, stack_t2


def otsu_threshold(prob_map: np.ndarray) -> float:
    from skimage.filters import threshold_otsu

    return float(threshold_otsu(prob_map))


def sliding_window_change(model, img1, img2, tile_size, overlap, device):
    c, h, w = img1.shape
    stride = tile_size - overlap
    prob_accum = np.zeros((h, w), dtype=np.float32)
    count_accum = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for row in range(0, h - tile_size + 1, stride):
            for col in range(0, w - tile_size + 1, stride):
                t1 = torch.from_numpy(img1[:, row:row+tile_size, col:col+tile_size]).unsqueeze(0).to(device)
                t2 = torch.from_numpy(img2[:, row:row+tile_size, col:col+tile_size]).unsqueeze(0).to(device)
                logits = model(t1, t2)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                prob_accum[row:row+tile_size, col:col+tile_size] += probs
                count_accum[row:row+tile_size, col:col+tile_size] += 1

    count_accum = np.maximum(count_accum, 1e-6)
    return prob_accum / count_accum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-t1", required=True)
    parser.add_argument("--scene-t2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])
    cd_cfg = cfg["change_detection"]

    stack_t1 = prepare_input_stack(args.scene_t1, cfg)
    stack_t2 = prepare_input_stack(args.scene_t2, cfg)
    stack_t1, stack_t2 = maybe_add_dwt_channel(stack_t1, stack_t2, cd_cfg["use_dwt_features"])

    in_channels = stack_t1.shape[0]
    model = SiameseUNetChangeDetector(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    prob_map = sliding_window_change(
        model, stack_t1, stack_t2,
        tile_size=cfg["data"]["tile_size"], overlap=cfg["data"]["tile_overlap"], device=device,
    )

    if cd_cfg["threshold_method"] == "otsu":
        thresh = otsu_threshold(prob_map)
    else:
        thresh = cd_cfg["fixed_threshold"]
    change_mask = (prob_map > thresh).astype(np.uint8)

    ref_band = f"{cfg['data']['bands'][0]}.tif"
    with rasterio.open(Path(args.scene_t1) / ref_band) as src:
        meta = src.meta.copy()

    write_geotiff(args.out, change_mask, meta, dtype="uint8")
    print(f"Wrote change map to {args.out} (threshold={thresh:.4f})")


if __name__ == "__main__":
    main()
