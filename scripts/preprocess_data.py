"""
Turns raw per-band GeoTIFF scenes in `data/raw/` into the tiled `.npy` layout
expected by the PyTorch Datasets in `src/datasets/`.

Expected raw layout
--------------------
data/raw/
    <scene_id>/
        B02.tif ... B12.tif      # per-band Sentinel-2 GeoTIFFs
        label.tif                # (segmentation only) categorical LULC mask

For change detection, two scenes of the same area at different dates are
paired via a `--pairs` CSV: columns `scene_t1,scene_t2,label` where `label`
is a binary change-mask GeoTIFF path (or omit the label column for
inference-only pairs).

Usage
-----
    # Segmentation tiles
    python scripts/preprocess_data.py --task segmentation \\
        --config configs/config.yaml --out data/processed/segmentation

    # Change-detection tiles
    python scripts/preprocess_data.py --task change \\
        --config configs/config.yaml --pairs data/raw/change_pairs.csv \\
        --out data/processed/change

    # Compression tiles (any scene, no labels needed)
    python scripts/preprocess_data.py --task compression \\
        --config configs/config.yaml --out data/processed/compression
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, set_global_seed
from src.preprocessing.loader import load_scene_bands, load_label_mask, normalize_reflectance, tile_scene
from src.preprocessing.indices import compute_index_stack
from src.preprocessing.transforms import dwt_change_score


def build_input_stack(scene_dir: Path, cfg: dict) -> np.ndarray:
    bands_list = cfg["data"]["bands"]
    raw = load_scene_bands(scene_dir, bands_list)
    raw = normalize_reflectance(raw)
    band_lookup = {name: raw[:, :, i] for i, name in enumerate(bands_list)}
    indices = compute_index_stack(band_lookup, cfg["indices"]["compute"])
    return np.concatenate([raw, indices], axis=-1)   # (H, W, C+K)


def split_ids(ids: list, cfg: dict):
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * cfg["data"]["train_split"])
    n_val = int(n * cfg["data"]["val_split"])
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def preprocess_segmentation(cfg: dict, out_dir: Path, raw_dir: Path):
    scene_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene subdirectories found under {raw_dir}")

    tile_records = []
    for scene_dir in tqdm(scene_dirs, desc="scenes"):
        stack = build_input_stack(scene_dir, cfg)
        label_path = scene_dir / "label.tif"
        label = load_label_mask(label_path) if label_path.exists() else None

        for tile, lbl_tile, (row, col) in tile_scene(
            stack, cfg["data"]["tile_size"], cfg["data"]["tile_overlap"], label
        ):
            if lbl_tile is None:
                continue
            tile_records.append((scene_dir.name, row, col, tile, lbl_tile))

    train_idx, val_idx, test_idx = split_ids(list(range(len(tile_records))), cfg)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx_list in splits.items():
        img_dir = out_dir / split_name / "images"
        mask_dir = out_dir / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(idx_list):
            scene_name, row, col, tile, lbl_tile = tile_records[idx]
            tile_id = f"{scene_name}_{row}_{col}"
            np.save(img_dir / f"{tile_id}.npy", np.transpose(tile, (2, 0, 1)).astype(np.float32))
            np.save(mask_dir / f"{tile_id}.npy", lbl_tile.astype(np.int64))
        print(f"[segmentation/{split_name}] wrote {len(idx_list)} tiles")


def preprocess_change(cfg: dict, out_dir: Path, pairs_csv: Path):
    use_dwt = cfg["change_detection"]["use_dwt_features"]
    nir_idx = cfg["data"]["bands"].index("B08")

    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        pairs = list(reader)

    tile_records = []
    for row_dict in tqdm(pairs, desc="pairs"):
        scene_t1_dir = Path(row_dict["scene_t1"])
        scene_t2_dir = Path(row_dict["scene_t2"])
        label_path = Path(row_dict["label"])

        stack1 = build_input_stack(scene_t1_dir, cfg)
        stack2 = build_input_stack(scene_t2_dir, cfg)
        label = load_label_mask(label_path)

        if use_dwt:
            change_map = dwt_change_score(stack1[:, :, nir_idx], stack2[:, :, nir_idx])[:, :, None]
            stack1 = np.concatenate([stack1, change_map], axis=-1)
            stack2 = np.concatenate([stack2, change_map], axis=-1)

        pair_name = f"{scene_t1_dir.name}_{scene_t2_dir.name}"
        for (t1, _, (r, c)), (t2, _, _) in zip(
            tile_scene(stack1, cfg["data"]["tile_size"], cfg["data"]["tile_overlap"]),
            tile_scene(stack2, cfg["data"]["tile_size"], cfg["data"]["tile_overlap"]),
        ):
            lbl_tile = label[r : r + cfg["data"]["tile_size"], c : c + cfg["data"]["tile_size"]]
            tile_records.append((pair_name, r, c, t1, t2, lbl_tile))

    train_idx, val_idx, test_idx = split_ids(list(range(len(tile_records))), cfg)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx_list in splits.items():
        t1_dir = out_dir / split_name / "t1"
        t2_dir = out_dir / split_name / "t2"
        lbl_dir = out_dir / split_name / "labels"
        for d in (t1_dir, t2_dir, lbl_dir):
            d.mkdir(parents=True, exist_ok=True)
        for idx in idx_list:
            name, r, c, t1, t2, lbl_tile = tile_records[idx]
            tile_id = f"{name}_{r}_{c}"
            np.save(t1_dir / f"{tile_id}.npy", np.transpose(t1, (2, 0, 1)).astype(np.float32))
            np.save(t2_dir / f"{tile_id}.npy", np.transpose(t2, (2, 0, 1)).astype(np.float32))
            np.save(lbl_dir / f"{tile_id}.npy", lbl_tile.astype(np.float32))
        print(f"[change/{split_name}] wrote {len(idx_list)} tiles")


def preprocess_compression(cfg: dict, out_dir: Path, raw_dir: Path):
    """Compression tiles use the full 12-band reflectance stack (no indices)."""
    scene_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    all_bands = cfg["data"]["bands"]   # extend/adjust to 12 bands as needed in config

    tile_records = []
    for scene_dir in tqdm(scene_dirs, desc="scenes"):
        raw = load_scene_bands(scene_dir, all_bands)
        raw = normalize_reflectance(raw)
        for tile, _, (row, col) in tile_scene(raw, cfg["data"]["tile_size"], cfg["data"]["tile_overlap"]):
            tile_records.append((scene_dir.name, row, col, tile))

    train_idx, val_idx, test_idx = split_ids(list(range(len(tile_records))), cfg)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx_list in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in idx_list:
            name, row, col, tile = tile_records[idx]
            tile_id = f"{name}_{row}_{col}"
            np.save(split_dir / f"{tile_id}.npy", np.transpose(tile, (2, 0, 1)).astype(np.float32))
        print(f"[compression/{split_name}] wrote {len(idx_list)} tiles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["segmentation", "change", "compression"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--pairs", default=None, help="CSV of scene_t1,scene_t2,label (change task only)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    out_dir = Path(args.out)

    if args.task == "segmentation":
        preprocess_segmentation(cfg, out_dir, Path(args.raw_dir))
    elif args.task == "change":
        if not args.pairs:
            raise ValueError("--pairs CSV is required for --task change")
        preprocess_change(cfg, out_dir, Path(args.pairs))
    elif args.task == "compression":
        preprocess_compression(cfg, out_dir, Path(args.raw_dir))


if __name__ == "__main__":
    main()
