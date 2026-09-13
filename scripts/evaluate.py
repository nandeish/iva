"""
Compute final held-out test-set metrics for a trained segmentation or
change-detection model and dump a JSON report.

Usage
-----
    python scripts/evaluate.py --task segmentation \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/segmentation/best_model.pt \\
        --data-root data/processed/segmentation \\
        --out outputs/segmentation_test_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device
from src.evaluation.metrics import segmentation_report


def evaluate_segmentation(cfg, checkpoint, data_root, device):
    from src.datasets.lulc_dataset import LULCSegmentationDataset
    from src.training.train_segmentation import build_model

    test_ds = LULCSegmentationDataset(data_root, split="test")
    loader = DataLoader(test_ds, batch_size=cfg["segmentation"]["batch_size"], shuffle=False)

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.numpy())

    preds_cat = np.concatenate([p.flatten() for p in all_preds])
    targets_cat = np.concatenate([t.flatten() for t in all_targets])
    return segmentation_report(preds_cat, targets_cat, cfg["data"]["num_classes"], cfg["data"]["class_names"])


def evaluate_change(cfg, checkpoint, data_root, device):
    from src.datasets.change_dataset import ChangeDetectionDataset
    from src.models.siamese_change import SiameseUNetChangeDetector
    from src.training.train_change_detection import compute_binary_metrics

    test_ds = ChangeDetectionDataset(data_root, split="test")
    loader = DataLoader(test_ds, batch_size=cfg["change_detection"]["batch_size"], shuffle=False)

    in_channels = cfg["segmentation"]["in_channels"] + (1 if cfg["change_detection"]["use_dwt_features"] else 0)
    model = SiameseUNetChangeDetector(in_channels=in_channels)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for img1, img2, label, _ in loader:
            img1, img2 = img1.to(device), img2.to(device)
            logits = model(img1, img2).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(label.numpy())

    preds_cat = np.concatenate([p.flatten() for p in all_preds])
    targets_cat = np.concatenate([t.flatten() for t in all_targets])
    return compute_binary_metrics(preds_cat, targets_cat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["segmentation", "change"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])

    if args.task == "segmentation":
        report = evaluate_segmentation(cfg, args.checkpoint, args.data_root, device)
    else:
        report = evaluate_change(cfg, args.checkpoint, args.data_root, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
