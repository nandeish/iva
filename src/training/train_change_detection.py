"""
Training entry point for the Siamese U-Net bitemporal change-detection model.

Usage
-----
    python -m src.training.train_change_detection --config configs/config.yaml

Optionally concatenates a DWT change-intensity feature (see
`preprocessing.transforms.dwt_change_score`) as an extra input channel when
`change_detection.use_dwt_features` is true in the config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.config import load_config, set_global_seed, get_device
from src.datasets.change_dataset import ChangeDetectionDataset
from src.models.siamese_change import SiameseUNetChangeDetector
from src.training.losses import BinaryDiceBCELoss


def compute_binary_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    preds = preds.astype(bool)
    targets = targets.astype(bool)
    tp = np.logical_and(preds, targets).sum()
    fp = np.logical_and(preds, ~targets).sum()
    fn = np.logical_and(~preds, targets).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for img1, img2, label, _ in tqdm(loader, desc="train" if train else "val", leave=False):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(img1, img2).squeeze(1)
            loss = criterion(logits, label)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * img1.size(0)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(label.cpu().numpy())

    preds_cat = np.concatenate([p.flatten() for p in all_preds])
    targets_cat = np.concatenate([t.flatten() for t in all_targets])
    metrics = compute_binary_metrics(preds_cat, targets_cat)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default="data/processed/change")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    train_ds = ChangeDetectionDataset(args.data_root, split="train")
    val_ds = ChangeDetectionDataset(args.data_root, split="val")

    cd_cfg = cfg["change_detection"]
    train_loader = DataLoader(train_ds, batch_size=cd_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cd_cfg["batch_size"], shuffle=False, num_workers=4)

    in_channels = cfg["segmentation"]["in_channels"] + (1 if cd_cfg["use_dwt_features"] else 0)
    model = SiameseUNetChangeDetector(in_channels=in_channels).to(device)
    criterion = BinaryDiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cd_cfg["lr"])

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "change_detection"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["logs"]) / "change_detection"))

    best_f1 = 0.0
    for epoch in range(cd_cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)

        print(
            f"Epoch {epoch+1}/{cd_cfg['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_F1={val_metrics['f1']:.4f} | val_IoU={val_metrics['iou']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        torch.save(model.state_dict(), ckpt_dir / "last_model.pt")

    writer.close()
    print(f"Training complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
