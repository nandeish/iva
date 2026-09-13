"""
Training entry point for the land-cover semantic segmentation models.

Usage
-----
    python -m src.training.train_segmentation --config configs/config.yaml

The architecture actually trained is chosen by `segmentation.architecture` in
the config (`unet` | `deeplabv3plus` | `segformer`). Checkpoints and
TensorBoard logs are written under `paths.checkpoints` / `paths.logs`.

NOTE: this script defines the full training loop but is not executed as part
of project delivery -- run it yourself once real tiled data is in
`data/processed/segmentation/` (see scripts/preprocess_data.py and the README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.config import load_config, set_global_seed, get_device
from src.datasets.lulc_dataset import LULCSegmentationDataset
from src.training.losses import ComboLoss
from src.evaluation.metrics import segmentation_report


def build_model(cfg: dict) -> torch.nn.Module:
    seg_cfg = cfg["segmentation"]
    arch = seg_cfg["architecture"]
    in_channels = seg_cfg["in_channels"]
    num_classes = cfg["data"]["num_classes"]

    if arch == "unet":
        from src.models.unet import UNet
        return UNet(in_channels=in_channels, num_classes=num_classes)
    elif arch == "deeplabv3plus":
        from src.models.deeplabv3plus import build_deeplabv3plus
        dl_cfg = seg_cfg["deeplabv3plus"]
        return build_deeplabv3plus(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_name=dl_cfg["encoder"],
            encoder_weights=dl_cfg["encoder_weights"],
        )
    elif arch == "segformer":
        from src.models.segformer import SegFormerLULC
        sf_cfg = seg_cfg["segformer"]
        return SegFormerLULC(in_channels=in_channels, num_classes=num_classes, pretrained=sf_cfg["pretrained"])
    else:
        raise ValueError(f"Unknown segmentation architecture: {arch}")


def run_epoch(model, loader, criterion, optimizer, device, num_classes, train: bool = True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks, _ in tqdm(loader, desc="train" if train else "val", leave=False):
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    import numpy as np
    preds_cat = np.concatenate([p.flatten() for p in all_preds])
    targets_cat = np.concatenate([t.flatten() for t in all_targets])
    report = segmentation_report(preds_cat, targets_cat, num_classes)
    report["loss"] = total_loss / len(loader.dataset)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default="data/processed/segmentation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    train_ds = LULCSegmentationDataset(args.data_root, split="train")
    val_ds = LULCSegmentationDataset(args.data_root, split="val")

    seg_cfg = cfg["segmentation"]
    train_loader = DataLoader(train_ds, batch_size=seg_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=seg_cfg["batch_size"], shuffle=False, num_workers=4)

    model = build_model(cfg).to(device)
    criterion = ComboLoss(num_classes=cfg["data"]["num_classes"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg_cfg["lr"], weight_decay=seg_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=seg_cfg["epochs"])

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "segmentation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["logs"]) / "segmentation"))

    best_miou = 0.0
    for epoch in range(seg_cfg["epochs"]):
        train_report = run_epoch(model, train_loader, criterion, optimizer, device, cfg["data"]["num_classes"], train=True)
        val_report = run_epoch(model, val_loader, criterion, optimizer, device, cfg["data"]["num_classes"], train=False)
        scheduler.step()

        writer.add_scalar("train/loss", train_report["loss"], epoch)
        writer.add_scalar("val/loss", val_report["loss"], epoch)
        writer.add_scalar("val/mean_iou", val_report["mean_iou"], epoch)
        writer.add_scalar("val/overall_accuracy", val_report["overall_accuracy"], epoch)

        print(
            f"Epoch {epoch+1}/{seg_cfg['epochs']} | "
            f"train_loss={train_report['loss']:.4f} | "
            f"val_loss={val_report['loss']:.4f} | "
            f"val_mIoU={val_report['mean_iou']:.4f} | "
            f"val_OA={val_report['overall_accuracy']:.4f}"
        )

        if val_report["mean_iou"] > best_miou:
            best_miou = val_report["mean_iou"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        torch.save(model.state_dict(), ckpt_dir / "last_model.pt")

    writer.close()
    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
