"""
Training entry point for the multi-spectral lossy compression autoencoder.

Usage
-----
    python -m src.training.train_compression --config configs/config.yaml

Trains either the plain `ConvAutoencoder` (MSE + entropy proxy) or the
CompressAI `HyperpriorCompressor` (proper rate-distortion loss with a real
entropy model), selected by `compression.architecture` in the config.
Evaluation each epoch reports PSNR / SSIM / SAM on held-out tiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.config import load_config, set_global_seed, get_device
from src.models.compression_autoencoder import ConvAutoencoder, HyperpriorCompressor, rate_distortion_loss
from src.evaluation.metrics import psnr, ssim_metric, spectral_angle_mapper


class SpectralTileDataset(Dataset):
    """Loads full 12-band reflectance tiles (no labels needed) for
    self-supervised compression training."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.dir = Path(root) / split
        self.files = sorted(self.dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No tiles found in {self.dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])   # (C, H, W), reflectance in [0,1]
        return torch.from_numpy(arr.astype(np.float32))


def build_model(cfg: dict, device):
    comp_cfg = cfg["compression"]
    in_channels = comp_cfg["target_bands"]
    if comp_cfg["architecture"] == "hyperprior":
        model = HyperpriorCompressor(in_channels=in_channels)
    else:
        model = ConvAutoencoder(in_channels=in_channels, latent_channels=comp_cfg["latent_channels"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default="data/processed/compression")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    comp_cfg = cfg["compression"]
    train_ds = SpectralTileDataset(args.data_root, "train")
    val_ds = SpectralTileDataset(args.data_root, "val")
    train_loader = DataLoader(train_ds, batch_size=comp_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=comp_cfg["batch_size"], shuffle=False, num_workers=4)

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=comp_cfg["lr"])

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "compression"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["logs"]) / "compression"))

    best_psnr = 0.0
    is_hyperprior = comp_cfg["architecture"] == "hyperprior"

    for epoch in range(comp_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="train", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            if is_hyperprior:
                out = model(batch)
                loss, mse, bpp = rate_distortion_loss(out["x_hat"], batch, out["likelihoods"])
            else:
                recon, _ = model(batch)
                loss, mse, bpp = rate_distortion_loss(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        psnr_scores, ssim_scores, sam_scores = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                batch = batch.to(device)
                if is_hyperprior:
                    out = model(batch)
                    recon = out["x_hat"]
                else:
                    recon, _ = model(batch)
                recon = torch.clamp(recon, 0.0, 1.0)

                orig_np = batch.cpu().numpy().transpose(0, 2, 3, 1)
                recon_np = recon.cpu().numpy().transpose(0, 2, 3, 1)
                for o, r in zip(orig_np, recon_np):
                    psnr_scores.append(psnr(o, r))
                    ssim_scores.append(ssim_metric(o, r))
                    sam_scores.append(spectral_angle_mapper(o, r))

        mean_psnr = float(np.mean(psnr_scores))
        mean_ssim = float(np.mean(ssim_scores))
        mean_sam = float(np.mean(sam_scores))

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/psnr", mean_psnr, epoch)
        writer.add_scalar("val/ssim", mean_ssim, epoch)
        writer.add_scalar("val/sam", mean_sam, epoch)

        print(
            f"Epoch {epoch+1}/{comp_cfg['epochs']} | train_loss={train_loss:.4f} | "
            f"val_PSNR={mean_psnr:.2f}dB | val_SSIM={mean_ssim:.4f} | val_SAM={mean_sam:.4f}rad"
        )

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        torch.save(model.state_dict(), ckpt_dir / "last_model.pt")

    writer.close()
    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
