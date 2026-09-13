import numpy as np
import torch

from src.inference.predict_segmentation import (
    prepare_input_stack,
    build_model_for_inference,
)
from src.utils.config import load_config, get_device

cfg = load_config("configs/config.yaml")
device = get_device(cfg["project"]["device"])

model = build_model_for_inference(
    cfg,
    "checkpoints/segmentation/best_model.pt",
    device,
)

image = prepare_input_stack("data/raw/bangalore_2024", cfg)

tile_size = cfg["data"]["tile_size"]
overlap = cfg["data"]["tile_overlap"]
stride = tile_size - overlap

print("Full input:", image.shape)
print("Tile size:", tile_size)
print("Stride:", stride)

with torch.no_grad():
    for row in [0, stride, stride * 2]:
        for col in [0, stride, stride * 2]:
            tile = image[:, row:row+tile_size, col:col+tile_size]

            if tile.shape != (image.shape[0], tile_size, tile_size):
                continue

            x = torch.from_numpy(tile).unsqueeze(0).float().to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            classes, counts = np.unique(pred, return_counts=True)

            print(
                f"tile row={row}, col={col}: "
                f"{dict(zip(classes.tolist(), counts.tolist()))}"
            )
