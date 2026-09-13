#!/usr/bin/env bash
set -e

CONFIG="configs/config.yaml"
SCENE="data/raw/bangalore_2024"
PROCESSED="data/processed/segmentation"
CHECKPOINT="checkpoints/segmentation/best_model.pt"
OUTPUT="outputs/bangalore_2024_lulc.tif"

echo "=========================================="
echo " Bengaluru LULC Pipeline - Continuation"
echo "=========================================="

# --------------------------------------------------
# 1. Check required files
# --------------------------------------------------
echo ""
echo "[1/5] Checking input data..."

for band in B02 B03 B04 B05 B06 B07 B08 B8A B11 B12; do
    if [ ! -f "$SCENE/$band.tif" ]; then
        echo "ERROR: Missing $SCENE/$band.tif"
        exit 1
    fi
done

if [ ! -f "$SCENE/label.tif" ]; then
    echo "ERROR: Missing $SCENE/label.tif"
    exit 1
fi

echo "All Sentinel-2 bands + label found."

# --------------------------------------------------
# 2. Preprocess segmentation data
# --------------------------------------------------
echo ""
echo "[2/5] Preprocessing segmentation tiles..."

rm -rf "$PROCESSED"

python scripts/preprocess_data.py \
    --task segmentation \
    --config "$CONFIG" \
    --raw-dir data/raw \
    --out "$PROCESSED"

echo "Segmentation preprocessing complete."

# --------------------------------------------------
# 3. Run inference
# --------------------------------------------------
echo ""
echo "[3/5] Running Bengaluru LULC inference..."

mkdir -p outputs

python -m src.inference.predict_segmentation \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --scene "$SCENE" \
    --out "$OUTPUT"

echo "Prediction written to $OUTPUT"

# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------
echo ""
echo "[4/5] Evaluating segmentation..."

python scripts/evaluate.py \
    --task segmentation \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$PROCESSED" \
    --out outputs/bangalore_2024_segmentation_report.json

echo "Evaluation complete."

# --------------------------------------------------
# 5. Generate visual map
# --------------------------------------------------
echo ""
echo "[5/5] Generating visual LULC map..."

python -c "
import yaml
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from src.utils.visualization import plot_lulc_map

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

with rasterio.open('$OUTPUT') as src:
    pred = src.read(1)

class_names = cfg['data']['class_names']

fig, ax = plt.subplots(figsize=(12, 10))

plot_lulc_map(
    pred,
    ax=ax,
    title='Bengaluru 2024 - Land Use/Land Cover',
    class_names=class_names
)

fig.savefig(
    'outputs/bangalore_2024_lulc_preview.png',
    dpi=200,
    bbox_inches='tight'
)

plt.close(fig)

print('Visual map saved to outputs/bangalore_2024_lulc_preview.png')
"

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Prediction:"
echo "  $OUTPUT"
echo ""
echo "Evaluation:"
echo "  outputs/bangalore_2024_segmentation_report.json"
echo ""
echo "Visual map:"
echo "  outputs/bangalore_2024_lulc_preview.png"