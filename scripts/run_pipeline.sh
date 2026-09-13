#!/usr/bin/env bash
#
# run_pipeline.sh -- one-shot automation for the land-cover pipeline:
#   1. Environment setup (conda env + pip requirements)
#   2. Earth Engine authentication check
#   3. Data download for an AOI (direct GEE download, no manual steps)
#   4. Preprocessing (indices, tiling) for whichever tasks have data
#   5. Training: compression autoencoder always (self-supervised, no labels
#      needed); segmentation only if you supply a label.tif; change detection
#      only if you supply a pairs CSV
#   6. Inference + evaluation on what was actually trained
#   7. A PNG preview of the results
#
# HONEST LIMITATION: land-cover segmentation and change-detection are
# *supervised* tasks. No download step can invent ground-truth labels for
# you -- this script will happily train the compression model end-to-end
# on downloaded imagery alone, but will SKIP segmentation/change training
# (with a clear message) unless you either pass --label path/to/label.tif
# you created yourself (e.g. by digitizing polygons in QGIS, per the
# README), or pass --auto-labels to generate weak labels automatically from
# ESA WorldCover (fast, free, no manual work -- but lower quality than real
# ground truth; see scripts/build_weak_labels.py for details/caveats).
#
# Usage
# -----
#   ./scripts/run_pipeline.sh \
#       --aoi my_area.geojson \
#       --start 2024-01-01 --end 2024-03-31 \
#       --scene-id scene_001 \
#       [--label path/to/label.tif | --auto-labels] \
#       [--skip-env] [--skip-download] [--conda-env lulc]
#
# Every step is idempotent-ish: re-running skips work whose output already
# exists (e.g. won't re-download a scene directory that's already populated).
# Use --force to override.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / argument parsing
# ---------------------------------------------------------------------------
CONDA_ENV="lulc"
AOI_GEOJSON=""
START_DATE=""
END_DATE=""
MAX_CLOUD_PCT="15"
SCENE_ID="scene_001"
LABEL_TIF=""
AUTO_LABELS="false"
SKIP_ENV="false"
SKIP_DOWNLOAD="false"
FORCE="false"
CONFIG="configs/config.yaml"

usage() {
    # Print only the leading header comment block (stops at the first
    # non-comment line), not every "# ---- Step N ----" divider later in
    # the file.
    awk 'NR==1{next} /^#/{print; next} {exit}' "$0" | sed -e 's/^#//' -e 's/^ //'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aoi) AOI_GEOJSON="$2"; shift 2 ;;
        --start) START_DATE="$2"; shift 2 ;;
        --end) END_DATE="$2"; shift 2 ;;
        --max-cloud-pct) MAX_CLOUD_PCT="$2"; shift 2 ;;
        --scene-id) SCENE_ID="$2"; shift 2 ;;
        --label) LABEL_TIF="$2"; shift 2 ;;
        --auto-labels) AUTO_LABELS="true"; shift ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --skip-env) SKIP_ENV="true"; shift ;;
        --skip-download) SKIP_DOWNLOAD="true"; shift ;;
        --force) FORCE="true"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$AOI_GEOJSON" || -z "$START_DATE" || -z "$END_DATE" ]]; then
    echo "ERROR: --aoi, --start and --end are required."
    usage
fi

RAW_DIR="data/raw/${SCENE_ID}"
SEG_PROCESSED="data/processed/segmentation"
COMP_PROCESSED="data/processed/compression"

log() { echo -e "\n\033[1;36m==> $1\033[0m"; }
# Accepts one or more string arguments and prints each on its own [WARN] line
# (bare `echo "$1"` would silently drop every argument after the first).
warn() { for line in "$@"; do echo -e "\033[1;33m[WARN] ${line}\033[0m"; done; }

# ---------------------------------------------------------------------------
# Step 1: Environment setup
# ---------------------------------------------------------------------------
if [[ "$SKIP_ENV" == "false" ]]; then
    log "Step 1/7: Environment setup"

    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda not found. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    if conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
        echo "Conda env '${CONDA_ENV}' already exists, reusing it."
    else
        echo "Creating conda env '${CONDA_ENV}' (python 3.10 + GDAL/rasterio via conda-forge)..."
        conda create -n "${CONDA_ENV}" python=3.10 -y
        conda install -n "${CONDA_ENV}" -c conda-forge gdal=3.7 rasterio=1.3 -y
    fi

    # Run the rest of this script's pip/python steps inside the env
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"

    echo "Installing Python dependencies from requirements.txt..."
    pip install -q -r requirements.txt
    pip install -q earthengine-api requests

    echo "Environment ready: $(python --version), $(which python)"
else
    log "Step 1/7: Environment setup [SKIPPED --skip-env]"
    if command -v conda &> /dev/null && conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Earth Engine authentication check
# ---------------------------------------------------------------------------
log "Step 2/7: Earth Engine authentication check"
python - <<'EOF' || { echo "Earth Engine not authenticated. Run 'earthengine authenticate' once, then re-run this script."; exit 1; }
import ee
ee.Initialize(project="iva-project-508311")
print("Earth Engine OK.")
EOF

# ---------------------------------------------------------------------------
# Step 3: Data download
# ---------------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    log "Step 3/7: Downloading Sentinel-2 bands for AOI -> ${RAW_DIR}"
    if [[ -d "$RAW_DIR" && "$(ls -A "$RAW_DIR" 2>/dev/null)" && "$FORCE" == "false" ]]; then
        echo "${RAW_DIR} already populated, skipping download (use --force to re-download)."
    else
        python scripts/download_data.py \
            --aoi-geojson "$AOI_GEOJSON" \
            --start "$START_DATE" --end "$END_DATE" \
            --max-cloud-pct "$MAX_CLOUD_PCT" \
            --mode direct \
            --out "$RAW_DIR"
    fi
else
    log "Step 3/7: Data download [SKIPPED --skip-download]"
fi

if [[ ! -d "$RAW_DIR" || -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: ${RAW_DIR} is empty -- nothing to preprocess. Check the download step above."
    exit 1
fi

# Get a label mask: either copy a user-supplied one, or generate weak labels
# from ESA WorldCover automatically.
if [[ -n "$LABEL_TIF" ]]; then
    cp "$LABEL_TIF" "${RAW_DIR}/label.tif"
    echo "Copied label mask to ${RAW_DIR}/label.tif"
elif [[ "$AUTO_LABELS" == "true" ]]; then
    if [[ -f "${RAW_DIR}/label.tif" && "$FORCE" == "false" ]]; then
        echo "${RAW_DIR}/label.tif already exists, skipping auto-label generation (use --force to regenerate)."
    else
        log "Generating weak labels from ESA WorldCover (--auto-labels)"
        python scripts/build_weak_labels.py \
            --aoi-geojson "$AOI_GEOJSON" \
            --reference-band "${RAW_DIR}/B02.tif" \
            --out "${RAW_DIR}/label.tif"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Preprocessing
# ---------------------------------------------------------------------------
log "Step 4/7: Preprocessing"

echo "-> Compression tiles (self-supervised, always runs)"
python scripts/preprocess_data.py --task compression \
    --config "$CONFIG" --raw-dir data/raw --out "$COMP_PROCESSED"

HAVE_LABELS="false"
if [[ -f "${RAW_DIR}/label.tif" ]]; then
    HAVE_LABELS="true"
    echo "-> Segmentation tiles (label.tif found)"
    python scripts/preprocess_data.py --task segmentation \
        --config "$CONFIG" --raw-dir data/raw --out "$SEG_PROCESSED"
else
    warn "No ${RAW_DIR}/label.tif found -- skipping segmentation preprocessing/training." \
         "Pass --label path/to/label.tif for your own labels, or --auto-labels to" \
         "generate weak labels from ESA WorldCover automatically (see README section 3)."
fi

# ---------------------------------------------------------------------------
# Step 5: Training
# ---------------------------------------------------------------------------
log "Step 5/7: Training"

echo "-> Compression autoencoder"
python -m src.training.train_compression --config "$CONFIG" --data-root "$COMP_PROCESSED"

if [[ "$HAVE_LABELS" == "true" ]]; then
    echo "-> Segmentation model"
    python -m src.training.train_segmentation --config "$CONFIG" --data-root "$SEG_PROCESSED"
else
    warn "Segmentation training skipped (no labels)."
fi

warn "Change-detection training skipped by default -- it needs a second scene +" \
     "a pairs CSV (data/raw/change_pairs.csv). Run manually once you have that:" \
     "  python scripts/preprocess_data.py --task change --pairs data/raw/change_pairs.csv --out data/processed/change" \
     "  python -m src.training.train_change_detection --config ${CONFIG}"

# ---------------------------------------------------------------------------
# Step 6: Inference + evaluation
# ---------------------------------------------------------------------------
log "Step 6/7: Inference + evaluation"
mkdir -p outputs

if [[ "$HAVE_LABELS" == "true" ]]; then
    python -m src.inference.predict_segmentation \
        --config "$CONFIG" \
        --checkpoint checkpoints/segmentation/best_model.pt \
        --scene "$RAW_DIR" \
        --out "outputs/${SCENE_ID}_lulc.tif"

    python scripts/evaluate.py --task segmentation \
        --config "$CONFIG" \
        --checkpoint checkpoints/segmentation/best_model.pt \
        --data-root "$SEG_PROCESSED" \
        --out "outputs/${SCENE_ID}_segmentation_report.json"
else
    warn "Skipping segmentation inference/evaluation (no trained model)."
fi

# ---------------------------------------------------------------------------
# Step 7: Visualization
# ---------------------------------------------------------------------------
log "Step 7/7: Visualization"
if [[ "$HAVE_LABELS" == "true" && -f "outputs/${SCENE_ID}_lulc.tif" ]]; then
    python - <<EOF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from src.utils.visualization import plot_lulc_map
from src.utils.config import load_config

cfg = load_config("${CONFIG}")
with rasterio.open("outputs/${SCENE_ID}_lulc.tif") as src:
    lulc = src.read(1)

fig, ax = plt.subplots(figsize=(8, 8))
plot_lulc_map(lulc, class_names=cfg["data"]["class_names"], ax=ax)
plt.savefig("outputs/${SCENE_ID}_lulc_preview.png", dpi=150, bbox_inches="tight")
print("Wrote outputs/${SCENE_ID}_lulc_preview.png")
EOF
else
    warn "No LULC map to visualize (segmentation stage was skipped)."
fi

log "Pipeline complete."
echo "Outputs directory:"
ls -la outputs/ 2>/dev/null || true
echo "Checkpoints directory:"
find checkpoints -type f 2>/dev/null || true
