# Automated Land Cover Classification & Change Detection Pipeline

Technical implementation for **CIA — Image and Video Analytics (CSEDS743E04)**,
CHRIST (Deemed to be University). Implements the hybrid framework described in
the accompanying report: multispectral semantic segmentation (U-Net /
DeepLabV3+ / SegFormer-B3), bitemporal Siamese change detection, DCT/learned
lossy compression, and Fourier-detrended NDVI anomaly detection — with DFT,
DWT and spectral vegetation indices as classical feature-extraction stages.

> **Note on compute**: this repository is delivered as complete, runnable
> code. No models have been trained or executed as part of delivering this —
> training deep nets requires a GPU and real satellite data, neither of which
> belong in this handoff. Follow the steps below on your own machine /
> Colab / university GPU cluster.

---

## 1. Project layout

```
land_cover_pipeline/
├── configs/
│   └── config.yaml                 # all hyperparameters, paths, band lists
├── data/
│   ├── raw/                        # per-scene GeoTIFF bands (you populate this)
│   └── processed/                  # tiled .npy training data (generated)
├── src/
│   ├── preprocessing/
│   │   ├── loader.py                # raster I/O, tiling, GeoTIFF write
│   │   ├── indices.py               # NDVI, EVI, NDWI, SAVI, BAI
│   │   └── transforms.py            # DFT, DWT (db4), block-DCT + quantization
│   ├── datasets/
│   │   ├── lulc_dataset.py          # segmentation PyTorch Dataset
│   │   └── change_dataset.py        # bitemporal change PyTorch Dataset
│   ├── models/
│   │   ├── unet.py                  # U-Net (from scratch)
│   │   ├── deeplabv3plus.py         # DeepLabV3+ (segmentation_models_pytorch)
│   │   ├── segformer.py             # SegFormer-B3 (HuggingFace transformers)
│   │   ├── siamese_change.py        # Siamese U-Net change detector
│   │   └── compression_autoencoder.py  # conv autoencoder / CompressAI hyperprior
│   ├── training/
│   │   ├── losses.py                 # Dice+Focal, Dice+BCE
│   │   ├── train_segmentation.py
│   │   ├── train_change_detection.py
│   │   └── train_compression.py
│   ├── inference/
│   │   ├── predict_segmentation.py   # sliding-window full-scene inference
│   │   ├── predict_change.py         # bitemporal change map inference
│   │   └── anomaly_detection.py      # Fourier-detrended Z-score alerts
│   ├── evaluation/
│   │   └── metrics.py                # OA, precision/recall/F1, mIoU, kappa, PSNR, SSIM, SAM
│   └── utils/
│       ├── config.py                 # YAML loading, seeding, device selection
│       └── visualization.py          # RGB / LULC / change / anomaly plotting
├── scripts/
│   ├── download_data.py              # optional Google Earth Engine export helper
│   ├── preprocess_data.py            # raw scenes -> tiled .npy training data
│   └── evaluate.py                   # final test-set metric report
└── requirements.txt
```

All classical-transform and index code (`src/preprocessing/`) has already been
unit-tested against synthetic arrays as part of building this repo (DFT
round-trip, DWT feature shapes, DCT round-trip + quantization PSNR scaling)
— see "What's already been verified" at the bottom of this file.

---

## 1a. One-shot automation (`scripts/run_pipeline.sh`)

If you just want to point this at an area and go, `scripts/run_pipeline.sh`
chains together everything below (env setup → GEE download → preprocessing →
training → inference → evaluation → a PNG preview) into one command:

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh \
    --aoi my_area.geojson \
    --start 2024-01-01 --end 2024-03-31 \
    --scene-id scene_001 \
    --auto-labels
```

Add `--auto-labels` and the script generates a `label.tif` for you
automatically (see "Getting a label.tif" below) so segmentation training runs
without any manual digitizing. Omit it and segmentation training is skipped.

**What it automates, and what it honestly can't:**
- Steps 1–4 (env setup, Earth Engine auth check, direct-download of Sentinel-2
  bands for your AOI via `getDownloadURL` — no manual Google Drive step —
  and preprocessing) are fully automatic.
- Step 5 always trains the **compression autoencoder**, since that's
  self-supervised (needs only imagery, no labels).
- Step 5 **skips segmentation training** unless you supply a `label.tif` —
  land-cover classification is a supervised task, and no download step can
  invent ground-truth class labels for your AOI on its own. Two ways to get
  one, see "Getting a label.tif" right below.
- Change-detection training is **not** run automatically (it needs a second
  date's scene plus a manually-built pairs CSV) — the script prints the two
  follow-up commands to run once you have that.
- Re-running is safe: it skips re-downloading a scene directory that's
  already populated (`--force` to override) and reuses an existing conda env
  (`--skip-env` to skip setup entirely).

Useful flags: `--conda-env <name>` (default `lulc`), `--max-cloud-pct <pct>`,
`--config configs/config.yaml`, `--skip-download`. Run
`./scripts/run_pipeline.sh --help` for the full list.

**Direct-download size limit**: Earth Engine's synchronous
`getDownloadURL` endpoint used by the script caps out around a few hundred MB
per band request, so it's reliable for small-to-moderate AOIs (roughly a few
km² at 10 m resolution). For larger areas, run
`python scripts/download_data.py --mode drive ...` instead (see Section 3) —
it exports asynchronously to Google Drive with no size cap, you just download
the results into `data/raw/<scene_id>/` yourself before running the rest of
the pipeline manually (Sections 4–7 below).

### Getting a `label.tif`

This is the one input the pipeline can't fetch from GEE alongside your
imagery: a second GeoTIFF, same grid as your downloaded bands, where every
pixel is an integer `0`–`6` naming its class (`0`=forest, `1`=cropland,
`2`=water, `3`=urban, `4`=wetland, `5`=barren, `6`=grassland — see
`configs/config.yaml -> data.class_names`). Without it there's nothing for
the segmentation model to learn from.

**Option A — automatic weak labels (fastest, no GIS work):**
```bash
python scripts/build_weak_labels.py \
    --aoi-geojson my_area.geojson \
    --reference-band data/raw/scene_001/B02.tif \
    --out data/raw/scene_001/label.tif
```
This pulls **ESA WorldCover** (a free, pre-classified, 10 m global land-cover
map already on Earth Engine) for your AOI, remaps its 11 classes onto this
project's 7, and aligns it pixel-for-pixel to your downloaded bands. It's
built into `run_pipeline.sh` too — just add `--auto-labels`. Caveat: it's a
*weak* label source (WorldCover's own accuracy is roughly 74–80% depending on
region), fine for getting a real pipeline running end-to-end, but say so
explicitly in your report's methodology if you report accuracy numbers
trained against it.

**Option B — hand-digitized labels (slower, more accurate):**
Open your downloaded scene as a raster layer in QGIS, use the polygon tool
to trace training areas for each class by eye (zoom in, compare against a
basemap), assign each polygon's attribute table a class index `0`–`6`, then
`Raster → Conversion → Rasterize (Vector to Raster)` using one of your
downloaded bands as the reference extent/resolution, burning the class
attribute into a single-band output. Save that as `label.tif` in the same
scene folder as your bands.

---

## 2. Environment setup

```bash
# 1. Create an isolated environment (conda recommended — GDAL is easier via conda-forge)
conda create -n lulc python=3.10 -y
conda activate lulc
conda install -c conda-forge gdal=3.7 rasterio=1.3 -y

# 2. Install the remaining Python dependencies
pip install -r requirements.txt

# 3. (Optional, for CUDA) install a GPU-matched torch build instead of the
#    default from requirements.txt, e.g. for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify the install:

```bash
python -c "import rasterio, torch, transformers, segmentation_models_pytorch, compressai, pywt; print('OK')"
```

---

## 3. Get satellite imagery

You need Sentinel-2 L2A scenes (10 bands: B02, B03, B04, B05, B06, B07, B08,
B8A, B11, B12) for your study area, each band as its own GeoTIFF, laid out as:

```
data/raw/<scene_id>/B02.tif
data/raw/<scene_id>/B03.tif
...
data/raw/<scene_id>/B12.tif
data/raw/<scene_id>/label.tif      # only needed for supervised segmentation training
```

Three ways to get this:

**A. Google Earth Engine (recommended, free, matches the report's data source)**
```bash
earthengine authenticate     # one-time
python scripts/download_data.py \
    --aoi-geojson my_area.geojson \
    --start 2024-01-01 --end 2024-03-31 \
    --max-cloud-pct 15 \
    --out data/raw/scene_001
```
This exports one GeoTIFF per band to a Google Drive folder
(`land_cover_pipeline_exports`); download them locally into
`data/raw/scene_001/` with the printed filenames.

**B. Copernicus Data Space Ecosystem** — manually download L2A products from
https://dataspace.copernicus.eu, then split the `.SAFE` product's `.jp2`
band files into the `data/raw/<scene_id>/B0X.tif` layout (use
`gdal_translate` to convert `.jp2` → `.tif`).

**C. Benchmark datasets** — for a first end-to-end test without acquiring
your own imagery, use a public labelled dataset such as **OpenEarthMap** or
**DeepGlobe Land Cover** (both referenced in the report's literature
review/benchmark table) and reformat their tiles into the same folder layout.

For **labels**, digitize training polygons in QGIS against a high-resolution
basemap (or reuse an existing LULC product like ESA WorldCover as weak
labels), then rasterize to `label.tif` with class indices `0..6` matching
`configs/config.yaml -> data.class_names`.

For **change detection**, you need two scenes of the same area at different
dates plus a binary change-mask GeoTIFF; list these in a CSV:

```csv
scene_t1,scene_t2,label
data/raw/scene_2023,data/raw/scene_2024,data/raw/labels/change_2023_2024.tif
```

---

## 4. Preprocess: raw scenes → training tiles

```bash
# Segmentation tiles (spectral bands + NDVI/EVI/NDWI/SAVI/BAI, 256x256, 70/15/15 split)
python scripts/preprocess_data.py --task segmentation \
    --config configs/config.yaml \
    --raw-dir data/raw \
    --out data/processed/segmentation

# Change-detection tiles
python scripts/preprocess_data.py --task change \
    --config configs/config.yaml \
    --pairs data/raw/change_pairs.csv \
    --out data/processed/change

# Compression tiles (full-band reflectance, no labels needed)
python scripts/preprocess_data.py --task compression \
    --config configs/config.yaml \
    --raw-dir data/raw \
    --out data/processed/compression
```

Each command writes `train/`, `val/`, `test/` subfolders of `.npy` tiles
under the given `--out` directory, per the split ratios in
`configs/config.yaml -> data`.

---

## 5. Train

All three training scripts read every hyperparameter (architecture choice,
batch size, epochs, learning rate, etc.) from `configs/config.yaml` — edit
that file rather than passing dozens of CLI flags.

```bash
# Segmentation (architecture selected via config: unet | deeplabv3plus | segformer)
python -m src.training.train_segmentation --config configs/config.yaml

# Change detection (Siamese U-Net)
python -m src.training.train_change_detection --config configs/config.yaml

# Compression autoencoder (conv_autoencoder | hyperprior via config)
python -m src.training.train_compression --config configs/config.yaml
```

Each script:
- logs to TensorBoard under `logs/<task>/` (`tensorboard --logdir logs`)
- checkpoints `best_model.pt` (best validation metric) and `last_model.pt`
  under `checkpoints/<task>/`
- prints per-epoch metrics (mIoU/OA for segmentation, F1/IoU for change
  detection, PSNR/SSIM/SAM for compression)

**Compute expectations**: SegFormer-B3 / DeepLabV3+-ResNet50 at 256×256,
batch size 8, need a GPU with ≥12 GB VRAM for comfortable training; U-Net is
lighter. Google Colab Pro+ (A100) matches what the report's technology table
specifies, or use a university HPC allocation.

---

## 6. Run inference on new scenes

```bash
# Full-scene LULC map (sliding-window inference, overlap-averaged)
python -m src.inference.predict_segmentation \
    --config configs/config.yaml \
    --checkpoint checkpoints/segmentation/best_model.pt \
    --scene data/raw/scene_new \
    --out outputs/scene_new_lulc.tif

# Bitemporal change map
python -m src.inference.predict_change \
    --config configs/config.yaml \
    --checkpoint checkpoints/change_detection/best_model.pt \
    --scene-t1 data/raw/scene_2023 \
    --scene-t2 data/raw/scene_2024 \
    --out outputs/change_2023_2024.tif
```

For **anomaly detection**, build a per-pixel NDVI time-series stack `(T, H,
W)` across several historical dates plus their day-of-year timestamps, then
call the module directly (it's a pure-numpy function, no trained model
needed):

```python
from src.inference.anomaly_detection import detect_anomalies
import numpy as np

ndvi_stack = np.load("data/processed/ndvi_timeseries_scene001.npy")   # (T,H,W)
doy = np.load("data/processed/timestamps_doy_scene001.npy")           # (T,)

result = detect_anomalies(ndvi_stack, doy, harmonics=3, z_threshold=2.5, min_area_px=25)
print("Flagged anomaly area fraction:", result["alert_area_fraction"])
```

---

## 7. Evaluate on the held-out test split

```bash
python scripts/evaluate.py --task segmentation \
    --config configs/config.yaml \
    --checkpoint checkpoints/segmentation/best_model.pt \
    --data-root data/processed/segmentation \
    --out outputs/segmentation_test_report.json

python scripts/evaluate.py --task change \
    --config configs/config.yaml \
    --checkpoint checkpoints/change_detection/best_model.pt \
    --data-root data/processed/change \
    --out outputs/change_test_report.json
```

Reports include overall accuracy, per-class precision/recall/F1, mean IoU,
Cohen's kappa (segmentation) and precision/recall/F1/IoU (change detection) —
matching the metrics table in Section 2.2.6 of the report. Compression
fidelity (PSNR/SSIM/SAM) is reported automatically each epoch by
`train_compression.py`; a standalone `evaluate.py --task compression` can be
added the same way if you need a frozen final-checkpoint report.

---

## 8. Visualize results

```python
import matplotlib.pyplot as plt
from src.utils.visualization import plot_lulc_map, plot_change_map
import rasterio

with rasterio.open("outputs/scene_new_lulc.tif") as src:
    lulc = src.read(1)

fig, ax = plt.subplots(figsize=(8, 8))
plot_lulc_map(lulc, class_names=[
    "forest", "cropland", "water", "urban", "wetland", "barren", "grassland"
], ax=ax)
plt.savefig("outputs/scene_new_lulc_preview.png", dpi=150, bbox_inches="tight")
```

---

## 9. What's already been verified in this handoff

No model has been trained (per your request), but the parts of the codebase
that don't require a GPU or real satellite imagery were exercised against
synthetic arrays while building this repo, to catch bugs before you invest
compute time:

- **Indices** (`src/preprocessing/indices.py`): NDVI/EVI/NDWI/SAVI/BAI all
  produce finite, correctly-shaped output on random band data.
- **DFT** (`transforms.py`): forward/inverse round-trips to numerical
  precision (~3e-7 max error); band-pass filtering runs end-to-end.
- **DWT** (`transforms.py`, db4): multi-level feature maps and bitemporal
  change scores produce correctly-shaped, finite output.
- **DCT + quantization** (`transforms.py`): block DCT round-trips to
  numerical precision; quantization now correctly scales its table for
  `[0,1]`-normalised reflectance via a `data_range` parameter (the initial
  version used a raw 0–255 JPEG table and silently zeroed almost every
  coefficient on reflectance-scale data — verified fixed: PSNR now rises
  monotonically with `quality`, e.g. ~13 dB at quality=10 to ~36 dB at
  quality=90 on synthetic data).
- **All 20+ Python files**: syntax-checked with `py_compile`.

**Not yet exercised** (needs your GPU + real data to validate): the PyTorch
model forward passes (U-Net/DeepLabV3+/SegFormer/Siamese U-Net/autoencoder),
the training loops, and full-scene sliding-window inference. The shape
assertions at the bottom of each `src/models/*.py` file (`if __name__ ==
"__main__":`) are ready to run as a first smoke test the moment you have
torch installed with a GPU:

```bash
python src/models/unet.py
python src/models/siamese_change.py
python src/models/compression_autoencoder.py
# deeplabv3plus.py and segformer.py additionally download pretrained
# ImageNet/MiT-B3 weights on first run
python src/models/deeplabv3plus.py
python src/models/segformer.py
```

---

## 10. Mapping back to the report

| Report section | Implementation |
|---|---|
| 2.2.1 Automated LULC classification | `src/models/{unet,deeplabv3plus,segformer}.py`, `src/training/train_segmentation.py` |
| 2.2.2 Temporal change detection | `src/models/siamese_change.py`, `src/training/train_change_detection.py`, DWT differencing in `transforms.py` |
| 2.2.3 Image compression | `src/models/compression_autoencoder.py` (ConvAutoencoder + CompressAI hyperprior), block-DCT quantization in `transforms.py` |
| 2.2.4 Disaster/anomaly detection | `src/inference/anomaly_detection.py` (Fourier-detrended Z-score) |
| 2.2.5 Hybrid classical + deep learning | DFT/DWT/indices concatenated as input channels (`indices.py`, `transforms.py`) feeding the segmentation/change models |
| 2.2.6 Performance evaluation | `src/evaluation/metrics.py`, `scripts/evaluate.py` |
| 4.4 Technology summary table | `requirements.txt`, `configs/config.yaml` |
