"""
Generate a `label.tif` automatically, with no manual digitizing, by pulling
ESA WorldCover (10 m, global, pre-classified land cover) for your AOI and
remapping its 11 classes onto this project's 7 classes.

This is a WEAK-LABEL shortcut, not a substitute for real ground truth: ESA
WorldCover is a single global product with its own error rate (~74-80%
overall accuracy depending on region -- see the ESA WorldCover validation
report), so a model trained only on these labels inherits that ceiling. It's
meant to get you a real, running end-to-end pipeline immediately; swap in
hand-digitized or field-verified labels later for a serious accuracy claim
in your report.

ESA WorldCover v200 (2021) raw class codes -> this project's 7 classes:
    10 Tree cover              -> 0 forest
    20 Shrubland                -> 6 grassland
    30 Grassland                -> 6 grassland
    40 Cropland                 -> 1 cropland
    50 Built-up                 -> 3 urban
    60 Bare / sparse vegetation -> 5 barren
    70 Snow and ice              -> 5 barren
    80 Permanent water bodies    -> 2 water
    90 Herbaceous wetland        -> 4 wetland
    95 Mangroves                 -> 4 wetland
    100 Moss and lichen           -> 5 barren

Usage
-----
    python scripts/build_weak_labels.py \\
        --aoi-geojson my_area.geojson \\
        --reference-band data/raw/scene_001/B02.tif \\
        --out data/raw/scene_001/label.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ESA WorldCover v200 raw code -> this project's class index (see mapping above)
WORLDCOVER_TO_PROJECT_CLASS = {
    10: 0,   # Tree cover -> forest
    20: 6,   # Shrubland -> grassland
    30: 6,   # Grassland -> grassland
    40: 1,   # Cropland -> cropland
    50: 3,   # Built-up -> urban
    60: 5,   # Bare / sparse vegetation -> barren
    70: 5,   # Snow and ice -> barren
    80: 2,   # Permanent water bodies -> water
    90: 4,   # Herbaceous wetland -> wetland
    95: 4,   # Mangroves -> wetland
    100: 5,  # Moss and lichen -> barren
}


def fetch_worldcover_remapped(aoi):
    """Load ESA WorldCover v200 (2021) for the AOI and remap its class codes
    onto this project's 0-6 class indices, returning an ee.Image."""
    import ee

    raw_codes = list(WORLDCOVER_TO_PROJECT_CLASS.keys())
    project_classes = list(WORLDCOVER_TO_PROJECT_CLASS.values())

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    remapped = worldcover.remap(raw_codes, project_classes, defaultValue=5)  # unseen codes -> barren
    return remapped.clip(aoi)


def get_reference_grid(reference_band_path: str) -> dict:
    """Read the CRS/transform/shape of an already-downloaded band GeoTIFF so
    the label raster comes out perfectly aligned with it."""
    import rasterio

    with rasterio.open(reference_band_path) as src:
        return {
            "crs": src.crs.to_string(),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
        }


def download_label_direct(remapped_image, aoi, ref_grid: dict, out_path: str):
    """Download the remapped WorldCover image via getDownloadURL, aligned to
    the reference band's grid (same scale/CRS), and save to `out_path`."""
    import requests

    scale = abs(ref_grid["transform"].a)   # pixel size in CRS units (metres for EPSG:4326-projected UTM, degrees otherwise)
    url = remapped_image.getDownloadURL({
        "region": aoi,
        "scale": 10,          # native WorldCover resolution
        "crs": ref_grid["crs"],
        "format": "GEO_TIFF",
    })
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(resp.content)


def resample_to_reference(label_path: str, ref_grid: dict, out_path: str):
    """WorldCover's own grid rarely lines up pixel-for-pixel with the
    Sentinel-2 band grid it was downloaded alongside; nearest-neighbour
    resample the label onto the exact reference grid so
    `preprocessing.loader.tile_scene` can slice both in lockstep."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(label_path) as src:
        label_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

    dst_data = np.zeros((ref_grid["height"], ref_grid["width"]), dtype=np.uint8)
    reproject(
        source=label_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_grid["transform"],
        dst_crs=ref_grid["crs"],
        resampling=Resampling.nearest,
    )

    meta = {
        "driver": "GTiff",
        "height": ref_grid["height"],
        "width": ref_grid["width"],
        "count": 1,
        "dtype": "uint8",
        "crs": ref_grid["crs"],
        "transform": ref_grid["transform"],
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(dst_data, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi-geojson", required=True)
    parser.add_argument("--reference-band", required=True, help="an already-downloaded band GeoTIFF (e.g. B02.tif) to align the label grid to")
    parser.add_argument("--out", required=True, help="output path, e.g. data/raw/scene_001/label.tif")
    args = parser.parse_args()

    import ee
    ee.Initialize(project="iva-project-508311")

    with open(args.aoi_geojson) as f:
        geojson = json.load(f)
    aoi = ee.Geometry(geojson["features"][0]["geometry"] if "features" in geojson else geojson)

    print("Fetching + remapping ESA WorldCover for AOI...")
    ref_grid = get_reference_grid(args.reference_band)
    remapped = fetch_worldcover_remapped(aoi)

    tmp_path = str(Path(args.out).with_suffix(".raw.tif"))
    download_label_direct(remapped, aoi, ref_grid, tmp_path)
    print(f"Downloaded raw WorldCover-derived label to {tmp_path}")

    print("Resampling onto the reference band's exact pixel grid...")
    resample_to_reference(tmp_path, ref_grid, args.out)
    Path(tmp_path).unlink(missing_ok=True)

    print(f"Wrote aligned label mask to {args.out}")
    print(
        "\nReminder: these are WEAK labels from ESA WorldCover, not "
        "ground-truth annotations -- good for getting the pipeline running "
        "end-to-end, but note this in your report's methodology/limitations "
        "if you use them for reported accuracy figures."
    )


if __name__ == "__main__":
    main()
