"""
Helper for pulling Sentinel-2 L2A scenes from Google Earth Engine and landing
per-band GeoTIFFs directly into the `data/raw/<scene_id>/` layout expected by
`scripts/preprocess_data.py`.

Requires a Google Earth Engine account with API access enabled
(https://signup.earthengine.google.com/) and `earthengine authenticate` run
once locally.

Two modes:

  --mode direct (default): calls `ee.Image.getDownloadURL` per band and
      streams the GeoTIFF straight to `data/raw/<scene_id>/`, with no manual
      steps -- suitable for full automation, but Earth Engine's synchronous
      download endpoint caps out at roughly a few hundred MB per request, so
      it's only reliable for small-to-moderate AOIs (a few km^2 at 10 m
      resolution). If a band exceeds the limit, that band is skipped with a
      warning and `--mode drive` should be used instead for that scene.

  --mode drive: kicks off asynchronous Earth Engine export tasks to Google
      Drive (no size limit), which you then download manually into
      `data/raw/<scene_id>/` -- unchanged from the original behaviour.

Usage
-----
    python scripts/download_data.py \\
        --aoi-geojson my_area.geojson \\
        --start 2024-01-01 --end 2024-03-31 \\
        --max-cloud-pct 15 \\
        --mode direct \\
        --out data/raw/scene_001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Earth Engine naming -> local file naming used throughout this project
BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05", "B6": "B06",
               "B7": "B07", "B8": "B08", "B8A": "B8A", "B11": "B11", "B12": "B12"}


def get_least_cloudy_image(aoi, start: str, end: str, max_cloud_pct: float):
    import ee

    def mask_clouds(image):
        scl = image.select("SCL")

        # Keep:
        # 4 = vegetation
        # 5 = bare soil
        # 6 = water
        # 7 = unclassified
        mask = (
            scl.eq(4)
            .Or(scl.eq(5))
            .Or(scl.eq(6))
            .Or(scl.eq(7))
        )

        return image.updateMask(mask)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_clouds)
    )

    # Combine all suitable Sentinel-2 acquisitions covering the AOI.
    # Median makes a cloud-masked composite and fills spatial gaps
    # between different Sentinel-2 granules.
    composite = collection.median().clip(aoi)

    return composite


def export_bands_to_drive(image, aoi, out_prefix: str, scale: int = 10):
    """Kicks off one Earth Engine export task per band to Google Drive. Tasks
    run asynchronously server-side; monitor progress at
    https://code.earthengine.google.com/tasks"""
    import ee

    tasks = []
    for band in BANDS:
        local_name = BAND_RENAME[band]
        task = ee.batch.Export.image.toDrive(
            image=image.select(band),
            description=f"{out_prefix}_{local_name}",
            folder="land_cover_pipeline_exports",
            fileNamePrefix=f"{out_prefix}_{local_name}",
            region=aoi,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
        )
        task.start()
        tasks.append(task)
    return tasks


def download_bands_direct(image, aoi, out_dir: Path, scale: int = 10) -> list[str]:
    """Stream each band straight to disk via `Image.getDownloadURL`, no Drive
    or manual steps required. Returns the list of local band names that
    failed to download (e.g. because the request exceeded EE's synchronous
    size limit) so the caller can warn / fall back to `--mode drive`."""
    import ee
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for band in BANDS:
        local_name = BAND_RENAME[band]
        dest = out_dir / f"{local_name}.tif"
        try:
            url = image.select(band).getDownloadURL({
                "region": aoi,
                "scale": scale,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  downloaded {dest} ({len(resp.content) / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - surface any EE/network error per band
            print(f"  WARNING: failed to download band {band} ({local_name}): {exc}")
            failed.append(local_name)

    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi-geojson", required=True, help="path to a GeoJSON polygon defining the area of interest")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-cloud-pct", type=float, default=15.0)
    parser.add_argument("--out", required=True, help="local scene directory (direct mode) / export file prefix (drive mode)")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--mode", choices=["direct", "drive"], default="direct")
    args = parser.parse_args()

    import ee
    ee.Initialize(project="iva-project-508311")

    with open(args.aoi_geojson) as f:
        geojson = json.load(f)
    aoi = ee.Geometry(geojson["features"][0]["geometry"] if "features" in geojson else geojson)

    image = get_least_cloudy_image(aoi, args.start, args.end, args.max_cloud_pct)
    if image is None:
        raise RuntimeError("No suitable image found for the given AOI/date range/cloud threshold.")

    if args.mode == "direct":
        out_dir = Path(args.out)
        print(f"Downloading {len(BANDS)} bands directly to {out_dir}/ ...")
        failed = download_bands_direct(image, aoi, out_dir, scale=args.scale)
        if failed:
            print(
                f"\n{len(failed)} band(s) failed ({', '.join(failed)}) -- likely too large for "
                "the synchronous download endpoint. Re-run with --mode drive for this scene, "
                "or shrink the AOI."
            )
            raise SystemExit(1)
        print(f"Done. All bands written to {out_dir}/")
    else:
        tasks = export_bands_to_drive(image, aoi, Path(args.out).name, scale=args.scale)
        print(
            f"Started {len(tasks)} export tasks to Google Drive folder "
            "'land_cover_pipeline_exports'. Monitor at "
            "https://code.earthengine.google.com/tasks, then download the "
            f"resulting GeoTIFFs into {args.out}/ with the local band names "
            f"({', '.join(BAND_RENAME.values())})."
        )


if __name__ == "__main__":
    main()
