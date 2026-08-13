#!/usr/bin/env python3
"""Segment H&E WSIs and write TCGA-compatible patch coordinates.

One HDF5 of level-0 `coords` per slide (pixels are read later, during encoding,
so this stage makes no image files). Defaults reproduce MahmoodLab's TCGA
contract: 256 px patches at a true 0.5 um/px, no overlap, CLAM TCGA HSV mask.

    python -m ebv.preprocessing.run --limit 1          # smoke test
    python -m ebv.preprocessing.run --jobs 32          # whole cohort
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import h5py
import openslide

from ebv.paths import ROOT
from ebv.preprocessing import scale, segment, tile
from ebv.preprocessing.config import Config

DEFAULT_INPUT = ROOT / "00_new_HE"
DEFAULT_OUTPUT = ROOT / "01_preprocessed"
MIN_DOWNSAMPLE = 0.95  # below this the slide is too coarse to supply the target scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--target-mpp", type=float, default=0.5, help="Extracted patch scale, um/px.")
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--base-magnification", type=float, default=None,
                   help="Fallback objective for slides missing objective/MPP metadata.")
    p.add_argument("--no-pen-filter", dest="pen_filter", action="store_false",
                   help="Keep surgical-marker ink in the mask (removed by default).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="Slides to segment in parallel.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N slides.")
    return p.parse_args()


def process_slide(slide_path: Path, output_dir: Path, cfg: Config) -> dict:
    out_path = output_dir / "patches" / f"{slide_path.stem}.h5"
    if out_path.exists() and not cfg.overwrite:
        return {"slide": slide_path.name, "status": "skipped", "patches": "", "message": "exists"}

    with openslide.OpenSlide(str(slide_path)) as slide:
        # TCGA ships ~0.5 MB placeholder thumbnails; name them for what they are.
        if slide.level_count == 1 and max(slide.dimensions) < 5000:
            raise ValueError(f"not a whole-slide image: {slide.dimensions}, single pyramid level")

        target_ds, scale_source = scale.target_downsample(slide, cfg.target_mpp)
        if target_ds < MIN_DOWNSAMPLE:
            raise ValueError(f"slide too coarse for target scale: downsample {target_ds:.4f}")
        base_mag, mag_source = scale.objective_power(slide, cfg.base_magnification)
        mpp = scale.level0_mpp(slide)

        seg_level = segment.closest_level(slide, cfg.seg_downsample)
        rgb, contours, holes, seg_ds, pen_fraction = segment.segment_tissue(slide, seg_level, cfg)
        coords = tile.coordinates_for_contours(contours, holes, target_ds, cfg)

        tmp = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as h:
            dset = h.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
            attrs = {
                "slide_path": str(slide_path.resolve()), "slide_name": slide_path.name,
                "patch_size": cfg.patch_size, "patch_size_level0": round(cfg.patch_size * target_ds),
                "target_magnification": cfg.target_mag, "target_downsample": target_ds,
                "target_mpp": cfg.target_mpp, "level0_mpp": mpp if mpp else float("nan"),
                "patch_mpp": (mpp * target_ds) if mpp else float("nan"),
                "scale_source": scale_source, "base_magnification": base_mag,
                "magnification_source": mag_source, "overlap": cfg.overlap,
                "level0_width": slide.dimensions[0], "level0_height": slide.dimensions[1],
                "segmentation_level": seg_level, "segmentation_method": "CLAM_TCGA_HSV",
                "pen_filter": cfg.pen_filter, "pen_fraction_removed": pen_fraction,
                "sthresh": cfg.sthresh, "mthresh": cfg.mthresh, "close": cfg.close,
                "area_tissue": cfg.area_tissue, "area_hole": cfg.area_hole, "max_holes": cfg.max_holes,
            }
            for k, v in attrs.items():
                h.attrs[k] = v
                dset.attrs[k] = v
        tmp.replace(out_path)
        segment.save_qc(rgb, contours, holes, seg_ds, output_dir / "qc" / f"{slide_path.stem}.jpg",
                        segment.pen_mask(rgb, cfg) if cfg.pen_filter else None)
    return {"slide": slide_path.name, "status": "ok", "patches": len(coords), "message": ""}


def safe_process(slide_path: Path, output_dir: Path, cfg: Config) -> dict:
    """One slide, turning any failure into a manifest row so a cohort run survives it."""
    try:
        return process_slide(slide_path, output_dir, cfg)
    except Exception as exc:
        traceback.print_exc()
        return {"slide": slide_path.name, "status": "failed", "patches": "",
                "message": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    args = parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    slides = sorted(p for p in input_dir.rglob("*.svs") if p.is_file())
    if args.limit is not None:
        slides = slides[: args.limit]
    if not slides:
        print(f"ERROR: no .svs files under {input_dir}", file=sys.stderr)
        return 2

    (output_dir / "patches").mkdir(parents=True, exist_ok=True)
    (output_dir / "qc").mkdir(parents=True, exist_ok=True)
    cfg = replace(Config(), patch_size=args.patch_size, target_mpp=args.target_mpp,
                  overlap=args.overlap, base_magnification=args.base_magnification,
                  pen_filter=args.pen_filter, overwrite=args.overwrite)

    rows: list[dict] = []
    if args.jobs == 1:
        for i, path in enumerate(slides, 1):
            row = safe_process(path, output_dir, cfg)
            print(f"[{i}/{len(slides)}] {path.name}: {row['status']} {row['patches']} {row['message']}",
                  flush=True)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(safe_process, p, output_dir, cfg): p for p in slides}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                print(f"[{i}/{len(slides)}] {row['slide']}: {row['status']} {row['patches']} "
                      f"{row['message']}", flush=True)
                rows.append(row)
        rows.sort(key=lambda r: str(r["slide"]))

    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "status", "patches", "message"])
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(r["status"] == "failed" for r in rows)
    print(f"Finished {len(slides)} slides ({failures} failed). Manifest: {output_dir / 'manifest.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
