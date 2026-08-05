#!/usr/bin/env python3
"""Segment H&E WSIs and write UNI2/TCGA-compatible patch coordinates.

The defaults reproduce MahmoodLab's published TCGA patching contract:
256 px patches at 20x, no overlap, and the CLAM TCGA HSV segmentation preset.
Only coordinates are stored; pixels are read directly from the WSI during UNI2
inference so this stage does not create millions of image files.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "00_new_HE"
DEFAULT_OUTPUT = ROOT / "01_preprocessed"
SVS_SUFFIXES = {".svs"}


@dataclass(frozen=True)
class Config:
    patch_size: int
    target_mag: float
    overlap: int
    seg_downsample: float
    sthresh: int
    mthresh: int
    close: int
    use_otsu: bool
    area_tissue: float
    area_hole: float
    max_holes: int
    base_magnification: float | None
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment .svs slides and save level-0 tissue patch coordinates."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--target-mag", type=float, default=20.0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument(
        "--base-magnification",
        type=float,
        default=None,
        help="Fallback objective power for slides missing objective/MPP metadata.",
    )
    parser.add_argument("--seg-downsample", type=float, default=64.0)
    parser.add_argument("--sthresh", type=int, default=8)
    parser.add_argument("--mthresh", type=int, default=7)
    parser.add_argument("--close", type=int, default=4)
    parser.add_argument("--use-otsu", action="store_true")
    parser.add_argument("--area-tissue", type=float, default=16.0)
    parser.add_argument("--area-hole", type=float, default=4.0)
    parser.add_argument("--max-holes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N slides (for a smoke test)."
    )
    args = parser.parse_args()
    if args.patch_size <= 0 or args.target_mag <= 0:
        parser.error("--patch-size and --target-mag must be positive")
    if not 0 <= args.overlap < args.patch_size:
        parser.error("--overlap must be >= 0 and smaller than --patch-size")
    if args.mthresh <= 0 or args.mthresh % 2 == 0:
        parser.error("--mthresh must be a positive odd integer")
    return args


def objective_power(slide: openslide.OpenSlide, fallback: float | None) -> tuple[float, str]:
    props = slide.properties
    raw_objective = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    objective = float(raw_objective) if raw_objective else None
    raw_mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    raw_mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
    mpp_x = float(raw_mpp_x) if raw_mpp_x else None
    mpp_y = float(raw_mpp_y) if raw_mpp_y else None

    if mpp_x is not None and mpp_x > 0:
        if mpp_y is not None and mpp_y > 0 and abs(mpp_x - mpp_y) / mpp_x > 0.05:
            raise ValueError(f"anisotropic pixel spacing is unsupported: MPP_X={mpp_x}, MPP_Y={mpp_y}")
        # Prefer measured physical resolution: 0.25 um/px ~= 40x and 0.50 ~= 20x.
        mpp_magnification = 10.0 / mpp_x
        if objective is not None and abs(objective - mpp_magnification) / objective > 0.15:
            warnings.warn(
                f"objective power ({objective:g}x) and MPP ({mpp_x:g} um/px, "
                f"~{mpp_magnification:g}x) disagree; using MPP",
                stacklevel=2,
            )
        return mpp_magnification, "openslide.mpp-x"

    if objective is not None:
        return objective, "openslide.objective-power"

    if fallback is not None:
        return fallback, "command-line fallback"
    raise ValueError(
        "slide has neither objective-power nor MPP metadata; pass --base-magnification"
    )


def level_downsamples_xy(slide: openslide.OpenSlide) -> list[tuple[float, float]]:
    w0, h0 = slide.level_dimensions[0]
    return [(w0 / w, h0 / h) for w, h in slide.level_dimensions]


def closest_level(slide: openslide.OpenSlide, wanted: float) -> int:
    downsamples = level_downsamples_xy(slide)
    return min(
        range(len(downsamples)),
        key=lambda idx: abs(math.log(max(np.mean(downsamples[idx]), 1e-9) / wanted)),
    )


def segment_tissue(
    slide: openslide.OpenSlide, level: int, cfg: Config
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]], tuple[float, float]]:
    rgb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    saturation = cv2.medianBlur(saturation, cfg.mthresh)
    threshold_type = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if cfg.use_otsu else 0)
    threshold = 0 if cfg.use_otsu else cfg.sthresh
    _, mask = cv2.threshold(saturation, threshold, 255, threshold_type)
    if cfg.close > 0:
        kernel = np.ones((cfg.close, cfg.close), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    found = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return rgb, [], [], level_downsamples_xy(slide)[level]

    hierarchy = hierarchy[0]
    ds_x, ds_y = level_downsamples_xy(slide)[level]
    scaled_ref_area = int(512**2 / (ds_x * ds_y))
    min_tissue = cfg.area_tissue * scaled_ref_area
    min_hole = cfg.area_hole * scaled_ref_area

    tissue: list[np.ndarray] = []
    holes_by_tissue: list[list[np.ndarray]] = []
    for idx in np.flatnonzero(hierarchy[:, 3] == -1):
        child_ids: list[int] = []
        child = hierarchy[idx, 2]
        while child != -1:
            child_ids.append(int(child))
            child = hierarchy[child, 0]
        area = cv2.contourArea(contours[idx]) - sum(cv2.contourArea(contours[j]) for j in child_ids)
        if area <= min_tissue:
            continue
        kept_holes = sorted(
            (contours[j] for j in child_ids if cv2.contourArea(contours[j]) > min_hole),
            key=cv2.contourArea,
            reverse=True,
        )[: cfg.max_holes]
        scale = np.array([[[ds_x, ds_y]]], dtype=np.float64)
        tissue.append(np.rint(contours[idx] * scale).astype(np.int32))
        holes_by_tissue.append([np.rint(h * scale).astype(np.int32) for h in kept_holes])
    return rgb, tissue, holes_by_tissue, (ds_x, ds_y)


def point_inside(contour: np.ndarray, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(contour, point, False) >= 0


def coordinates_for_contours(
    contours: list[np.ndarray],
    holes_by_tissue: list[list[np.ndarray]],
    level0_size: tuple[int, int],
    target_downsample: float,
    cfg: Config,
) -> np.ndarray:
    footprint = max(1, int(round(cfg.patch_size * target_downsample)))
    step = max(1, int(round((cfg.patch_size - cfg.overlap) * target_downsample)))
    coords: list[tuple[int, int]] = []

    for contour, holes in zip(contours, holes_by_tissue):
        start_x, start_y, width, height = cv2.boundingRect(contour)
        for x in range(start_x, start_x + width, step):
            for y in range(start_y, start_y + height, step):
                # CLAM's TCGA preset uses the permissive four-point contour test.
                quarter = footprint / 4.0
                center_x, center_y = x + footprint / 2.0, y + footprint / 2.0
                probes = (
                    (center_x - quarter, center_y - quarter),
                    (center_x + quarter, center_y + quarter),
                    (center_x + quarter, center_y - quarter),
                    (center_x - quarter, center_y + quarter),
                )
                if not any(point_inside(contour, probe) for probe in probes):
                    continue
                if any(point_inside(hole, (center_x, center_y)) for hole in holes):
                    continue
                # CLAM's use_padding=True permits boundary tiles; OpenSlide pads them.
                coords.append((x, y))

    if not coords:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def save_qc(
    rgb: np.ndarray,
    contours: list[np.ndarray],
    holes: list[list[np.ndarray]],
    seg_ds: tuple[float, float],
    path: Path,
) -> None:
    canvas = rgb.copy()
    scale = np.array([[[1.0 / seg_ds[0], 1.0 / seg_ds[1]]]])
    small_contours = [np.rint(c * scale).astype(np.int32) for c in contours]
    small_holes = [np.rint(h * scale).astype(np.int32) for group in holes for h in group]
    cv2.drawContours(canvas, small_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, small_holes, -1, (255, 0, 0), 2)
    Image.fromarray(canvas).save(path, quality=90)


def process_slide(slide_path: Path, output_dir: Path, cfg: Config) -> dict[str, object]:
    patch_dir = output_dir / "patches"
    qc_dir = output_dir / "qc"
    out_path = patch_dir / f"{slide_path.stem}.h5"
    if out_path.exists() and not cfg.overwrite:
        return {"slide": slide_path.name, "status": "skipped", "patches": "", "message": "exists"}

    with openslide.OpenSlide(str(slide_path)) as slide:
        base_mag, mag_source = objective_power(slide, cfg.base_magnification)
        if base_mag + 1e-6 < cfg.target_mag:
            raise ValueError(f"base magnification {base_mag:g}x is below requested {cfg.target_mag:g}x")
        target_ds = base_mag / cfg.target_mag
        seg_level = closest_level(slide, cfg.seg_downsample)
        rgb, contours, holes, seg_ds = segment_tissue(slide, seg_level, cfg)
        coords = coordinates_for_contours(contours, holes, slide.dimensions, target_ds, cfg)

        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as handle:
            dataset = handle.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
            metadata = {
                "slide_path": str(slide_path.resolve()),
                "slide_name": slide_path.name,
                "patch_size": cfg.patch_size,
                "patch_level": -1,
                "target_magnification": cfg.target_mag,
                "target_downsample": target_ds,
                "base_magnification": base_mag,
                "magnification_source": mag_source,
                "overlap": cfg.overlap,
                "level0_width": slide.dimensions[0],
                "level0_height": slide.dimensions[1],
                "segmentation_level": seg_level,
                "segmentation_downsample_x": seg_ds[0],
                "segmentation_downsample_y": seg_ds[1],
                "segmentation_method": "CLAM_TCGA_HSV",
                "sthresh": cfg.sthresh,
                "mthresh": cfg.mthresh,
                "close": cfg.close,
                "use_otsu": cfg.use_otsu,
                "area_tissue": cfg.area_tissue,
                "area_hole": cfg.area_hole,
                "max_holes": cfg.max_holes,
            }
            for key, value in metadata.items():
                handle.attrs[key] = value
                dataset.attrs[key] = value
        tmp_path.replace(out_path)
        save_qc(rgb, contours, holes, seg_ds, qc_dir / f"{slide_path.stem}.jpg")
    return {"slide": slide_path.name, "status": "ok", "patches": len(coords), "message": ""}


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    slides = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SVS_SUFFIXES)
    if args.limit is not None:
        slides = slides[: args.limit]
    if not slides:
        print(f"ERROR: no .svs files found under {input_dir}", file=sys.stderr)
        return 2
    stems = [p.stem for p in slides]
    if len(stems) != len(set(stems)):
        print("ERROR: duplicate slide stems found; output HDF5 names would collide", file=sys.stderr)
        return 2

    (output_dir / "patches").mkdir(parents=True, exist_ok=True)
    (output_dir / "qc").mkdir(parents=True, exist_ok=True)
    cfg = Config(
        patch_size=args.patch_size,
        target_mag=args.target_mag,
        overlap=args.overlap,
        seg_downsample=args.seg_downsample,
        sthresh=args.sthresh,
        mthresh=args.mthresh,
        close=args.close,
        use_otsu=args.use_otsu,
        area_tissue=args.area_tissue,
        area_hole=args.area_hole,
        max_holes=args.max_holes,
        base_magnification=args.base_magnification,
        overwrite=args.overwrite,
    )

    rows: list[dict[str, object]] = []
    failures = 0
    for index, slide_path in enumerate(slides, start=1):
        print(f"[{index}/{len(slides)}] {slide_path.name}", flush=True)
        try:
            row = process_slide(slide_path, output_dir, cfg)
            print(f"  {row['status']}: {row['patches']} patches", flush=True)
        except Exception as exc:  # continue so one malformed slide does not lose the cohort run
            failures += 1
            row = {"slide": slide_path.name, "status": "failed", "patches": "", "message": str(exc)}
            traceback.print_exc()
        rows.append(row)

    with (output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slide", "status", "patches", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finished {len(slides)} slides ({failures} failed). Manifest: {output_dir / 'manifest.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
