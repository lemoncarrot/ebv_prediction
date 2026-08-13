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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Below this, the slide genuinely cannot supply the requested patch scale.
MIN_DOWNSAMPLE = 0.95


@dataclass(frozen=True)
class Config:
    patch_size: int
    target_mag: float
    target_mpp: float
    scale_mode: str
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
    pen_filter: bool
    pen_green_sat: int
    pen_cyan_sat: int
    pen_red_sat: int
    pen_dark_value: int
    pen_dilate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment .svs slides and save level-0 tissue patch coordinates."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--target-mag", type=float, default=20.0)
    parser.add_argument(
        "--target-mpp",
        type=float,
        default=0.5,
        help="Physical scale of the extracted patch, in um/px (scale-mode=physical).",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("physical", "reference"),
        default="physical",
        help="physical: resample to --target-mpp using the slide's measured MPP "
        "(correct patch scale on any scanner). reference: derive the scale from "
        "the objective power, reproducing MahmoodLab's coordinate lattice exactly.",
    )
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
    parser.add_argument(
        "--no-pen-filter",
        dest="pen_filter",
        action="store_false",
        help="Keep surgical-marker ink in the tissue mask (it is removed by default).",
    )
    parser.add_argument("--pen-green-sat", type=int, default=60)
    parser.add_argument("--pen-cyan-sat", type=int, default=100)
    parser.add_argument("--pen-red-sat", type=int, default=200)
    parser.add_argument("--pen-dark-value", type=int, default=50)
    parser.add_argument(
        "--pen-dilate", type=int, default=5,
        help="Dilate the ink mask by this many px at segmentation resolution.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Slides to segment in parallel. Each worker opens its own slide, "
        "so this scales with cores and IO.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N slides (for a smoke test)."
    )
    args = parser.parse_args()
    if args.patch_size <= 0 or args.target_mag <= 0:
        parser.error("--patch-size and --target-mag must be positive")
    if not 0 <= args.overlap < args.patch_size:
        parser.error("--overlap must be >= 0 and smaller than --patch-size")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.mthresh <= 0 or args.mthresh % 2 == 0:
        parser.error("--mthresh must be a positive odd integer")
    return args


STANDARD_OBJECTIVES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0)


def snap_to_standard(magnification: float, tolerance: float = 0.15) -> float | None:
    """Round a measured magnification to the nearest standard objective."""
    nearest = min(STANDARD_OBJECTIVES, key=lambda value: abs(value - magnification))
    if abs(nearest - magnification) / nearest <= tolerance:
        return nearest
    return None


def objective_power(slide: openslide.OpenSlide, fallback: float | None) -> tuple[float, str]:
    """Return the slide's level-0 magnification, snapped to a standard objective.

    MahmoodLab's published TCGA features record an integer `level0_magnification`
    (20 or 40) and derive `patch_size_level0` from it, so the level-0 field of
    view is exactly 256 or 512 px. Using the raw MPP instead shifts the field of
    view by however much the scanner's MPP deviates from nominal, and UNI2-h is
    sensitive to that: on TCGA-MX-A5UG a 0.8% field-of-view error moves patch
    embeddings from cosine 0.99996 to 0.986 against the published features.
    Matching the reference convention therefore matters more than honouring a
    scanner's exact reported spacing, so the objective power wins and MPP is
    used to cross-check it.
    """
    props = slide.properties
    raw_objective = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    objective = float(raw_objective) if raw_objective else None
    raw_mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    raw_mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
    mpp_x = float(raw_mpp_x) if raw_mpp_x else None
    mpp_y = float(raw_mpp_y) if raw_mpp_y else None

    mpp_magnification: float | None = None
    if mpp_x is not None and mpp_x > 0:
        if mpp_y is not None and mpp_y > 0 and abs(mpp_x - mpp_y) / mpp_x > 0.05:
            raise ValueError(f"anisotropic pixel spacing is unsupported: MPP_X={mpp_x}, MPP_Y={mpp_y}")
        # 0.25 um/px ~= 40x and 0.50 um/px ~= 20x.
        mpp_magnification = 10.0 / mpp_x

    if objective is not None:
        if mpp_magnification is not None and abs(objective - mpp_magnification) / objective > 0.03:
            warnings.warn(
                f"objective power ({objective:g}x) and MPP ({mpp_x:g} um/px, "
                f"~{mpp_magnification:.3f}x) disagree by "
                f"{100 * abs(objective - mpp_magnification) / objective:.1f}%; using the "
                f"objective power to match the published TCGA field of view",
                stacklevel=2,
            )
        return objective, "openslide.objective-power"

    if mpp_magnification is not None:
        snapped = snap_to_standard(mpp_magnification)
        if snapped is not None:
            return snapped, "openslide.mpp-x (snapped to standard objective)"
        warnings.warn(
            f"MPP implies {mpp_magnification:.3f}x, which is not near a standard "
            f"objective; using it unsnapped",
            stacklevel=2,
        )
        return mpp_magnification, "openslide.mpp-x"

    if fallback is not None:
        return fallback, "command-line fallback"
    raise ValueError(
        "slide has neither objective-power nor MPP metadata; pass --base-magnification"
    )


def level0_mpp(slide: openslide.OpenSlide) -> float | None:
    """Measured level-0 pixel spacing in um/px, or None if the slide omits it."""
    raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if not raw:
        return None
    value = float(raw)
    return value if value > 0 else None


def target_scale(slide: openslide.OpenSlide, cfg: Config) -> tuple[float, str]:
    """Return the level-0 downsample that yields the requested patch scale.

    Two conventions, and they only agree when the scanner is near-nominal:

    `physical`
        Resample by measured MPP so the extracted patch really is
        `--target-mpp` um/px. MahmoodLab's published TCGA patches land at
        0.4937-0.5040 um/px, so this is what puts a patch in the same scale
        band as the reference regardless of scanner.

    `reference`
        Derive the scale from objective power, as the published pipeline does
        (`level0_magnification` is an integer there, so the level-0 field is
        exactly 256 or 512 px). Use this to reproduce their coordinate lattice.

    On a nominal scanner these differ by well under 1%. On a Leica GT450, whose
    "40x" is 0.263 um/px rather than 0.25, they differ by 5% -- and UNI2-h is
    sensitive enough that a 5% field-of-view error costs ~0.05 cosine.
    """
    mpp = level0_mpp(slide)
    if cfg.scale_mode == "physical":
        if mpp is None:
            raise ValueError(
                "slide has no MPP metadata; use --scale-mode reference "
                "or supply --base-magnification"
            )
        return cfg.target_mpp / mpp, f"mpp {mpp:g} -> {cfg.target_mpp:g} um/px"

    base_mag, mag_source = objective_power(slide, cfg.base_magnification)
    if base_mag + 1e-6 < cfg.target_mag:
        raise ValueError(
            f"base magnification {base_mag:g}x is below requested {cfg.target_mag:g}x"
        )
    return base_mag / cfg.target_mag, f"{mag_source} ({base_mag:g}x -> {cfg.target_mag:g}x)"


def level_downsamples_xy(slide: openslide.OpenSlide) -> list[tuple[float, float]]:
    w0, h0 = slide.level_dimensions[0]
    return [(w0 / w, h0 / h) for w, h in slide.level_dimensions]


def closest_level(slide: openslide.OpenSlide, wanted: float) -> int:
    downsamples = level_downsamples_xy(slide)
    return min(
        range(len(downsamples)),
        key=lambda idx: abs(math.log(max(np.mean(downsamples[idx]), 1e-9) / wanted)),
    )


def pen_mask(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask of surgical-marker ink, computed on the segmentation thumbnail.

    Pure OpenCV: `cvtColor` to HSV, threshold four colour bands, then a morphological
    open/dilate. No stain-normalisation or pen-specific library is involved.

    H&E occupies a narrow slice of hue -- eosin pink around H=160-180 and
    hematoxylin purple around H=125-160 (OpenCV's H is 0-179). Marker ink sits
    outside that slice, or inside it at a saturation stain never reaches. Measured
    across 28 STAD slides, these bands flag a median of 0.001% of tissue on clean
    slides while catching 5.9-46% on the six inked ones:

    - green  H 35-85,  S >= pen_green_sat   -- H&E has essentially no green
    - cyan   H 85-100, S >= pen_cyan_sat
    - red    H <20 or >=175, S >= pen_red_sat -- eosin sits near S~90 at these
             hues, marker red near S~240, so saturation separates them
    - dark   V < pen_dark_value, S < 100     -- black marker cores

    The blue band 100-125 is deliberately excluded: it overlaps real hematoxylin,
    and including it flagged 21% of a clean slide as ink during calibration.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = (hsv[:, :, index].astype(np.int16) for index in range(3))
    ink = (
        ((hue >= 35) & (hue < 85) & (sat >= cfg.pen_green_sat))
        | ((hue >= 85) & (hue < 100) & (sat >= cfg.pen_cyan_sat))
        | (((hue < 20) | (hue >= 175)) & (sat >= cfg.pen_red_sat))
        | ((val < cfg.pen_dark_value) & (sat < 100))
    )
    ink = ink.astype(np.uint8)
    # Drop isolated speckles, then grow the mask so a tile cannot clip an ink edge.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    if cfg.pen_dilate > 0:
        kernel = np.ones((cfg.pen_dilate, cfg.pen_dilate), dtype=np.uint8)
        ink = cv2.dilate(ink, kernel)
    return ink.astype(bool)


def segment_tissue(
    slide: openslide.OpenSlide, level: int, cfg: Config
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]], tuple[float, float], float]:
    rgb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    saturation = cv2.medianBlur(saturation, cfg.mthresh)
    threshold_type = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if cfg.use_otsu else 0)
    threshold = 0 if cfg.use_otsu else cfg.sthresh
    _, mask = cv2.threshold(saturation, threshold, 255, threshold_type)
    if cfg.close > 0:
        kernel = np.ones((cfg.close, cfg.close), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove ink after the close, or the close would fill the holes it leaves.
    pen_fraction = 0.0
    if cfg.pen_filter:
        ink = pen_mask(rgb, cfg)
        tissue_before = int((mask > 0).sum())
        mask[ink] = 0
        if tissue_before:
            pen_fraction = float((tissue_before - int((mask > 0).sum())) / tissue_before)

    found = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return rgb, [], [], level_downsamples_xy(slide)[level], pen_fraction

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
    return rgb, tissue, holes_by_tissue, (ds_x, ds_y), pen_fraction


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
        box_x, box_y, width, height = cv2.boundingRect(contour)
        # Anchor the grid at the slide origin, as the published TCGA features do
        # (every coordinate there is an exact multiple of patch_size_level0).
        # Anchoring at each contour's bounding box instead gives every contour
        # its own phase offset, so no tile ever lands on the reference lattice.
        start_x = box_x - (box_x % step)
        start_y = box_y - (box_y % step)
        for x in range(start_x, box_x + width, step):
            for y in range(start_y, box_y + height, step):
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
    ink: np.ndarray | None = None,
) -> None:
    canvas = rgb.copy()
    if ink is not None and ink.any():
        # Tint excluded ink so an over-aggressive pen filter is visible on review.
        canvas[ink] = (0.45 * canvas[ink] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
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
        # TCGA ships a number of ~0.5 MB stand-ins (1600x1200, one pyramid
        # level, no vendor metadata). They fail later with a confusing "no MPP"
        # error, so name them for what they are.
        if slide.level_count == 1 and max(slide.dimensions) < 5000:
            raise ValueError(
                f"not a whole-slide image: {slide.dimensions[0]}x{slide.dimensions[1]}, "
                f"single pyramid level -- this is a placeholder/thumbnail file, not a WSI"
            )
        target_ds, scale_source = target_scale(slide, cfg)
        # Some native-20x slides sit a hair coarser than the target (e.g. 0.5005
        # um/px -> downsample 0.999). Rejecting those loses real slides over a
        # 0.1% difference, so allow mild upsampling and only refuse a slide that
        # is genuinely too coarse to supply the requested scale.
        if target_ds < MIN_DOWNSAMPLE:
            raise ValueError(
                f"slide is too coarse for the requested patch scale: downsample "
                f"{target_ds:.4f} would upsample by {100 * (1 / target_ds - 1):.1f}%"
            )
        if target_ds < 1.0 - 1e-6:
            warnings.warn(
                f"slide is marginally coarser than the target scale; upsampling by "
                f"{100 * (1 / target_ds - 1):.2f}%",
                stacklevel=2,
            )
        base_mag, mag_source = objective_power(slide, cfg.base_magnification)
        mpp = level0_mpp(slide)
        seg_level = closest_level(slide, cfg.seg_downsample)
        rgb, contours, holes, seg_ds, pen_fraction = segment_tissue(slide, seg_level, cfg)
        coords = coordinates_for_contours(contours, holes, slide.dimensions, target_ds, cfg)

        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as handle:
            dataset = handle.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
            metadata = {
                "slide_path": str(slide_path.resolve()),
                "slide_name": slide_path.name,
                "patch_size": cfg.patch_size,
                "patch_size_level0": int(round(cfg.patch_size * target_ds)),
                "patch_level": -1,
                "target_magnification": cfg.target_mag,
                "target_downsample": target_ds,
                "target_mpp": cfg.target_mpp,
                "patch_mpp": (mpp * target_ds) if mpp is not None else float("nan"),
                "level0_mpp": mpp if mpp is not None else float("nan"),
                "scale_mode": cfg.scale_mode,
                "scale_source": scale_source,
                "base_magnification": base_mag,
                "magnification_source": mag_source,
                "overlap": cfg.overlap,
                "level0_width": slide.dimensions[0],
                "level0_height": slide.dimensions[1],
                "segmentation_level": seg_level,
                "segmentation_downsample_x": seg_ds[0],
                "segmentation_downsample_y": seg_ds[1],
                "segmentation_method": "CLAM_TCGA_HSV",
                "pen_filter": cfg.pen_filter,
                "pen_fraction_removed": pen_fraction,
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
        save_qc(
            rgb, contours, holes, seg_ds, qc_dir / f"{slide_path.stem}.jpg",
            pen_mask(rgb, cfg) if cfg.pen_filter else None,
        )
    return {"slide": slide_path.name, "status": "ok", "patches": len(coords), "message": ""}


def safe_process(slide_path: Path, output_dir: Path, cfg: Config) -> dict[str, object]:
    """Run one slide, converting any failure into a manifest row.

    One malformed slide must not lose a cohort run, and in a worker process an
    uncaught exception would surface only as a broken future.
    """
    try:
        return process_slide(slide_path, output_dir, cfg)
    except Exception as exc:
        traceback.print_exc()
        return {
            "slide": slide_path.name,
            "status": "failed",
            "patches": "",
            "message": f"{type(exc).__name__}: {exc}",
        }


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
        target_mpp=args.target_mpp,
        scale_mode=args.scale_mode,
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
        pen_filter=args.pen_filter,
        pen_green_sat=args.pen_green_sat,
        pen_cyan_sat=args.pen_cyan_sat,
        pen_red_sat=args.pen_red_sat,
        pen_dark_value=args.pen_dark_value,
        pen_dilate=args.pen_dilate,
    )

    rows: list[dict[str, object]] = []
    failures = 0
    if args.jobs == 1:
        for index, slide_path in enumerate(slides, start=1):
            print(f"[{index}/{len(slides)}] {slide_path.name}", flush=True)
            row = safe_process(slide_path, output_dir, cfg)
            failures += row["status"] == "failed"
            print(f"  {row['status']}: {row['patches']} {row['message']}", flush=True)
            rows.append(row)
    else:
        # Each slide opens its own OpenSlide handle and writes its own files, so
        # slides are independent; a pool just needs the picklable Config.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(safe_process, path, output_dir, cfg): path for path in slides
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                failures += row["status"] == "failed"
                print(
                    f"[{index}/{len(slides)}] {row['slide']}: "
                    f"{row['status']} {row['patches']} {row['message']}",
                    flush=True,
                )
                rows.append(row)
        rows.sort(key=lambda item: str(item["slide"]))

    with (output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slide", "status", "patches", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finished {len(slides)} slides ({failures} failed). Manifest: {output_dir / 'manifest.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
