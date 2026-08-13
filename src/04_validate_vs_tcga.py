#!/usr/bin/env python3
"""Validate this pipeline against MahmoodLab's precomputed TCGA UNI2-h features.

Two independent checks, because two independent things can be wrong:

`encoder`
    Take *their* coordinates, read those patches with *this* repo's WSI reading
    and transform code, run UNI2-h, and compare vectors to their published ones.
    This isolates patch extraction + resampling + normalization + model call
    from tissue segmentation. Cosine similarity should be ~1.0.

`tiling`
    Run this repo's CLAM-style segmentation and compare the resulting coordinate
    grid to theirs. Their release was produced with Trident (`savetodir` in the
    HDF5 attributes), whose tissue segmentation is a learned model rather than
    CLAM's HSV threshold, so exact agreement is not expected. The useful numbers
    are grid alignment (are coordinates on the same lattice at all?) and the
    Jaccard overlap of the retained tile sets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = ROOT / "TCGA_STAD_precomputed_features" / "TCGA"
DEFAULT_SLIDE_DIR = ROOT / "00_tcga_WSI_raw"
DEFAULT_REPORT = ROOT / "03_validation" / "tcga_validation.json"


def load_sibling(module_name: str, file_stem: str):
    """Import a sibling script whose filename starts with a digit."""
    path = Path(__file__).resolve().parent / f"{file_stem}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=("encoder", "tiling"),
        default=["encoder", "tiling"],
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=64,
        help="Patches per slide for the encoder check (0 = all). UNI2-h on CPU is slow.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Write a QC image per slide comparing both coordinate sets.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("none", "float16", "bfloat16"), default=None)
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.999,
        help="Median cosine similarity below this marks the encoder check as failed.",
    )
    return parser.parse_args()


def reference_coords(path: Path) -> tuple[np.ndarray, dict]:
    """Read a published feature file. `coords`/`features` carry a leading axis of 1."""
    with h5py.File(path, "r") as handle:
        coords = np.asarray(handle["coords"])
        features = np.asarray(handle["features"])
        attrs = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle["coords"].attrs.items()
        }
    if coords.ndim == 3:
        coords = coords[0]
    if features.ndim == 3:
        features = features[0]
    if len(coords) != len(features):
        raise ValueError(f"{path.name}: {len(coords)} coords vs {len(features)} features")
    attrs["_features"] = features
    return coords, attrs


def check_encoder(
    slide_path: Path,
    coords: np.ndarray,
    reference: np.ndarray,
    attrs: dict,
    model,
    transform,
    args: argparse.Namespace,
    uni,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    patch_size = int(attrs.get("patch_size", 256))
    level0_patch = int(attrs.get("patch_size_level0", patch_size))
    target_ds = level0_patch / patch_size

    index = np.arange(len(coords))
    if args.max_patches and args.max_patches < len(index):
        index = np.sort(
            np.random.default_rng(args.seed).choice(index, args.max_patches, replace=False)
        )

    dataset = uni.WSIPatchDataset(slide_path, coords[index], patch_size, target_ds, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(args.device, non_blocking=True)
            with uni.autocast_context(args.device, args.amp):
                embeddings = model(images)
            blocks.append(embeddings.float().cpu().numpy())
    ours = np.concatenate(blocks, axis=0)
    theirs = reference[index].astype(np.float64)
    ours = ours.astype(np.float64)

    cosine = np.sum(ours * theirs, axis=1) / (
        np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1)
    )
    relative = np.linalg.norm(ours - theirs, axis=1) / np.linalg.norm(theirs, axis=1)
    return {
        "patches_compared": int(len(index)),
        "patches_available": int(len(coords)),
        "target_downsample": target_ds,
        "cosine_median": float(np.median(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_p01": float(np.percentile(cosine, 1)),
        "cosine_frac_above_0.99": float(np.mean(cosine > 0.99)),
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_max": float(relative.max()),
        "passed": bool(np.median(cosine) >= args.cosine_threshold),
    }


OVERLAY_LEGEND = (
    ((0, 170, 0), "both (agree)"),
    ((220, 30, 30), "ours only"),
    ((40, 90, 230), "theirs only"),
)


def save_overlay(
    rgb: np.ndarray,
    seg_ds: tuple[float, float],
    stride: int,
    ours: set[tuple[int, int]],
    theirs: set[tuple[int, int]],
    path: Path,
) -> None:
    """Draw both coordinate sets on the segmentation thumbnail, with a legend.

    Three disjoint categories, so a colour always means one thing: agreement,
    a tile only this pipeline produced, or a tile only the reference has.
    """
    import cv2

    ds_x, ds_y = seg_ds
    groups = ((ours & theirs, 0), (ours - theirs, 1), (theirs - ours, 2))
    layer = rgb.copy()
    for tiles, index in groups:
        colour = OVERLAY_LEGEND[index][0]
        for x, y in tiles:
            cv2.rectangle(
                layer,
                (int(x / ds_x), int(y / ds_y)),
                (int((x + stride) / ds_x), int((y + stride) / ds_y)),
                colour,
                -1,
            )
    canvas = cv2.addWeighted(layer, 0.5, rgb, 0.5, 0)

    scale = max(1.0, canvas.shape[1] / 1600.0)
    pad, box, line = int(12 * scale), int(22 * scale), int(30 * scale)
    cv2.rectangle(
        canvas, (0, 0), (int(330 * scale), pad * 2 + line * len(OVERLAY_LEGEND)), (255, 255, 255), -1
    )
    for index, (colour, label) in enumerate(OVERLAY_LEGEND):
        top = pad + index * line
        cv2.rectangle(canvas, (pad, top), (pad + box, top + box), colour, -1)
        count = len(groups[index][0])
        cv2.putText(
            canvas,
            f"{label}: {count}",
            (pad * 2 + box, top + box - int(4 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale,
            (0, 0, 0),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
    Image.fromarray(canvas).save(path, quality=88)


def check_tiling(
    slide_path: Path,
    coords: np.ndarray,
    attrs: dict,
    prep,
    overlay_path: Path | None = None,
) -> dict:
    import openslide

    cfg = prep.Config(
        patch_size=int(attrs.get("patch_size", 256)),
        target_mag=float(attrs.get("target_magnification", 20)),
        target_mpp=0.5,
        pen_filter=True,
        pen_green_sat=60,
        pen_cyan_sat=100,
        pen_red_sat=200,
        pen_dark_value=50,
        pen_dilate=5,
        # Their lattice is derived from an integer objective power, so only
        # `reference` mode can reproduce it. This says nothing about which mode
        # is right for extracting the cohort's own features.
        scale_mode="reference",
        overlap=int(attrs.get("overlap", 0)),
        seg_downsample=64.0,
        sthresh=8,
        mthresh=7,
        close=4,
        use_otsu=False,
        area_tissue=16.0,
        area_hole=4.0,
        max_holes=8,
        base_magnification=None,
        overwrite=True,
    )
    with openslide.OpenSlide(str(slide_path)) as slide:
        target_ds, mag_source = prep.target_scale(slide, cfg)
        base_mag = target_ds * cfg.target_mag
        seg_level = prep.closest_level(slide, cfg.seg_downsample)
        rgb, contours, holes, seg_ds, _ = prep.segment_tissue(slide, seg_level, cfg)
        ours = prep.coordinates_for_contours(
            contours, holes, slide.dimensions, target_ds, cfg
        )

    mine = {(int(x), int(y)) for x, y in ours}
    theirs = {(int(x), int(y)) for x, y in coords}
    shared = mine & theirs
    stride = int(round(cfg.patch_size * target_ds))
    off_lattice = sum(1 for x, y in theirs if x % stride or y % stride)
    if overlay_path is not None:
        save_overlay(rgb, seg_ds, stride, mine, theirs, overlay_path)
    return {
        "base_magnification": float(base_mag),
        "magnification_source": mag_source,
        "target_downsample": float(target_ds),
        "level0_stride": stride,
        "ours_n": len(mine),
        "theirs_n": len(theirs),
        "shared_n": len(shared),
        "jaccard": len(shared) / len(mine | theirs) if mine | theirs else 0.0,
        "recall_of_theirs": len(shared) / len(theirs) if theirs else 0.0,
        "precision_vs_theirs": len(shared) / len(mine) if mine else 0.0,
        "their_coords_off_our_lattice": off_lattice,
    }


def main() -> int:
    args = parse_args()
    feature_dir = args.feature_dir.resolve()
    slide_dir = args.slide_dir.resolve()
    if not feature_dir.is_dir() or not slide_dir.is_dir():
        print(f"ERROR: expected directories {feature_dir} and {slide_dir}", file=sys.stderr)
        return 2

    slides = {path.stem: path for path in sorted(slide_dir.rglob("*.svs"))}
    if not slides:
        print(
            f"ERROR: no .svs under {slide_dir}. Run src/03_fetch_tcga_slides.py first.",
            file=sys.stderr,
        )
        return 2
    pairs = [(name, slides[name], feature_dir / f"{name}.h5") for name in sorted(slides)]
    missing = [name for name, _, feature in pairs if not feature.exists()]
    if missing:
        print(f"ERROR: no precomputed features for: {', '.join(missing)}", file=sys.stderr)
        return 2

    prep = load_sibling("_prep", "00_preprocessing") if "tiling" in args.checks else None
    model = transform = uni = None
    if "encoder" in args.checks:
        import torch

        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.amp is None:
            args.amp = "float16" if args.device.startswith("cuda") else "none"
        if not args.device.startswith("cuda"):
            print(
                "WARNING: no CUDA device; UNI2-h on CPU is slow. Keep --max-patches small.",
                file=sys.stderr,
            )
        uni = load_sibling("_uni", "01_call_uni")
        from uni import get_encoder

        print(f"Loading UNI2-h on {args.device} ...", flush=True)
        model, transform = get_encoder(enc_name="uni2-h", device=args.device)
        model.eval()

    report: dict[str, dict] = {}
    failures = 0
    for name, slide_path, feature_path in pairs:
        print(f"\n=== {name}")
        coords, attrs = reference_coords(feature_path)
        reference = attrs.pop("_features")
        entry: dict[str, object] = {
            "slide": str(slide_path),
            "reference": str(feature_path),
            "reference_attrs": {k: v for k, v in attrs.items() if not k.startswith("_")},
        }
        try:
            if "tiling" in args.checks:
                entry["tiling"] = check_tiling(
                    slide_path, coords, attrs, prep,
                    args.report.parent / f"overlay_{name[:23]}.jpg" if args.overlay else None,
                )
                tiling = entry["tiling"]
                print(
                    f"  tiling : ours={tiling['ours_n']} theirs={tiling['theirs_n']} "
                    f"shared={tiling['shared_n']} jaccard={tiling['jaccard']:.3f} "
                    f"recall={tiling['recall_of_theirs']:.3f} "
                    f"off-lattice={tiling['their_coords_off_our_lattice']}"
                )
            if "encoder" in args.checks:
                entry["encoder"] = check_encoder(
                    slide_path, coords, reference, attrs, model, transform, args, uni
                )
                encoder = entry["encoder"]
                print(
                    f"  encoder: n={encoder['patches_compared']} "
                    f"cos_median={encoder['cosine_median']:.6f} "
                    f"cos_min={encoder['cosine_min']:.6f} "
                    f"relL2_median={encoder['relative_l2_median']:.4f} "
                    f"-> {'PASS' if encoder['passed'] else 'FAIL'}"
                )
                failures += not encoder["passed"]
        except Exception as exc:  # keep going so one bad slide does not lose the run
            failures += 1
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {entry['error']}", file=sys.stderr)
        report[name] = entry

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
