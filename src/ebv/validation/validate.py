#!/usr/bin/env python3
"""Validate the encoder against MahmoodLab's precomputed TCGA UNI2-h features.

Read patches at *their* coordinates with this repo's WSI code + transform, run
UNI2-h, and compare to their published vectors. This isolates patch extraction,
resampling, normalization, and the model call. Cosine similarity should be ~1.0.

    python -m ebv.validation.fetch_tcga --auto 3
    python -m ebv.validation.validate --max-patches 64
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import h5py
import numpy as np

from ebv.embedding.dataset import WSIPatchDataset, autocast_context
from ebv.paths import ROOT

DEFAULT_FEATURE_DIR = ROOT / "TCGA_STAD_precomputed_features" / "TCGA"
DEFAULT_SLIDE_DIR = ROOT / "00_tcga_WSI_raw"
DEFAULT_REPORT = ROOT / "03_validation" / "tcga_validation.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--max-patches", type=int, default=64, help="Patches per slide (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", choices=("none", "float16", "bfloat16"), default=None)
    p.add_argument("--cosine-threshold", type=float, default=0.999,
                   help="Median cosine below this marks the check failed.")
    return p.parse_args()


def reference_coords(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a published feature file; coords/features may carry a leading axis of 1."""
    with h5py.File(path, "r") as h:
        coords = np.asarray(h["coords"])
        features = np.asarray(h["features"])
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in h["coords"].attrs.items()}
    if coords.ndim == 3:
        coords = coords[0]
    if features.ndim == 3:
        features = features[0]
    return coords, features, attrs


def check_encoder(slide_path, coords, reference, attrs, model, transform, args) -> dict:
    import torch
    from torch.utils.data import DataLoader

    patch_size = int(attrs.get("patch_size", 256))
    target_ds = int(attrs.get("patch_size_level0", patch_size)) / patch_size

    index = np.arange(len(coords))
    if args.max_patches and args.max_patches < len(index):
        index = np.sort(np.random.default_rng(args.seed).choice(index, args.max_patches, replace=False))

    dataset = WSIPatchDataset(slide_path, coords[index], patch_size, target_ds, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=str(args.device).startswith("cuda"))
    blocks = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(args.device, non_blocking=True)
            with autocast_context(args.device, args.amp):
                blocks.append(model(images).float().cpu().numpy())
    ours = np.concatenate(blocks).astype(np.float64)
    theirs = reference[index].astype(np.float64)

    cosine = np.sum(ours * theirs, axis=1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1))
    relative = np.linalg.norm(ours - theirs, axis=1) / np.linalg.norm(theirs, axis=1)
    return {"patches_compared": int(len(index)), "patches_available": int(len(coords)),
            "target_downsample": target_ds, "cosine_median": float(np.median(cosine)),
            "cosine_min": float(cosine.min()), "cosine_p01": float(np.percentile(cosine, 1)),
            "relative_l2_median": float(np.median(relative)), "relative_l2_max": float(relative.max()),
            "passed": bool(np.median(cosine) >= args.cosine_threshold)}


def main() -> int:
    args = parse_args()
    feature_dir, slide_dir = args.feature_dir.resolve(), args.slide_dir.resolve()
    slides = {p.stem: p for p in sorted(slide_dir.rglob("*.svs"))}
    if not slides:
        print(f"ERROR: no .svs under {slide_dir}. Run ebv.validation.fetch_tcga first.", file=sys.stderr)
        return 2
    pairs = [(name, slides[name], feature_dir / f"{name}.h5") for name in sorted(slides)]
    missing = [name for name, _, feat in pairs if not feat.exists()]
    if missing:
        print(f"ERROR: no precomputed features for: {', '.join(missing)}", file=sys.stderr)
        return 2

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp is None:
        args.amp = "float16" if args.device.startswith("cuda") else "none"
    if not args.device.startswith("cuda"):
        print("WARNING: no CUDA device; UNI2-h on CPU is slow. Keep --max-patches small.", file=sys.stderr)

    from uni import get_encoder
    print(f"Loading UNI2-h on {args.device} ...", flush=True)
    model, transform = get_encoder(enc_name="uni2-h", device=args.device)
    model.eval()

    report, failures = {}, 0
    for name, slide_path, feature_path in pairs:
        print(f"\n=== {name}")
        coords, reference, attrs = reference_coords(feature_path)
        entry = {"slide": str(slide_path), "reference": str(feature_path)}
        try:
            encoder = check_encoder(slide_path, coords, reference, attrs, model, transform, args)
            entry["encoder"] = encoder
            print(f"  encoder: n={encoder['patches_compared']} cos_median={encoder['cosine_median']:.6f} "
                  f"cos_min={encoder['cosine_min']:.6f} relL2_median={encoder['relative_l2_median']:.4f} "
                  f"-> {'PASS' if encoder['passed'] else 'FAIL'}")
            failures += not encoder["passed"]
        except Exception as exc:
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
