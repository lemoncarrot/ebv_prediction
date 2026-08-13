#!/usr/bin/env python3
"""Encode preprocessed patch coordinates into per-slide UNI2-h embeddings.

One HDF5 per slide: `coords` (N x 2) and `features` (N x 1536). Pixels are read
straight from the WSI, so this needs both the coordinate files and the slides.

    python -m ebv.embedding.encode --limit 1 --batch-size 16          # smoke test
    python -m ebv.embedding.encode --batch-size 32 --workers 8        # cohort
    CUDA_VISIBLE_DEVICES=0 python -m ebv.embedding.encode --shard-count 2 --shard-index 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import traceback

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ebv.embedding.dataset import WSIPatchDataset, autocast_context
from ebv.paths import ROOT

DEFAULT_INPUT = ROOT / "00_new_HE"
DEFAULT_COORDS = ROOT / "01_preprocessed" / "patches"
DEFAULT_OUTPUT = ROOT / "02_uni2_embeddings"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--coords-dir", type=Path, default=DEFAULT_COORDS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", choices=("none", "float16", "bfloat16"), default="float16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-count", type=int, default=1,
                   help="Split the slide list across this many processes (one GPU each).")
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error("--shard-index must be in [0, --shard-count)")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.device.startswith("cuda") and args.amp == "float16":
        args.amp = "none"
    return args


def shard(coord_files: list[Path], count: int, index: int) -> list[Path]:
    """Balance slides across shards by patch count (greedy longest-first), since
    slides span two orders of magnitude and round-robin would idle a GPU."""
    if count == 1:
        return coord_files
    sizes = []
    for path in coord_files:
        with h5py.File(path, "r") as h:
            sizes.append((h["coords"].shape[0], path))
    loads, buckets = [0] * count, [[] for _ in range(count)]
    for patches, path in sorted(sizes, key=lambda s: -s[0]):
        target = min(range(count), key=lambda i: loads[i])
        loads[target] += patches
        buckets[target].append(path)
    print(f"shard {index}/{count}: {len(buckets[index])} slides, {loads[index]:,} patches", flush=True)
    return sorted(buckets[index])


def find_slides(input_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in input_dir.rglob("*.svs") if p.is_file()}


def extract_one(coord_path: Path, slide_path: Path, output_path: Path, model, transform, args) -> int:
    with h5py.File(coord_path, "r") as src:
        coords = src["coords"][:]
        attrs = dict(src.attrs)
    patch_size = int(attrs.get("patch_size", 256))
    target_ds = float(attrs["target_downsample"])

    dataset = WSIPatchDataset(slide_path, coords, patch_size, target_ds, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), persistent_workers=args.workers > 0)

    tmp = output_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as out:
        out.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        if len(coords) == 0:  # a blank slide: write an empty feature set and stop
            out.create_dataset("features", data=np.empty((0, 1536), dtype=np.float32))
            for k, v in attrs.items():
                out.attrs[k] = v
            out.attrs.update(encoder="MahmoodLab/UNI2-h", embedding_dim=1536,
                             model_input_size=224, amp=args.amp)
            tmp.replace(output_path)
            return 0
        features = out.create_dataset("features", shape=(len(coords), 1536), dtype=np.float32,
                                      chunks=(min(args.batch_size, len(coords)), 1536))
        offset = 0
        with torch.inference_mode():
            for images in loader:
                images = images.to(args.device, non_blocking=True)
                with autocast_context(args.device, args.amp):
                    batch = model(images).float().cpu().numpy()
                features[offset: offset + len(batch)] = batch
                offset += len(batch)
        for k, v in attrs.items():
            out.attrs[k] = v
        out.attrs.update(encoder="MahmoodLab/UNI2-h", embedding_dim=1536,
                         model_input_size=224, amp=args.amp)
    tmp.replace(output_path)
    return len(coords)


def main() -> int:
    args = parse_args()
    coords_dir, output_dir = args.coords_dir.resolve(), args.output_dir.resolve()
    coord_files = sorted(coords_dir.glob("*.h5"))
    if args.limit is not None:
        coord_files = coord_files[: args.limit]
    if not coord_files:
        print(f"ERROR: no coordinate HDF5 files in {coords_dir}", file=sys.stderr)
        return 2
    coord_files = shard(coord_files, args.shard_count, args.shard_index)

    slides = find_slides(args.input_dir.resolve())
    missing = [p.stem for p in coord_files if p.stem not in slides]
    if missing:
        print(f"ERROR: no matching .svs for: {', '.join(missing[:10])}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading UNI2-h on {args.device} ...", flush=True)
    from uni import get_encoder
    model, transform = get_encoder(enc_name="uni2-h", device=args.device)
    model.eval()

    rows, failures = [], 0
    for i, coord_path in enumerate(coord_files, 1):
        output_path = output_dir / coord_path.name
        print(f"[{i}/{len(coord_files)}] {coord_path.stem}", flush=True)
        if output_path.exists() and not args.overwrite:
            rows.append({"slide": coord_path.stem, "status": "skipped", "patches": "", "message": "exists"})
            continue
        try:
            count = extract_one(coord_path, slides[coord_path.stem], output_path, model, transform, args)
            rows.append({"slide": coord_path.stem, "status": "ok", "patches": count, "message": ""})
            print(f"  ok: {count} embeddings", flush=True)
        except Exception as exc:
            failures += 1
            rows.append({"slide": coord_path.stem, "status": "failed", "patches": "", "message": str(exc)})
            traceback.print_exc()

    name = "manifest.csv" if args.shard_count == 1 else f"manifest_shard{args.shard_index}.csv"
    with (output_dir / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "status", "patches", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finished {len(coord_files)} slides ({failures} failed). Manifest: {output_dir / name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
