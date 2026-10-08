#!/usr/bin/env python3
"""Segment, tile and encode H&E slides in a single pass.

Nothing intermediate is persisted except one QC overlay PNG per slide: the
tissue mask stays in memory and the tiles are encoded as they are read, so the
only real output is one HDF5 of embeddings per slide.

    python run_pipeline.py --slides ../00_new_HE --out-dir ../02_new_embeddings
    python run_pipeline.py --slides a.svs b.svs --out-dir out --batch-size 32
    python run_pipeline.py --slides ../00_new_HE --out-dir out --limit 1   # smoke test

Each output is <out-dir>/<slide_id>.h5 with `coords` (N, 2) level-0 tile origins
and `features` (N, embed_dim) float32, plus geometry and segmentation params in
the file attrs. Re-running skips slides whose .h5 already exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from SlideReaderClass import SlideReader
from TissueSegmentation import segment_slide
from encoders.UNI import UNIEncoder

SLIDE_SUFFIXES = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slides", type=Path, nargs="+", required=True,
                   help="Slide files, and/or directories searched recursively.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where the per-slide embedding HDF5 files are written.")
    p.add_argument("--qc-dir", type=Path, default=None,
                   help="Segmentation overlay PNGs (default: <out-dir>/qc).")
    p.add_argument("--no-qc", action="store_true", help="Skip the overlay PNGs entirely.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4,
                   help="Threads reading tiles ahead of the GPU (openslide releases the GIL).")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", choices=("none", "float16", "bfloat16"), default="float16")
    p.add_argument("--min-tissue", type=float, default=0.10,
                   help="Minimum masked fraction for a tile to be kept.")
    p.add_argument("--max-dim", type=int, default=4096, help="Thumbnail size used for segmentation.")
    p.add_argument("--sat-thresh", type=int, default=None,
                   help="Saturation threshold; Otsu is used when unset.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def find_slides(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        item = item.expanduser().resolve()
        if item.is_dir():
            found += [p for p in item.rglob("*") if p.suffix.lower() in SLIDE_SUFFIXES and p.is_file()]
        elif item.is_file():
            found.append(item)
        else:
            print(f"WARNING: no such slide or directory: {item}", file=sys.stderr)
    # Duplicate stems would silently overwrite each other's HDF5.
    unique = sorted(set(found))
    seen: dict[str, Path] = {}
    for path in unique:
        if path.stem in seen:
            raise SystemExit(f"ERROR: duplicate slide id {path.stem!r}: {seen[path.stem]} and {path}")
        seen[path.stem] = path
    return unique


def tile_batches(reader: SlideReader, coords: np.ndarray, batch_size: int, workers: int):
    """Yield uint8 (B, tile, tile, 3) batches, reading a few batches ahead.

    The GPU forward pass and the openslide reads overlap: `workers` batches are
    in flight while the caller works on the one it just received.
    """
    slices = [coords[i: i + batch_size] for i in range(0, len(coords), batch_size)]

    def load(batch: np.ndarray) -> np.ndarray:
        return np.stack([reader.read_tile(int(x), int(y)) for x, y in batch])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque = deque()
        nxt = 0
        while nxt < len(slices) and len(pending) < max(1, workers):
            pending.append(pool.submit(load, slices[nxt]))
            nxt += 1
        while pending:
            tiles = pending.popleft().result()
            if nxt < len(slices):
                pending.append(pool.submit(load, slices[nxt]))
                nxt += 1
            yield tiles


def embed_slide(slide_path: Path, out_path: Path, encoder: UNIEncoder, args) -> int:
    """Segment, tile and encode one slide. Returns the number of tiles written."""
    spec = encoder.spec
    slide_id = slide_path.stem

    seg_params = {} if args.sat_thresh is None else {"sat_thresh": args.sat_thresh}
    qc_path = None if args.no_qc else args.qc_dir / f"{slide_id}.png"

    with SlideReader(slide_path, mpp_target=spec.mpp_target, tile_size=spec.tile_size) as reader:
        g = reader.geometry
        mask, _thumb, thumb_ds, thumb_mpp = segment_slide(
            reader, max_dim=args.max_dim, qc_path=qc_path, **seg_params
        )
        coords = np.array(
            list(reader.tile_origins(mask=mask, mask_downsample=thumb_ds,
                                     min_tissue=args.min_tissue)),
            dtype=np.int32,
        ).reshape(-1, 2)

        attrs = {
            "slide_id": slide_id,
            "slide_path": str(slide_path),
            "encoder": spec.name,
            "embedding_dim": spec.embed_dim,
            "model_input_size": spec.model_input_size,
            "amp": encoder.amp,
            # Tiling lattice, in the names the existing heatmap code looks for.
            "patch_size": spec.tile_size,
            "patch_size_level0": int(round(g.step_level0)),
            "target_downsample": g.mpp_target / g.mpp_base,
            # Geometry, to audit what scale each slide was actually read at.
            "mpp_base": g.mpp_base,
            "mpp_target": g.mpp_target,
            "mpp_achieved": g.mpp_achieved,
            "mpp_source": g.mpp_source,
            "level": g.level,
            "read_size": g.read_size,
            # Segmentation, since the mask itself is no longer kept.
            "tissue_fraction": float(mask.mean()),
            "min_tissue": args.min_tissue,
            "thumb_downsample": thumb_ds,
            "thumb_mpp": thumb_mpp,
            "max_dim": args.max_dim,
            **{f"seg_{k}": v for k, v in seg_params.items()},
        }

        tmp = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as out:
            out.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
            features = out.create_dataset(
                "features",
                shape=(len(coords), spec.embed_dim),
                dtype=np.float32,
                chunks=(min(args.batch_size, len(coords)), spec.embed_dim) if len(coords) else None,
            )
            offset = 0
            for tiles in tile_batches(reader, coords, args.batch_size, args.workers):
                batch = encoder.encode(tiles)
                features[offset: offset + len(batch)] = batch
                offset += len(batch)
            if offset != len(coords):
                raise RuntimeError(f"encoded {offset} tiles but expected {len(coords)}")
            out.attrs.update(attrs)

    tmp.replace(out_path)
    return len(coords)


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.qc_dir = (args.qc_dir or args.out_dir / "qc").expanduser().resolve()

    slides = find_slides(args.slides)
    if args.limit is not None:
        slides = slides[: args.limit]
    if not slides:
        print(f"ERROR: no slides found in {', '.join(map(str, args.slides))}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {UNIEncoder.spec.name} ...", flush=True)
    encoder = UNIEncoder(device=args.device, amp=args.amp)
    print(f"  device={encoder.device} amp={encoder.amp} "
          f"tile={encoder.spec.tile_size}px @ {encoder.spec.mpp_target} MPP", flush=True)

    rows, failures = [], 0
    for i, slide_path in enumerate(slides, 1):
        out_path = args.out_dir / f"{slide_path.stem}.h5"
        print(f"[{i}/{len(slides)}] {slide_path.stem}", flush=True)
        if out_path.exists() and not args.overwrite:
            rows.append({"slide": slide_path.stem, "status": "skipped", "tiles": "", "message": "exists"})
            print("  skipped (exists)", flush=True)
            continue
        try:
            n = embed_slide(slide_path, out_path, encoder, args)
            rows.append({"slide": slide_path.stem, "status": "ok", "tiles": n, "message": ""})
            print(f"  ok: {n} embeddings -> {out_path.name}", flush=True)
        except Exception as exc:
            failures += 1
            rows.append({"slide": slide_path.stem, "status": "failed", "tiles": "", "message": str(exc)})
            traceback.print_exc()

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "status", "tiles", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finished {len(slides)} slides ({failures} failed). Manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
