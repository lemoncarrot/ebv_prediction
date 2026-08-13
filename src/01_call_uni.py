#!/usr/bin/env python3
"""Extract patch-level UNI2-h embeddings from preprocessed WSI coordinates."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "00_new_HE"
DEFAULT_COORDS = ROOT / "01_preprocessed" / "patches"
DEFAULT_OUTPUT = ROOT / "02_uni2_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNI2-h over WSI patch coordinates.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coords-dir", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("none", "float16", "bfloat16"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help="Split the slide list across this many independent processes (one GPU each).",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Which shard this process runs.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be non-negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cpu":
            print(
                "WARNING: no CUDA device visible to torch, falling back to CPU (orders of "
                "magnitude slower).\n"
                "         Check `nvidia-smi` and `torch.cuda.is_available()`; pass "
                "--device cpu to silence this.",
                file=sys.stderr,
                flush=True,
            )
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"--device {args.device} requested but torch reports no CUDA device")
    if not str(args.device).startswith("cuda") and args.amp == "float16":
        args.amp = "none"
    return args


class WSIPatchDataset(Dataset):
    def __init__(
        self,
        slide_path: Path,
        coords: np.ndarray,
        patch_size: int,
        target_downsample: float,
        transform,
    ) -> None:
        self.slide_path = str(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.target_downsample = target_downsample
        self.transform = transform
        self._slide: openslide.OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.coords)

    def _get_slide(self) -> openslide.OpenSlide:
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getitem__(self, index: int) -> torch.Tensor:
        slide = self._get_slide()
        x, y = (int(v) for v in self.coords[index])
        level = slide.get_best_level_for_downsample(self.target_downsample)
        level_ds = float(slide.level_downsamples[level])
        read_size = max(1, int(round(self.patch_size * self.target_downsample / level_ds)))
        patch = slide.read_region((x, y), level, (read_size, read_size)).convert("RGB")
        if patch.size != (self.patch_size, self.patch_size):
            patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
        return self.transform(patch)

    def __del__(self) -> None:
        if self._slide is not None:
            self._slide.close()


def shard(coord_files: list[Path], count: int, index: int) -> list[Path]:
    """Split slides across shards by patch count, not by position.

    Slides differ by two orders of magnitude in patch count, so round-robin would
    leave one GPU idle for hours. Greedy longest-processing-time first: hand the
    largest remaining slide to whichever shard has the least work queued.
    """
    if count == 1:
        return coord_files
    sizes = []
    for path in coord_files:
        with h5py.File(path, "r") as handle:
            sizes.append((handle["coords"].shape[0], path))
    loads = [0] * count
    buckets: list[list[Path]] = [[] for _ in range(count)]
    for patches, path in sorted(sizes, key=lambda item: -item[0]):
        target = min(range(count), key=lambda i: loads[i])
        loads[target] += patches
        buckets[target].append(path)
    print(f"shard {index}/{count}: {len(buckets[index])} slides, {loads[index]:,} patches "
          f"(shard loads: {', '.join(f'{n:,}' for n in loads)})", flush=True)
    return sorted(buckets[index])


def find_slides(input_dir: Path) -> dict[str, Path]:
    slides = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".svs"]
    by_stem: dict[str, Path] = {}
    for path in slides:
        if path.stem in by_stem:
            raise ValueError(f"duplicate slide stem: {path.stem}")
        by_stem[path.stem] = path
    return by_stem


def autocast_context(device: str, amp: str):
    enabled = amp != "none"
    dtype = torch.float16 if amp == "float16" else torch.bfloat16
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def extract_one(
    coord_path: Path,
    slide_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    transform,
    args: argparse.Namespace,
) -> int:
    with h5py.File(coord_path, "r") as source:
        coords = source["coords"][:]
        attrs = dict(source.attrs)
    patch_size = int(attrs.get("patch_size", 256))
    target_ds = float(attrs["target_downsample"])

    dataset = WSIPatchDataset(slide_path, coords, patch_size, target_ds, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )

    tmp_path = output_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as target:
        target.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        if len(coords):
            features = target.create_dataset(
                "features", shape=(len(coords), 1536), dtype=np.float32,
                chunks=(min(args.batch_size, len(coords)), 1536),
            )
        else:
            features = target.create_dataset(
                "features", data=np.empty((0, 1536), dtype=np.float32)
            )
        offset = 0
        with torch.inference_mode():
            for images in loader:
                images = images.to(args.device, non_blocking=True)
                with autocast_context(args.device, args.amp):
                    embeddings = model(images)
                if not isinstance(embeddings, torch.Tensor):
                    raise TypeError(f"UNI2 returned {type(embeddings).__name__}, expected Tensor")
                batch = embeddings.float().cpu().numpy()
                if batch.ndim != 2 or batch.shape[1] != 1536:
                    raise ValueError(f"unexpected UNI2 feature shape: {batch.shape}")
                features[offset : offset + len(batch)] = batch
                offset += len(batch)

        for key, value in attrs.items():
            target.attrs[key] = value
        target.attrs["encoder"] = "MahmoodLab/UNI2-h"
        target.attrs["embedding_dim"] = 1536
        target.attrs["model_input_size"] = 224
        target.attrs["amp"] = args.amp
    tmp_path.replace(output_path)
    return len(coords)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    coords_dir = args.coords_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir() or not coords_dir.is_dir():
        print(f"ERROR: expected directories {input_dir} and {coords_dir}", file=sys.stderr)
        return 2
    coord_files = sorted(coords_dir.glob("*.h5"))
    if args.limit is not None:
        coord_files = coord_files[: args.limit]
    if not coord_files:
        print(f"ERROR: no coordinate HDF5 files found in {coords_dir}", file=sys.stderr)
        return 2
    coord_files = shard(coord_files, args.shard_count, args.shard_index)

    try:
        slides = find_slides(input_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    missing = [path.stem for path in coord_files if path.stem not in slides]
    if missing:
        print(f"ERROR: no matching .svs for coordinate files: {', '.join(missing[:10])}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading UNI2-h on {args.device} ...", flush=True)
    try:
        from uni import get_encoder
    except ImportError as exc:
        print("ERROR: UNI is not installed. Complete the official UNI installation first.", file=sys.stderr)
        raise SystemExit(2) from exc
    model, transform = get_encoder(enc_name="uni2-h", device=args.device)
    model.eval()

    rows: list[dict[str, object]] = []
    failures = 0
    for index, coord_path in enumerate(coord_files, start=1):
        output_path = output_dir / coord_path.name
        print(f"[{index}/{len(coord_files)}] {coord_path.stem}", flush=True)
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
    with (output_dir / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slide", "status", "patches", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finished {len(coord_files)} slides ({failures} failed). Manifest: {output_dir / name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
