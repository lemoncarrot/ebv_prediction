#!/usr/bin/env python3
"""Benchmark UNI2-h embedding throughput and project cohort wall-clock.

Two measurements, because they have different bottlenecks and different fixes:

`gpu`
    Forward pass on synthetic tensors already resident on the device. This is
    the ceiling -- what the GPU can do if data never starves it. Tune batch size
    and AMP dtype here.

`e2e`
    The real path: OpenSlide reads a region, resizes it, the UNI2 transform runs,
    and the batch is copied to the device. If `e2e` sits well below `gpu`, the
    job is IO/CPU bound and more workers help; more GPUs will not.

Reported as patches/sec plus projected hours for the full cohort, so the numbers
translate directly into a scheduling decision.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORTS = {
    "chile": (ROOT / "01_preprocessed" / "patches", ROOT / "00_new_HE"),
    "tcga": (ROOT / "01_preprocessed_tcga" / "patches", ROOT / "00_tcga_WSI_raw"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", nargs="+", choices=("gpu", "e2e"), default=["gpu", "e2e"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--amp", nargs="+", choices=("none", "float16", "bfloat16"),
                        default=["float16", "bfloat16"])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 24])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-batches", type=int, default=12, help="Timed batches per gpu config.")
    parser.add_argument("--e2e-patches", type=int, default=2048, help="Patches per e2e config.")
    parser.add_argument("--e2e-batch-size", type=int, default=None,
                        help="Batch size for the e2e sweep (default: best from the gpu sweep).")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "03_validation" / "uni_benchmark.json")
    return parser.parse_args()


def load_sibling(name: str, stem: str):
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cohort_totals() -> dict[str, int]:
    totals = {}
    for name, (coords_dir, _) in DEFAULT_COHORTS.items():
        total = 0
        for path in glob.glob(str(coords_dir / "*.h5")):
            with h5py.File(path, "r") as handle:
                total += handle["coords"].shape[0]
        totals[name] = total
    return totals


def autocast(device: str, amp: str):
    return torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.float16 if amp == "float16" else torch.bfloat16,
        enabled=amp != "none",
    )


def bench_gpu(model, args, totals: dict[str, int]) -> list[dict]:
    results = []
    for amp in args.amp:
        for batch_size in args.batch_sizes:
            images = torch.randn(batch_size, 3, 224, 224, device=args.device)
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            try:
                with torch.inference_mode():
                    for _ in range(3):  # warm up autotuner and allocator
                        with autocast(args.device, amp):
                            model(images)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.gpu_batches):
                        with autocast(args.device, amp):
                            model(images)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  amp={amp:<9} bs={batch_size:<4} OOM")
                continue
            rate = args.gpu_batches * batch_size / elapsed
            peak = torch.cuda.max_memory_allocated(args.device) / 1e9
            torch.cuda.reset_peak_memory_stats(args.device)
            results.append({"mode": "gpu", "amp": amp, "batch_size": batch_size,
                            "patches_per_s": rate, "peak_gb": peak})
            print(f"  amp={amp:<9} bs={batch_size:<4} {rate:8.1f} patch/s   "
                  f"peak {peak:5.2f} GB   1 GPU: {sum(totals.values())/rate/3600:6.2f} h")
            del images
            torch.cuda.empty_cache()
    return results


def bench_e2e(model, transform, args, totals: dict[str, int], uni) -> list[dict]:
    from torch.utils.data import DataLoader

    # Pick the slide with the most coordinates so a sweep never runs dry.
    best = None
    for cohort, (coords_dir, slide_dir) in DEFAULT_COHORTS.items():
        for path in sorted(glob.glob(str(coords_dir / "*.h5")))[:80]:
            with h5py.File(path, "r") as handle:
                count = handle["coords"].shape[0]
            if best is None or count > best[0]:
                best = (count, Path(path), slide_dir)
    if best is None:
        print("  no coordinate files found; skipping e2e")
        return []
    count, coord_path, slide_dir = best
    slide_path = slide_dir / f"{coord_path.stem}.svs"
    if not slide_path.exists():
        print(f"  slide missing for {coord_path.stem}; skipping e2e")
        return []

    with h5py.File(coord_path, "r") as handle:
        coords = handle["coords"][:]
        attrs = dict(handle.attrs)
    patch_size = int(attrs.get("patch_size", 256))
    target_ds = float(attrs["target_downsample"])
    take = min(args.e2e_patches, len(coords))
    print(f"  slide {coord_path.stem[:40]} ({count} coords, using {take}), "
          f"target_ds={target_ds:.4f}")

    amp = args.amp[0]
    batch_size = args.e2e_batch_size
    results = []
    for workers in args.workers:
        dataset = uni.WSIPatchDataset(slide_path, coords[:take], patch_size, target_ds, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True,
                            persistent_workers=False, prefetch_factor=4 if workers else None)
        done = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for images in loader:
                images = images.to(args.device, non_blocking=True)
                with autocast(args.device, amp):
                    model(images)
                done += len(images)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rate = done / elapsed
        results.append({"mode": "e2e", "amp": amp, "batch_size": batch_size,
                        "workers": workers, "patches_per_s": rate})
        print(f"  workers={workers:<3} bs={batch_size:<4} {rate:8.1f} patch/s   "
              f"1 GPU: {sum(totals.values())/rate/3600:6.2f} h   "
              f"2 GPUs: {sum(totals.values())/rate/2/3600:6.2f} h")
    return results


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device", file=sys.stderr)
        return 2

    totals = cohort_totals()
    total = sum(totals.values())
    print("cohort patch counts:")
    for name, count in totals.items():
        print(f"  {name:<8} {count:>12,}")
    print(f"  {'TOTAL':<8} {total:>12,}\n")

    name = torch.cuda.get_device_name(args.device)
    print(f"device: {args.device} ({name}), torch {torch.__version__}")
    print(f"CPU cores: {os.cpu_count()}, channels_last={args.channels_last}\n")

    uni = load_sibling("_uni", "01_call_uni")
    from uni import get_encoder

    model, transform = get_encoder(enc_name="uni2-h", device=args.device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    results: list[dict] = []
    if "gpu" in args.mode:
        print("GPU-only (synthetic tensors, no data loading):")
        results += bench_gpu(model, args, totals)
    if "e2e" in args.mode:
        if args.e2e_batch_size is None:
            gpu_rows = [r for r in results if r["mode"] == "gpu"]
            args.e2e_batch_size = (
                max(gpu_rows, key=lambda r: r["patches_per_s"])["batch_size"] if gpu_rows else 32
            )
        print(f"\nEnd-to-end (real WSI reads, batch_size={args.e2e_batch_size}):")
        results += bench_e2e(model, transform, args, totals, uni)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"device": name, "torch": torch.__version__, "cohort_patches": totals,
         "results": results}, indent=2))
    print(f"\nWrote {args.report}")

    gpu_rows = [r for r in results if r["mode"] == "gpu"]
    e2e_rows = [r for r in results if r["mode"] == "e2e"]
    if gpu_rows and e2e_rows:
        ceiling = max(r["patches_per_s"] for r in gpu_rows)
        actual = max(r["patches_per_s"] for r in e2e_rows)
        print(f"\nGPU ceiling {ceiling:.0f} patch/s; best end-to-end {actual:.0f} patch/s "
              f"({100 * actual / ceiling:.0f}% of ceiling)")
        if actual < 0.75 * ceiling:
            print("Data loading is the bottleneck -- add workers before adding GPUs.")
        else:
            print("GPU-bound -- a second GPU should nearly halve wall-clock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
