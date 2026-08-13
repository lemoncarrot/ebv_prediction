#!/usr/bin/env python3
"""Turn MIL attention (and additive contributions) into patch tables and heatmaps.

Reads the `attention/*.h5` written by `ebv.mil.train`. Coordinates sit on the
tiling lattice (anchored at level-0 (0,0), step patch_size_level0), so no
resampling is needed. Per slide: a patch table CSV, an optional grid CSV, and a
rendered PNG.

    python -m ebv.mil.heatmaps --run-dir 04_mil/chile_5fold
    python -m ebv.mil.heatmaps --run-dir 04_mil/tcga_additive --grid-value contribution --overlay
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ebv.paths import ROOT  # noqa: E402

DEFAULT_OUTPUT = ROOT / "05_attention_maps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--attention-dir", type=Path, default=None, help="Default: <run-dir>/attention.")
    p.add_argument("--output-dir", type=Path, default=None, help=f"Default: {DEFAULT_OUTPUT}/<run>.")
    p.add_argument("--slides", nargs="+", default=None, help="Only these slide ids.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overlay", action="store_true", help="Draw the slide thumbnail under the heatmap.")
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--cmap", default=None, help="Default RdBu_r for signed values, magma otherwise.")
    p.add_argument("--grid-value", choices=("log2_enrichment", "attention", "percentile", "contribution"),
                   default="log2_enrichment")
    p.add_argument("--clip-percentile", type=float, default=99.0, help="Colour saturates here (0 disables).")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-grid-csv", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def lattice(coords: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Level-0 coordinates -> integer (col, row) lattice indices."""
    if step <= 0:
        raise ValueError(f"invalid patch_size_level0: {step}")
    if (coords % step).any():
        off = int((coords % step).any(axis=1).sum())
        print(f"  note: {off} coordinates are not multiples of {step}; rounding", file=sys.stderr)
    return np.rint(coords[:, 0] / step).astype(int), np.rint(coords[:, 1] / step).astype(int)


def patch_table(coords, attention, step, contributions=None) -> pd.DataFrame:
    cols, rows = lattice(coords, step)
    n = len(attention)
    # log2(attention * n): 0 = average patch, +1 = twice average. Raw attention
    # is not comparable across slides (a 40k-patch bag has a 1/40000 baseline).
    enrichment = np.log2(np.maximum(attention, np.finfo(np.float32).tiny) * n)
    table = pd.DataFrame({
        "x": coords[:, 0].astype(int), "y": coords[:, 1].astype(int), "col": cols, "row": rows,
        "attention": attention.astype(np.float64), "log2_enrichment": enrichment,
        "percentile": pd.Series(attention).rank(pct=True).to_numpy(),
        "rank": pd.Series(attention).rank(ascending=False, method="min").astype(int).to_numpy(),
    })
    if contributions is not None:
        # Signed logit terms, already comparable across slides; rank by magnitude.
        table["contribution"] = contributions.astype(np.float64)
        table["contribution_abs"] = np.abs(contributions)
        table["contribution_rank"] = pd.Series(np.abs(contributions)).rank(
            ascending=False, method="min").astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def to_grid(table: pd.DataFrame, value: str):
    n_cols, n_rows = int(table["col"].max()) + 1, int(table["row"].max()) + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[table["row"].to_numpy(), table["col"].to_numpy()] = table[value].to_numpy()
    return grid, n_rows, n_cols


def colour_limits(values, grid_value, clip):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0, False
    if grid_value in ("log2_enrichment", "contribution"):  # signed, symmetric about 0
        span = float(max(np.percentile(np.abs(finite), clip) if clip else np.abs(finite).max(), 1e-3))
        return -span, span, True
    if clip:
        return float(np.percentile(finite, 100 - clip)), float(np.percentile(finite, clip)), False
    return float(finite.min()), float(finite.max()), False


def thumbnail(attrs, n_cols, n_rows):
    slide_path = attrs.get("slide_path")
    if not slide_path or not Path(str(slide_path)).exists():
        print(f"  overlay skipped: slide not found at {slide_path}", file=sys.stderr)
        return None
    try:
        import openslide
        with openslide.OpenSlide(str(slide_path)) as slide:
            return np.asarray(slide.get_thumbnail((n_cols * 4, n_rows * 4)).convert("RGB"))
    except Exception as exc:
        print(f"  overlay skipped: {exc}", file=sys.stderr)
        return None


def render(grid, attrs, args, out_path, n_rows, n_cols) -> None:
    vmin, vmax, diverging = colour_limits(grid, args.grid_value, args.clip_percentile)
    cmap = args.cmap or ("RdBu_r" if diverging else "magma")
    background = thumbnail(attrs, n_cols, n_rows) if args.overlay else None

    long_edge = 11.0
    if n_cols >= n_rows:
        width, height = long_edge, max(2.5, long_edge * n_rows / max(n_cols, 1))
    else:
        width, height = max(2.5, long_edge * n_cols / max(n_rows, 1)), long_edge
    figure, axis = plt.subplots(figsize=(width + 1.5, height), dpi=args.dpi)

    extent = (0, n_cols, n_rows, 0)
    if background is not None:
        axis.imshow(background, extent=extent, interpolation="bilinear")
    image = axis.imshow(np.ma.masked_invalid(grid), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="nearest", alpha=args.alpha if background is not None else 1.0)
    axis.set_xticks([]); axis.set_yticks([])

    label = {"log2_enrichment": "log2(attention x n patches)   0 = average patch",
             "attention": "attention weight", "percentile": "attention percentile",
             "contribution": "additive logit contribution   >0 excitatory, <0 inhibitory"}[args.grid_value]
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02).set_label(label)

    parts = [str(attrs.get("slide_id", out_path.stem))]
    if attrs.get("label") is not None:
        parts.append(f"EBV {'+' if int(attrs['label']) == 1 else '-'}")
    if attrs.get("prob") is not None:
        parts.append(f"p(EBV+) = {float(attrs['prob']):.3f}")
    parts.append(f"{int(attrs.get('n_patches', np.isfinite(grid).sum()))} patches")
    if "fold" in attrs:
        parts.append(f"fold {int(attrs['fold'])}")
    axis.set_title("   |   ".join(parts), fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def process(path: Path, args, output_dir: Path) -> dict:
    slide_id = path.stem
    heatmap_path = output_dir / f"{slide_id}_attention_heatmap.png"
    if heatmap_path.exists() and not args.overwrite:
        return {"slide_id": slide_id, "status": "skipped", "message": "exists"}

    with h5py.File(path, "r") as h:
        coords, attention = h["coords"][:], h["attention"][:]
        contributions = h["contributions"][:] if "contributions" in h else None
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in h.attrs.items()}
    if args.grid_value == "contribution" and contributions is None:
        raise ValueError("--grid-value contribution needs an additive-MIL run (no 'contributions' dataset).")

    step = int(attrs.get("patch_size_level0") or 0)
    if step <= 0:
        step = int(attrs.get("patch_size", 256) * float(attrs.get("target_downsample", 1.0)))
    table = patch_table(coords, np.asarray(attention, np.float64), step,
                        None if contributions is None else np.asarray(contributions, np.float64))
    table.to_csv(output_dir / f"{slide_id}_patch_attention.csv", index=False)

    grid, n_rows, n_cols = to_grid(table, args.grid_value)
    if not args.no_grid_csv:
        pd.DataFrame(grid, index=pd.Index(np.arange(n_rows) * step, name="y"),
                     columns=np.arange(n_cols) * step).to_csv(output_dir / f"{slide_id}_attention_grid.csv")
    render(grid, attrs, args, heatmap_path, n_rows, n_cols)

    top = table.iloc[0]
    return {"slide_id": slide_id, "status": "ok", "n_patches": len(table), "label": attrs.get("label"),
            "prob": attrs.get("prob"), "top_x": int(top["x"]), "top_y": int(top["y"]),
            "top_log2_enrichment": float(top["log2_enrichment"]), "message": ""}


def _worker(payload):
    path, args, output_dir = payload
    try:
        return process(path, args, output_dir)
    except Exception as exc:
        traceback.print_exc()
        return {"slide_id": Path(path).stem, "status": "failed", "message": str(exc)}


def main() -> int:
    args = parse_args()
    attention_dir = args.attention_dir or (args.run_dir / "attention")
    if not attention_dir.is_dir():
        print(f"ERROR: no attention directory at {attention_dir}. Run ebv.mil.train first.", file=sys.stderr)
        return 2
    paths = sorted(attention_dir.glob("*.h5"))
    if args.slides:
        wanted = set(args.slides)
        paths = [p for p in paths if p.stem in wanted]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"ERROR: no attention HDF5 files matched in {attention_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (DEFAULT_OUTPUT / args.run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} slides -> {output_dir}")

    payloads = [(path, args, output_dir) for path in paths]
    rows: list[dict] = []
    if args.workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, pl): pl[0] for pl in payloads}
            for done, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                print(f"[{done}/{len(paths)}] {row['slide_id']} {row['status']}", flush=True)
    else:
        for done, pl in enumerate(payloads, 1):
            row = _worker(pl)
            rows.append(row)
            print(f"[{done}/{len(paths)}] {row['slide_id']} {row['status']}", flush=True)

    summary = pd.DataFrame(rows).sort_values("slide_id")
    summary.to_csv(output_dir / "summary.csv", index=False)
    failed = int((summary["status"] == "failed").sum())
    print(f"\nwrote {output_dir} ({len(summary)} slides, {failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
