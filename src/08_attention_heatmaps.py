#!/usr/bin/env python3
"""Turn MIL attention weights into per-slide patch-importance tables and heatmaps.

Input is the `attention/*.h5` written by `07_train_mil.py` -- one attention
weight per patch, plus the level-0 coordinate of that patch and the slide
geometry. Because preprocessing anchors every slide's tiling lattice at level-0
`(0, 0)` with a step of `patch_size_level0`, coordinates map onto an exact
integer grid, so the heatmap needs no resampling.

For each slide three files are written to `--output-dir/<run>/`:

    <slide>_patch_attention.csv   one row per patch, with:
        x, y                level-0 pixel coordinate of the patch corner
        col, row            its position in the tiling lattice
        attention           raw MIL weight; sums to 1 over the slide
        log2_enrichment     log2(attention * n_patches). 0 = an average patch,
                            +1 = twice the attention an average patch got. This
                            is the scale the heatmap uses, because raw attention
                            is not comparable between slides -- a 40k-patch
                            slide has a 1/40000 baseline and a 3k-patch slide a
                            1/3000 one.
        percentile          rank of the patch within the slide, in [0, 1]
        rank                1 = most attended patch on the slide

    <slide>_attention_grid.csv    the same weights as a row/column matrix
                                  (blank where no tissue was tiled), for anyone
                                  who would rather plot it themselves

    <slide>_attention_heatmap.png the rendered map; with --overlay the slide
                                  thumbnail is drawn underneath it

Usage:

    python src/08_attention_heatmaps.py --run-dir 04_mil/chile_5fold
    python src/08_attention_heatmaps.py --run-dir 04_mil/chile_5fold --overlay \\
        --slides 140-A4_150818 --grid-value log2_enrichment
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "05_attention_maps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A run directory from 07_train_mil.py (contains attention/).")
    parser.add_argument("--attention-dir", type=Path, default=None,
                        help="Override the attention directory (default: <run-dir>/attention).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Default: {DEFAULT_OUTPUT}/<run-dir name>")
    parser.add_argument("--slides", nargs="+", default=None, help="Only these slide ids.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overlay", action="store_true",
                        help="Draw the slide thumbnail under the heatmap (needs openslide).")
    parser.add_argument("--alpha", type=float, default=0.55, help="Heatmap opacity with --overlay.")
    parser.add_argument("--cmap", default=None,
                        help="Matplotlib colormap (default: RdBu_r for log2_enrichment, "
                             "magma otherwise).")
    parser.add_argument("--grid-value",
                        choices=("log2_enrichment", "attention", "percentile", "contribution"),
                        default="log2_enrichment", help="What the PNG and the grid CSV hold.")
    parser.add_argument("--clip-percentile", type=float, default=99.0,
                        help="Colour scale saturates at this percentile of |value| (0 disables).")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-grid-csv", action="store_true", help="Skip the matrix CSV.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo slides whose heatmap already exists.")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if not 0.0 <= args.clip_percentile <= 100.0:
        parser.error("--clip-percentile must be in [0, 100]")
    return args


def lattice(coords: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Level-0 coordinates -> integer (col, row) lattice indices."""
    if step <= 0:
        raise ValueError(f"invalid patch_size_level0: {step}")
    remainder = coords % step
    if remainder.any():
        off = int((remainder.any(axis=1)).sum())
        print(f"  note: {off} coordinates are not multiples of {step}; rounding to the lattice",
              file=sys.stderr)
    cols = np.rint(coords[:, 0] / step).astype(int)
    rows = np.rint(coords[:, 1] / step).astype(int)
    return cols, rows


def patch_table(coords: np.ndarray, attention: np.ndarray, step: int,
                contributions: np.ndarray | None = None) -> pd.DataFrame:
    cols, rows = lattice(coords, step)
    n = len(attention)
    # Attention is >= 0 and sums to 1; guard the log for exact zeros.
    floor = np.finfo(np.float32).tiny
    enrichment = np.log2(np.maximum(attention, floor) * n)
    order = pd.Series(attention).rank(ascending=False, method="min").astype(int)
    table = pd.DataFrame({
        "x": coords[:, 0].astype(int),
        "y": coords[:, 1].astype(int),
        "col": cols,
        "row": rows,
        "attention": attention.astype(np.float64),
        "log2_enrichment": enrichment,
        "percentile": pd.Series(attention).rank(pct=True).to_numpy(),
        "rank": order.to_numpy(),
    })
    if contributions is not None:
        # Signed additive logit terms; they sum to the slide logit, so unlike
        # attention they are already comparable across slides and need no
        # enrichment rescaling. Rank by magnitude, since a strongly inhibitory
        # patch is as interesting as a strongly excitatory one.
        table["contribution"] = contributions.astype(np.float64)
        table["contribution_abs"] = np.abs(contributions)
        table["contribution_rank"] = (
            pd.Series(np.abs(contributions)).rank(ascending=False, method="min").astype(int)
        )
    return table.sort_values("rank").reset_index(drop=True)


def to_grid(table: pd.DataFrame, value: str) -> tuple[np.ndarray, int, int]:
    """Dense (row, col) array of `value`, NaN where no patch was tiled."""
    n_cols = int(table["col"].max()) + 1
    n_rows = int(table["row"].max()) + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    grid[table["row"].to_numpy(), table["col"].to_numpy()] = table[value].to_numpy()
    return grid, n_rows, n_cols


def colour_limits(values: np.ndarray, grid_value: str, clip: float) -> tuple[float, float, bool]:
    """(vmin, vmax, diverging). log2 enrichment is centred on 0 = average patch."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0, False
    if grid_value in ("log2_enrichment", "contribution"):
        # Both are signed and meaningful at 0 (average patch / no contribution),
        # so the colour scale has to be symmetric about it.
        span = np.percentile(np.abs(finite), clip) if clip else np.abs(finite).max()
        span = float(max(span, 1e-3))
        return -span, span, True
    if clip:
        return float(np.percentile(finite, 100 - clip)), float(np.percentile(finite, clip)), False
    return float(finite.min()), float(finite.max()), False


def thumbnail(attrs: dict, n_cols: int, n_rows: int) -> np.ndarray | None:
    """Slide thumbnail resampled onto the patch lattice, or None if unavailable."""
    slide_path = attrs.get("slide_path")
    if not slide_path or not Path(str(slide_path)).exists():
        print(f"  overlay skipped: slide not found at {slide_path}", file=sys.stderr)
        return None
    try:
        import openslide
    except ImportError:
        print("  overlay skipped: openslide is not installed", file=sys.stderr)
        return None
    try:
        with openslide.OpenSlide(str(slide_path)) as slide:
            # Ask for a thumbnail a few patches wide per lattice cell so the tissue
            # stays legible under a semi-transparent heatmap.
            image = slide.get_thumbnail((n_cols * 4, n_rows * 4)).convert("RGB")
        return np.asarray(image)
    except Exception as exc:
        print(f"  overlay skipped: {exc}", file=sys.stderr)
        return None


def render(grid: np.ndarray, attrs: dict, args: argparse.Namespace, out_path: Path,
           n_rows: int, n_cols: int) -> None:
    vmin, vmax, diverging = colour_limits(grid, args.grid_value, args.clip_percentile)
    cmap = args.cmap or ("RdBu_r" if diverging else "magma")
    masked = np.ma.masked_invalid(grid)

    background = thumbnail(attrs, n_cols, n_rows) if args.overlay else None
    # Match the figure to the lattice aspect ratio so the slide fills the canvas
    # instead of floating in white space.
    long_edge = 11.0
    if n_cols >= n_rows:
        width, height = long_edge, max(2.5, long_edge * n_rows / max(n_cols, 1))
    else:
        width, height = max(2.5, long_edge * n_cols / max(n_rows, 1)), long_edge
    figure, axis = plt.subplots(figsize=(width + 1.5, height), dpi=args.dpi)

    extent = (0, n_cols, n_rows, 0)  # lattice units, origin top-left like the slide
    if background is not None:
        axis.imshow(background, extent=extent, interpolation="bilinear")
    image = axis.imshow(masked, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="nearest",
                        alpha=args.alpha if background is not None else 1.0)
    axis.set_xticks([])
    axis.set_yticks([])

    label = {
        "log2_enrichment": "log2(attention x n patches)   0 = average patch",
        "attention": "attention weight",
        "percentile": "attention percentile",
        "contribution": "additive logit contribution   >0 excitatory, <0 inhibitory",
    }[args.grid_value]
    bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    bar.set_label(label)

    prob = attrs.get("prob")
    truth = attrs.get("label")
    parts = [str(attrs.get("slide_id", out_path.stem))]
    if truth is not None:
        parts.append(f"EBV {'+' if int(truth) == 1 else '-'}")
    if prob is not None:
        parts.append(f"p(EBV+) = {float(prob):.3f}")
    parts.append(f"{int(attrs.get('n_patches', np.isfinite(grid).sum()))} patches")
    if "fold" in attrs:
        parts.append(f"fold {int(attrs['fold'])}")
    axis.set_title("   |   ".join(parts), fontsize=10)

    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def process(path: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    slide_id = path.stem
    heatmap_path = output_dir / f"{slide_id}_attention_heatmap.png"
    if heatmap_path.exists() and not args.overwrite:
        return {"slide_id": slide_id, "status": "skipped", "message": "exists"}

    with h5py.File(path, "r") as handle:
        coords = handle["coords"][:]
        attention = handle["attention"][:]
        # Only Additive MIL runs write this; attention-MIL runs have no per-patch
        # decomposition to store.
        contributions = handle["contributions"][:] if "contributions" in handle else None
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in handle.attrs.items()}
    if len(coords) != len(attention):
        raise ValueError(f"{len(coords)} coordinates vs {len(attention)} weights")
    if args.grid_value == "contribution" and contributions is None:
        raise ValueError(
            "--grid-value contribution needs an Additive MIL run: this attention file has no "
            "'contributions' dataset. Retrain with 07_train_mil.py --arch additive."
        )

    step = int(attrs.get("patch_size_level0") or 0)
    if step <= 0:
        step = int(attrs.get("patch_size", 256) * float(attrs.get("target_downsample", 1.0)))
    table = patch_table(
        coords, np.asarray(attention, dtype=np.float64), step,
        None if contributions is None else np.asarray(contributions, dtype=np.float64),
    )
    table.to_csv(output_dir / f"{slide_id}_patch_attention.csv", index=False)

    grid, n_rows, n_cols = to_grid(table, args.grid_value)
    if not args.no_grid_csv:
        frame = pd.DataFrame(grid, index=pd.Index(np.arange(n_rows) * step, name="y"),
                             columns=np.arange(n_cols) * step)
        frame.to_csv(output_dir / f"{slide_id}_attention_grid.csv")
    render(grid, attrs, args, heatmap_path, n_rows, n_cols)

    top = table.iloc[0]
    return {
        "slide_id": slide_id,
        "status": "ok",
        "n_patches": len(table),
        "label": attrs.get("label"),
        "prob": attrs.get("prob"),
        "top_x": int(top["x"]),
        "top_y": int(top["y"]),
        "top_log2_enrichment": float(top["log2_enrichment"]),
        "message": "",
    }


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
        print(f"ERROR: no attention directory at {attention_dir}. Run 07_train_mil.py first "
              f"(without --no-attention).", file=sys.stderr)
        return 2
    paths = sorted(attention_dir.glob("*.h5"))
    if args.slides:
        wanted = set(args.slides)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            print(f"WARNING: no attention file for {sorted(missing)}", file=sys.stderr)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"ERROR: no attention HDF5 files matched in {attention_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (DEFAULT_OUTPUT / args.run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} slides -> {output_dir}")

    rows: list[dict[str, object]] = []
    payloads = [(path, args, output_dir) for path in paths]
    if args.workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"[{done}/{len(paths)}] {row['slide_id']} {row['status']}", flush=True)
    else:
        for done, payload in enumerate(payloads, start=1):
            row = _worker(payload)
            rows.append(row)
            print(f"[{done}/{len(paths)}] {row['slide_id']} {row['status']}", flush=True)

    summary = pd.DataFrame(rows).sort_values("slide_id")
    summary.to_csv(output_dir / "summary.csv", index=False)
    failed = int((summary["status"] == "failed").sum())
    print(f"\nwrote {output_dir} ({len(summary)} slides, {failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
