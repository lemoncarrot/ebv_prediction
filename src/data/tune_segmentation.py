#!/usr/bin/env python3
"""Compare tissue-segmentation settings side by side on a few slides.

    python tune_segmentation.py --slides a.bif b.bif --out-dir ../../results/seg_tuning \
        --sat 0 10 15 20 --min-area 4e6 2e5 --max-dark-value 60

One PNG per slide: a grid of thumbnails, tissue tinted green, one panel per
(saturation threshold, minimum piece area); sat 0 means Otsu (the pipeline
default). The first panel is always the pipeline default (Otsu, 4 mm^2, no dark
filter). Each panel title gives the tissue area and the approximate number of
256 px / 0.5 um tiles, to pick run_pipeline.py flags before a cohort run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SlideReaderClass import SlideReader  # noqa: E402
from TissueSegmentation import segment_tissue  # noqa: E402

TILE_UM = 128.0                           # 256 px at 0.5 um/px


def panel(ax, thumb: np.ndarray, mask: np.ndarray, thumb_mpp: float, title: str) -> None:
    shown = thumb.copy()
    shown[mask] = (0.55 * shown[mask] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    ax.imshow(shown)
    area_mm2 = mask.sum() * thumb_mpp ** 2 / 1e6
    ax.set_title(f"{title}\n{area_mm2:.0f} mm², ~{area_mm2 * 1e6 / TILE_UM ** 2:,.0f} tiles",
                 fontsize=9)
    ax.axis("off")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slides", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sat", type=int, nargs="+", default=[0, 10, 15, 20],
                   help="Saturation thresholds; 0 = Otsu.")
    p.add_argument("--min-area", type=float, nargs="+", default=[4e6, 2e5],
                   help="Minimum tissue piece, um^2.")
    p.add_argument("--max-dark-value", type=int, default=60)
    p.add_argument("--max-dim", type=int, default=4096)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for slide in args.slides:
        with SlideReader(str(slide), mpp_target=0.5, tile_size=256) as reader:
            thumb, ds = reader.thumbnail(max_dim=args.max_dim)
            thumb_mpp = reader.geometry.mpp_base * ds
        sat = cv2.medianBlur(cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[..., 1], 7)
        otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        settings = [("default: Otsu, 4 mm²", {})]
        for s in args.sat:
            for a in args.min_area:
                params = {"min_tissue_um2": a, "max_dark_value": args.max_dark_value}
                if s:
                    params["sat_thresh"] = s
                settings.append((f"sat {'Otsu' if not s else s}, min {a / 1e6:g} mm², "
                                 f"dark≤{args.max_dark_value}", params))
        cols = 1 + len(args.min_area)
        rows = -(-len(settings) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 6.5 * rows))
        for ax, (title, params) in zip(axes.ravel(), settings):
            panel(ax, thumb, segment_tissue(thumb, thumb_mpp=thumb_mpp, **params), thumb_mpp, title)
        for ax in axes.ravel()[len(settings):]:
            ax.axis("off")
        fig.suptitle(f"{slide.stem}  (Otsu threshold = {otsu:.0f})", fontsize=11)
        fig.tight_layout()
        out = args.out_dir / f"{slide.stem}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"{slide.stem}: Otsu {otsu:.0f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
