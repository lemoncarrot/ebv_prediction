#!/usr/bin/env python3
"""Plot confusion matrices for the final model on the external split of a train.py run.

    python confusion_matrix.py --run-dir ../../results/tcga2chile_12345

--folds instead draws one matrix per CV fold (out-of-fold predictions.csv).

The final model is trained on all training slides and scores each external
slide once. When the external set mixes cohorts, each cohort gets its own panel
next to the combined one. Writes confusion_matrix.png next to
external_predictions.csv and prints the counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

INK, MUTED = "#0b0b0b", "#898781"
# Sequential single hue (the fold-0 blue from auc_curves.py), light -> dark.
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f7fc", "#2a78d6", "#0d3a73"])
CLASSES = ["EBV−", "EBV+"]


def draw(ax, cm: np.ndarray, title: str, scale: float = 1.0) -> None:
    """Color by row-normalized rate so the rare positive class stays readable.
    `scale` multiplies every font size (e.g. 1.5 for slides)."""
    rates = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rates, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if rates[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n{rates[i, j]:.0%}", ha="center", va="center",
                    color=color, fontsize=12 * scale)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=10 * scale)
    ax.set_ylabel("True", color=MUTED, fontsize=10 * scale)
    ax.set_title(title, fontsize=10 * scale)
    ax.tick_params(colors=MUTED, labelsize=9 * scale, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def plot_panels(panels: list[tuple[str, pd.DataFrame]], threshold: float, suptitle: str,
                out: Path) -> None:
    """One confusion matrix per (title, predictions) panel, side by side."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    axes = np.atleast_1d(axes)

    for ax, (title, frame) in zip(axes, panels):
        pred = (frame["prob"] >= threshold).astype(int)
        cm = confusion_matrix(frame["label"], pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        draw(ax, cm, f"{title}, n={len(frame)}")
        print(f"{title}: n={len(frame)} TN={tn} FP={fp} FN={fn} TP={tp} "
              f"sens={ratio(tp, tp + fn):.3f} spec={ratio(tn, tn + fp):.3f} "
              f"acc={(tp + tn) / len(frame):.3f}")

    fig.suptitle(f"{suptitle} (threshold {threshold:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory holding external_predictions.csv.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability cutoff for EBV+ (train.py uses 0.5).")
    parser.add_argument("--folds", action="store_true",
                        help="Instead: one matrix per CV fold from the out-of-fold "
                             "predictions.csv -> fold_confusion_matrices.png.")
    args = parser.parse_args()

    if args.folds:
        oof = pd.read_csv(args.run_dir / "predictions.csv")
        oof = oof[oof["split"] == "test"]
        panels = [(f"fold {fold} test", group) for fold, group in oof.groupby("fold")]
        plot_panels(panels, args.threshold, "CV fold models on their held-out fold",
                    args.run_dir / "fold_confusion_matrices.png")
        return 0

    external = pd.read_csv(args.run_dir / "external_predictions.csv")
    cohorts = external["cohort"].nunique()
    panels = [(f"external: {'all' if cohorts > 1 else external['cohort'].iloc[0]}", external)]
    if cohorts > 1:
        panels += [(f"external: {c}", g) for c, g in external.groupby("cohort")]
    plot_panels(panels, args.threshold, "Final model, confusion matrix",
                args.run_dir / "confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
