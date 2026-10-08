#!/usr/bin/env python3
"""Plot the final model's PR curve on the external split, plus a per-fold CV summary.

    python auc_curves.py --run-dir ../../results/tcga2chile_12345

pr_curves.png: the final (all-training-data) model scored on the external set,
with one curve per cohort when the external set mixes cohorts.
fold_metrics.png: per-fold inner-validation and out-of-fold test AUC from CV --
a view of training, not the headline result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pr_figure(external: pd.DataFrame, out: Path) -> None:
    """Final-model precision-recall on the external set; per-cohort curves when it
    is mixed. Each dotted line is that curve's chance level (its positive rate)."""
    fig, ax = plt.subplots(figsize=(5, 5))

    def curve(frame: pd.DataFrame, name: str, color: str, width: float) -> None:
        precision, recall, _ = precision_recall_curve(frame["label"], frame["prob"])
        ap = average_precision_score(frame["label"], frame["prob"])
        # steps match how average precision integrates the curve
        ax.step(recall, precision, where="post", color=color, linewidth=width,
                label=f"{name} (n={len(frame)}, pos={int(frame['label'].sum())})  "
                      f"AUPRC {ap:.3f}")
        ax.axhline(frame["label"].mean(), color=color, linewidth=1, linestyle=":")

    cohorts = sorted(external["cohort"].unique())
    if len(cohorts) > 1:
        for i, (cohort, group) in enumerate(external.groupby("cohort")):
            curve(group, cohort, FOLD_COLORS[i % len(FOLD_COLORS)], 2)
    curve(external, "all" if len(cohorts) > 1 else cohorts[0], INK, 2.5)

    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (sensitivity)",
           ylabel="Precision (PPV)", title="Precision-recall, final model on external set")
    ax.legend(loc="best", fontsize=8, frameon=False)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def metrics_figure(folds: pd.DataFrame, out: Path) -> None:
    """Per-fold inner-val vs out-of-fold test AUC as grouped bars -- the CV spread."""
    columns = {"best_val_auc": "inner val", "test_auc": "CV test"}
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    width = 0.8 / len(columns)

    for i, (column, label) in enumerate(columns.items()):
        offset = (i - (len(columns) - 1) / 2) * width
        ax.bar(folds["fold"] + offset, folds[column], width * 0.94,
               color=FOLD_COLORS[i], label=label)
        mean = folds[column].mean()
        ax.axhline(mean, color=FOLD_COLORS[i], linewidth=1, linestyle="--")
        ax.text(folds["fold"].max() + 0.5, mean, f"mean {mean:.3f}",
                color=FOLD_COLORS[i], fontsize=8, va="center")

    low = min(0.5, float(folds[list(columns)].min().min()) - 0.05)
    ax.set(ylim=(low, 1.0), xticks=folds["fold"], xlabel="fold", ylabel="AUC",
           title="Cross-validation AUC per fold")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory holding external_predictions.csv and folds.csv.")
    args = parser.parse_args()

    pr_figure(pd.read_csv(args.run_dir / "external_predictions.csv"),
              args.run_dir / "pr_curves.png")
    metrics_figure(pd.read_csv(args.run_dir / "folds.csv"), args.run_dir / "fold_metrics.png")
    print(f"wrote {args.run_dir}/pr_curves.png, fold_metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
