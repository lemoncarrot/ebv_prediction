#!/usr/bin/env python3
"""Lab-meeting slide figures: one PR curve + confusion matrix per experiment,
plus a seed-level summary of the TCGA -> Chile variants.

    python slide_figures.py --out-dir ../../results/lab_meeting_2026-10

Each experiment figure (PNG at 300 dpi + PDF) shows the final model on that
experiment's test set: precision-recall on the left, confusion matrix at 0.5 on
the right. No titles or captions -- those belong on the slide. Experiments with
several seeds draw seed 0 (methods share folds within a seed); every seed's
numbers go to results_table.csv.
The summary slide compares only the TCGA -> Chile variants: the mixed
experiment is tested on a different, smaller held-out set, so its numbers are
not comparable and it stays on its own slide.
Also writes results_table.csv (one row per run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from auc_curves import FOLD_COLORS, INK, MUTED, tidy
from confusion_matrix import draw

SECONDARY = "#52514e"
THRESHOLD = 0.5
SEEDS = (0, 1, 2)

# key -> output file. `pattern` is a results/ glob with {seed}; `runs` lists fixed run dirs.
EXPERIMENTS = [
    {"key": "01_tcga_to_chile_baseline",
     "pattern": "tcga2chile_binary_s{seed}_*"},
    {"key": "02_mixed_heldout",
     "runs": ["mixed_45065873"]},
    {"key": "03_tcga_to_chile_3class",
     "pattern": "tcga2chile_3class_s{seed}_*"},
    {"key": "04_tcga_to_chile_3class_msiw",
     "pattern": "tcga2chile_3class_msiw3_s{seed}_*"},
]
# summary rows, top to bottom; the weighted-binary run is shown as a reference
SUMMARY_ARMS = [
    ("binary", "Binary (baseline)", "tcga2chile_binary_s{seed}_*"),
    ("msiw3", "Binary + MSI weight ×3", "tcga2chile_msiw3_s{seed}_*"),
    ("3class", "3-class", "tcga2chile_3class_s{seed}_*"),
    ("3class_msiw3", "3-class + MSI weight ×3", "tcga2chile_3class_msiw3_s{seed}_*"),
]


def find_runs(results: Path, pattern: str) -> dict[int, Path]:
    """seed -> finished run dir (latest job id if a seed was run twice)."""
    found = {}
    for seed in SEEDS:
        done = [d for d in sorted(results.glob(pattern.format(seed=seed)))
                if (d / "external_predictions.csv").exists()]
        if done:
            found[seed] = done[-1]
    return found


def run_metrics(ext: pd.DataFrame) -> dict:
    called = ext["prob"] >= THRESHOLD
    pos, neg = ext["label"] == 1, ext["label"] == 0
    return {"n": len(ext), "pos": int(pos.sum()),
            "auprc": average_precision_score(ext["label"], ext["prob"]),
            "auroc": roc_auc_score(ext["label"], ext["prob"]),
            "sens": float((called & pos).sum() / pos.sum()),
            "spec": float((~called & neg).sum() / neg.sum())}


def pr_panel(ax, ext: pd.DataFrame) -> None:
    """Precision-recall on the test set; per-cohort curves when it is mixed.
    Dotted line in each curve's colour = its chance level (EBV+ rate)."""
    cohorts = sorted(ext["cohort"].unique())
    curves = [(cohort_name(cohorts[0]), ext, FOLD_COLORS[0])]
    if len(cohorts) > 1:
        curves = [("All", ext, INK)] + [
            (cohort_name(c), g, FOLD_COLORS[i])
            for i, (c, g) in enumerate(ext.groupby("cohort"))]
    for name, frame, color in curves:
        precision, recall, _ = precision_recall_curve(frame["label"], frame["prob"])
        ap = average_precision_score(frame["label"], frame["prob"])
        ax.step(recall, precision, where="post", color=color, linewidth=2.5,
                label=f"{name}  AUPRC {ap:.2f}")
        ax.axhline(frame["label"].mean(), color=color, linewidth=1.2, linestyle=":")
    ax.plot([], [], color=MUTED, linestyle=":", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.03))
    ax.set_xlabel("Recall", fontsize=14, color=SECONDARY)
    ax.set_ylabel("Precision", fontsize=14, color=SECONDARY)
    ax.legend(loc="center left", fontsize=13, frameon=False, labelcolor=INK)
    tidy(ax)
    ax.tick_params(labelsize=12)


def cohort_name(cohort: str) -> str:
    return cohort.upper() if cohort == "tcga" else cohort.capitalize()


def experiment_slide(exp: dict, runs: dict[int, Path], out_dir: Path) -> list[dict]:
    """PR curve + confusion matrix for one run (the lowest seed); every seed's
    numbers go to the returned rows for results_table.csv."""
    rows = []
    for seed, run_dir in runs.items():
        ext = pd.read_csv(run_dir / "external_predictions.csv")
        rows.append({"experiment": exp["key"], "seed": seed, "run": run_dir.name,
                     **run_metrics(ext)})
    ext = pd.read_csv(runs[min(runs)] / "external_predictions.csv")

    fig = plt.figure(figsize=(12, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=0.07, right=0.98,
                            top=0.96, bottom=0.13, wspace=0.28)
    pr_panel(fig.add_subplot(grid[0]), ext)
    cm = confusion_matrix(ext["label"], (ext["prob"] >= THRESHOLD).astype(int), labels=[0, 1])
    draw(fig.add_subplot(grid[1]), cm, "", scale=1.5)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{exp['key']}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {exp['key']}  (seed {min(runs)} shown, {len(runs)} run(s) in table)")
    return rows


def summary_slide(results: Path, out_dir: Path) -> pd.DataFrame:
    """Seed points + mean per variant, one panel per metric (Chile test set)."""
    rows = []
    for arm, label, pattern in SUMMARY_ARMS:
        for seed, run_dir in find_runs(results, pattern).items():
            ext = pd.read_csv(run_dir / "external_predictions.csv")
            rows.append({"arm": arm, "label": label, "seed": seed, **run_metrics(ext)})
    table = pd.DataFrame(rows)
    labels = [label for arm, label, _ in SUMMARY_ARMS if arm in set(table["arm"])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                             gridspec_kw={"left": 0.235, "right": 0.97, "top": 0.88,
                                          "bottom": 0.12, "wspace": 0.12})
    for ax, (metric, name) in zip(axes, [("auprc", "AUPRC"), ("auroc", "AUROC")]):
        values = table[metric]
        span = (values.min() - 0.03, values.max() + 0.045)
        for y, label in enumerate(labels):
            pts = table.loc[table["label"] == label, metric]
            ax.scatter(pts, [y] * len(pts), s=90, color=FOLD_COLORS[0], edgecolor="white",
                       linewidth=2, zorder=3, label="Seed" if y == 0 else None)
            ax.scatter([pts.mean()], [y], marker="|", s=900, color=INK, linewidth=3,
                       zorder=4, label="Mean" if y == 0 else None)
            ax.text(span[1] - 0.004, y, f"{pts.mean():.3f}", ha="right", va="center",
                    fontsize=14, color=INK)
        ax.set_xlim(span)
        ax.set_title(name, fontsize=16, loc="left", color=INK)
        tidy(ax)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=12)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=14, color=INK)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=12,
               frameon=False, labelcolor=INK, bbox_to_anchor=(0.97, 1.0))
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"05_summary_tcga_to_chile.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote 05_summary_tcga_to_chile")
    return table


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=here.parent.parent / "results")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        runs = ({0: args.results / exp["runs"][0]} if "runs" in exp
                else find_runs(args.results, exp["pattern"]))
        if not runs:
            print(f"skipped {exp['key']}: no finished runs")
            continue
        rows += experiment_slide(exp, runs, args.out_dir)
    summary_slide(args.results, args.out_dir)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "results_table.csv", index=False, float_format="%.4f")
    print(f"wrote {args.out_dir}/results_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
