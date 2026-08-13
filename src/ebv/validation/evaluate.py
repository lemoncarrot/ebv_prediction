#!/usr/bin/env python3
"""Confusion matrices, threshold sweeps, and bootstrap CIs for an ebv.mil.train run.

`metrics.json` reports rates (sensitivity, specificity, AUPRC). This turns them
back into counts, puts AUPRC next to the only baseline that makes it readable,
and attaches confidence intervals -- which matter here because a cohort with 22
positives cannot support three-decimal AUROC claims.

Two things this is built to make visible:

*   **AUPRC has a floor, and the floor is the prevalence.** A no-skill classifier
    scores AUPRC = P(positive), so 0.53 at 8% prevalence and 0.78 at 17%
    prevalence are not directly comparable, and neither is "low" on its own. The
    lift column divides by that floor. AUROC has a fixed 0.5 baseline and does
    not have this problem, which is why the two metrics disagree so often on
    imbalanced cohorts.

*   **A threshold does not transfer across cohorts for free.** `ebv.mil.train`
    picks the Youden point on held-out training-cohort data and reuses it for
    `--external-cohort`. If the two cohorts differ in prevalence or calibration,
    that point lands in the wrong place and sensitivity collapses even when the
    ranking (AUROC) is fine. The sweep prints what the same scores would give at
    other operating points, so the loss attributable to the threshold alone is
    separable from the loss attributable to the model.

Usage:

    python -m ebv.validation.evaluate --run-dir 04_mil/tcga_train_chile_ext
    python -m ebv.validation.evaluate --run-dir 04_mil/pooled_5fold --by-cohort
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A run directory from ebv.mil.train.")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the CIs (0 disables).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by-cohort", action="store_true",
                        help="Also break the out-of-fold predictions down per cohort.")
    parser.add_argument("--sweep", type=int, default=9,
                        help="Rows in the operating-point sweep (0 disables).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the summary as JSON here (default: <run-dir>/evaluation.json).")
    return parser.parse_args()


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = scores >= threshold
    positive = labels == 1
    return {
        "tp": int((predicted & positive).sum()),
        "fn": int((~predicted & positive).sum()),
        "fp": int((predicted & ~positive).sum()),
        "tn": int((~predicted & ~positive).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    total = tp + fn + fp + tn
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_accuracy": (
            0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) and (tn + fp) else float("nan")
        ),
    }


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray, resamples: int,
                 seed: int) -> dict[str, tuple[float, float]]:
    """Percentile CIs for AUROC and AUPRC, resampling slides with replacement."""
    if resamples <= 0 or len(np.unique(labels)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue  # a resample with one class has no defined AUROC
        aurocs.append(roc_auc_score(labels[index], scores[index]))
        auprcs.append(average_precision_score(labels[index], scores[index]))
    if not aurocs:
        return {}
    return {
        "auroc": (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))),
        "auprc": (float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))),
    }


def matrix_block(name: str, labels: np.ndarray, scores: np.ndarray, threshold: float,
                 args: argparse.Namespace) -> dict[str, object]:
    n, positives = len(labels), int((labels == 1).sum())
    prevalence = positives / n if n else float("nan")
    counts = confusion(labels, scores, threshold)
    rate = rates(counts)
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    auprc = (float(average_precision_score(labels, scores))
             if len(np.unique(labels)) > 1 else float("nan"))
    ci = bootstrap_ci(labels, scores, args.bootstrap, args.seed)

    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    print(f"{n} slides, {positives} positive ({100 * prevalence:.1f}% prevalence), "
          f"threshold {threshold:.4f}")

    print(f"\n                 predicted +   predicted -        total")
    print(f"  actual +   {counts['tp']:>11}   {counts['fn']:>11}   {positives:>10}")
    print(f"  actual -   {counts['fp']:>11}   {counts['tn']:>11}   {n - positives:>10}")
    print(f"  total      {counts['tp'] + counts['fp']:>11}   "
          f"{counts['fn'] + counts['tn']:>11}   {n:>10}")

    auroc_ci = f"  95% CI [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}]" if ci else ""
    auprc_ci = f"  95% CI [{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}]" if ci else ""
    print(f"\n  AUROC              {auroc:.3f}{auroc_ci}   (no-skill 0.500)")
    print(f"  AUPRC              {auprc:.3f}{auprc_ci}   (no-skill {prevalence:.3f} "
          f"= prevalence, so lift = {auprc / prevalence:.1f}x)")
    print(f"\n  sensitivity        {rate['sensitivity']:.3f}   "
          f"({counts['tp']}/{positives} positives found)")
    print(f"  specificity        {rate['specificity']:.3f}   "
          f"({counts['tn']}/{n - positives} negatives cleared)")
    print(f"  PPV (precision)    {rate['ppv']:.3f}   "
          f"({counts['tp']}/{counts['tp'] + counts['fp']} flagged are truly positive)")
    print(f"  NPV                {rate['npv']:.3f}")
    print(f"  F1                 {rate['f1']:.3f}")
    print(f"  balanced accuracy  {rate['balanced_accuracy']:.3f}")
    print(f"  accuracy           {rate['accuracy']:.3f}   "
          f"(predicting all-negative would give {1 - prevalence:.3f})")

    return {"n": n, "n_pos": positives, "prevalence": prevalence, "threshold": float(threshold),
            "counts": counts, "rates": rate, "auroc": auroc, "auprc": auprc,
            "auprc_lift": auprc / prevalence if prevalence else float("nan"), "ci": ci}


def sweep(name: str, labels: np.ndarray, scores: np.ndarray, applied: float, rows: int) -> None:
    """What the same scores would give at other operating points."""
    if rows <= 0 or len(np.unique(labels)) < 2:
        return
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = float(thresholds[int(np.argmax(tpr - fpr))])
    candidates = sorted({float(np.quantile(scores, q)) for q in np.linspace(0.5, 0.99, rows)}
                        | {applied, youden})
    print(f"\n  operating points for {name} (same scores, different cut):")
    print(f"    {'threshold':>10} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} "
          f"{'sens':>6} {'spec':>6} {'PPV':>6} {'F1':>6}")
    for threshold in candidates:
        counts = confusion(labels, scores, threshold)
        rate = rates(counts)
        tag = ""
        if abs(threshold - applied) < 1e-12:
            tag = "  <- applied"
        if abs(threshold - youden) < 1e-12:
            tag += "  <- Youden on this cohort"
        print(f"    {threshold:>10.4f} {counts['tp']:>4} {counts['fn']:>4} {counts['fp']:>4} "
              f"{counts['tn']:>4} {rate['sensitivity']:>6.3f} {rate['specificity']:>6.3f} "
              f"{rate['ppv']:>6.3f} {rate['f1']:>6.3f}{tag}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: no metrics.json in {run_dir}; has the run finished?", file=sys.stderr)
        return 2
    metrics = json.loads(metrics_path.read_text())
    config = json.loads((run_dir / "config.json").read_text())
    threshold = float(metrics["oof"]["slide"]["threshold"])

    print(f"run: {run_dir.name}    arch: {config.get('arch', 'gated_attention')}    "
          f"{metrics.get('n_folds')}-fold CV")
    print(f"trained on: {config.get('cohort') or 'all'}"
          f"    external: {config.get('external_cohort') or 'none'}"
          f"    balance: {config.get('balance', 'none')}"
          f"    pos_weight: {config.get('pos_weight') if config.get('pos_weight') is not None else 'auto'}")

    summary: dict[str, object] = {"run": run_dir.name, "threshold": threshold}

    # Out-of-fold predictions are where the threshold came from, not a test set.
    internal_name = "OUT-OF-FOLD (cross-validated on the training cohort)"
    internal = pd.read_csv(run_dir / "oof_predictions.csv")
    labels, scores = internal["label"].to_numpy(), internal["prob"].to_numpy()
    summary["internal"] = matrix_block(internal_name, labels, scores, threshold, args)
    sweep("out-of-fold", labels, scores, threshold, args.sweep)

    if args.by_cohort and internal["cohort"].nunique() > 1:
        for cohort, group in internal.groupby("cohort"):
            summary[f"internal_{cohort}"] = matrix_block(
                f"{internal_name.split(' (')[0]}, {cohort} slides only", group["label"].to_numpy(),
                group["prob"].to_numpy(), threshold, args)

    external_path = run_dir / "external_predictions.csv"
    if external_path.exists():
        external = pd.read_csv(external_path)
        cohorts = "+".join(sorted(external["cohort"].unique()))
        labels, scores = external["label"].to_numpy(), external["prob"].to_numpy()
        summary["external"] = matrix_block(
            f"EXTERNAL COHORT ({cohorts}), never trained on, fold-ensembled",
            labels, scores, threshold, args)
        sweep("external", labels, scores, threshold, args.sweep)

    output = args.output or (run_dir / "evaluation.json")
    output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
