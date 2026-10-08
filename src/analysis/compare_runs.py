#!/usr/bin/env python3
"""Compare training variants across seeds: external EBV performance + MSI confound.

    python compare_runs.py --results ../../results --pattern 'tcga2chile_{arm}_s{seed}_*' \
        --arms binary msiw3 3class --seeds 0 1 2

Per run:
  external (final model): AUROC, AUPRC, sensitivity/specificity at 0.5
  TCGA out-of-fold:       AUROC, AUROC EBV+ vs MSI negatives, MSI false-positive
                          rate (definitions as in msi_confound.py)
  3-class runs also:      AUROC of p_msi for MSI vs other negatives (TCGA OOF)
Then mean +/- SD per arm, and the per-seed paired difference vs the first arm
(same seed = same folds, so the pairing removes split noise).

Writes <results>/compare_<first pattern token>.csv (per run) and prints the tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from msi_confound import analyse, tcga_msi  # noqa: E402


def run_row(run_dir: Path, msi: pd.DataFrame, threshold: float) -> dict:
    ext = pd.read_csv(run_dir / "external_predictions.csv")
    called = ext["prob"] >= threshold
    pos, neg = ext["label"] == 1, ext["label"] == 0
    row = {"ext_auroc": roc_auc_score(ext["label"], ext["prob"]),
           "ext_auprc": average_precision_score(ext["label"], ext["prob"]),
           "ext_sens": float((called & pos).sum() / pos.sum()),
           "ext_spec": float((~called & neg).sum() / neg.sum())}

    oof = pd.read_csv(run_dir / "predictions.csv")
    oof = oof[(oof["split"] == "test") & (oof["cohort"] == "tcga")].merge(msi, on="slide_id")
    oof["is_msi"] = oof["is_msi"].fillna(False).astype(bool)
    confound = analyse(run_dir.name, oof, threshold)
    row.update({"oof_auroc": confound["auc_all"],
                "oof_auroc_ebv_vs_msi": confound["auc_ebv_vs_msi"],
                "oof_auroc_ebv_vs_other": confound["auc_ebv_vs_other"],
                "oof_fp_rate_msi": confound["fp_rate_msi"],
                "oof_fp_rate_other": confound["fp_rate_other"]})
    if "p_msi" in oof:
        negatives = oof[oof["label"] == 0]
        row["oof_msi_detection_auroc"] = roc_auc_score(negatives["is_msi"], negatives["p_msi"])
    return row


def main() -> int:
    here = Path(__file__).resolve().parent
    project = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=project / "results")
    p.add_argument("--pattern", required=True, help="glob with {arm} and {seed}")
    p.add_argument("--arms", nargs="+", required=True, help="First arm is the reference.")
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--labels", type=Path, default=project / "metadata/cohort_labels.csv")
    p.add_argument("--merged", type=Path, default=project / "metadata/STAD_merged_table.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    msi = tcga_msi(args.labels, args.merged)
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            matches = sorted(args.results.glob(args.pattern.format(arm=arm, seed=seed)))
            done = [m for m in matches if (m / "external_predictions.csv").exists()]
            if not done:
                print(f"missing: {arm} seed {seed}")
                continue
            if len(done) > 1:
                print(f"note: {arm} seed {seed} has {len(done)} runs; using {done[-1].name}")
            rows.append({"arm": arm, "seed": seed, "run": done[-1].name,
                         **run_row(done[-1], msi, args.threshold)})
    table = pd.DataFrame(rows)
    out = args.results / f"compare_{args.pattern.split('_')[0]}.csv"
    table.to_csv(out, index=False)

    metrics = [c for c in table.columns if c.startswith(("ext_", "oof_"))]
    pd.set_option("display.width", 200)
    print("\nmean (SD) over seeds")
    summary = table.groupby("arm", sort=False)[metrics].agg(["mean", "std"])
    print(pd.DataFrame({m: [f"{summary.loc[a, (m, 'mean')]:.3f} ({summary.loc[a, (m, 'std')]:.3f})"
                            for a in summary.index] for m in metrics},
                       index=summary.index).T.to_string())

    reference = args.arms[0]
    print(f"\npaired difference vs {reference} (same seed), mean [min, max] over seeds")
    base = table[table["arm"] == reference].set_index("seed")[metrics]
    for arm in args.arms[1:]:
        diff = table[table["arm"] == arm].set_index("seed")[metrics] - base
        diff = diff.dropna(how="all")
        print(f"  {arm}:")
        for m in metrics:
            d = diff[m].dropna()
            if len(d):
                print(f"    {m:<26} {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}]")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
