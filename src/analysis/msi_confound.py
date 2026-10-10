#!/usr/bin/env python3
"""Do MSI-high tumors drive EBV false positives? TCGA slides only.

    python msi_confound.py --run chile2tcga=../../results/chile2tcga_45065817:external \
                           --run tcga2chile_oof=../../results/tcga2chile_45065815:test

Each --run is NAME=RUN_DIR:SPLIT, where SPLIT "external" reads
external_predictions.csv and "test" reads the out-of-fold predictions.csv.
MSI comes from two sources: the TCGA molecular subtype (cohort_labels.csv) and
the MSI/MSS call in STAD_merged_table.csv. Chile has no MSI annotation.

Per run it reports, among EBV-negative slides:
  - predicted-positive rate at the threshold, MSI vs non-MSI (Fisher exact)
  - the MSI share of false positives vs of all negatives
  - AUROC of EBV+ against MSI negatives vs against other negatives
Writes one summary row per run to --out (default results/msi_confound.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score


def tcga_msi(labels_csv: Path, merged_csv: Path) -> pd.DataFrame:
    """slide_id -> subtype, MSI call, and an MSI flag from either source."""
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["cohort"] == "tcga"][["slide_id", "sample_id", "tcga_subtype"]]
    merged = pd.read_csv(merged_csv, usecols=["Sample", "MSI"]).drop_duplicates("Sample")
    frame = labels.merge(merged, left_on="sample_id", right_on="Sample", how="left")
    frame["is_msi"] = (frame["tcga_subtype"].eq("STAD_MSI")
                       | frame["MSI"].astype(str).str.upper().str.startswith("MSI"))
    frame["msi_known"] = frame["tcga_subtype"].notna() | frame["MSI"].notna()
    return frame[["slide_id", "tcga_subtype", "MSI", "is_msi", "msi_known"]]


def analyse(name: str, preds: pd.DataFrame, threshold: float) -> dict:
    neg = preds[preds["label"] == 0]
    pos = preds[preds["label"] == 1]
    msi, other = neg[neg["is_msi"]], neg[~neg["is_msi"]]
    called = neg["prob"] >= threshold
    table = [[int((msi["prob"] >= threshold).sum()), int((msi["prob"] < threshold).sum())],
             [int((other["prob"] >= threshold).sum()), int((other["prob"] < threshold).sum())]]
    odds, p_fisher = fisher_exact(table, alternative="greater")

    def auc_vs(negatives: pd.DataFrame) -> float:
        both = pd.concat([pos, negatives])
        return float(roc_auc_score(both["label"], both["prob"])) if len(negatives) else np.nan

    return {
        "run": name, "n": len(preds), "ebv_pos": len(pos),
        "neg_msi": len(msi), "neg_other": len(other),
        "fp_total": int(called.sum()),
        "fp_msi": table[0][0], "fp_other": table[1][0],
        "fp_rate_msi": table[0][0] / max(len(msi), 1),
        "fp_rate_other": table[1][0] / max(len(other), 1),
        "msi_share_of_negatives": len(msi) / max(len(neg), 1),
        "msi_share_of_fp": table[0][0] / max(int(called.sum()), 1),
        "fisher_odds": odds, "fisher_p": p_fisher,
        "median_prob_msi_neg": float(msi["prob"].median()),
        "median_prob_other_neg": float(other["prob"].median()),
        "median_prob_ebv_pos": float(pos["prob"].median()),
        "mwu_p_msi_gt_other": float(mannwhitneyu(msi["prob"], other["prob"],
                                                 alternative="greater").pvalue),
        "auc_ebv_vs_msi": auc_vs(msi), "auc_ebv_vs_other": auc_vs(other),
        "auc_all": float(roc_auc_score(preds["label"], preds["prob"])),
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    project = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR:SPLIT")
    p.add_argument("--labels", type=Path, default=project / "metadata/cohort_labels.csv")
    p.add_argument("--merged", type=Path, default=project / "metadata/STAD_merged_table.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=project / "results/msi_confound.csv")
    args = p.parse_args()

    msi = tcga_msi(args.labels, args.merged)
    rows = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        run_dir, split = rest.rsplit(":", 1)
        file = "external_predictions.csv" if split == "external" else "predictions.csv"
        preds = pd.read_csv(Path(run_dir) / file)
        preds = preds[(preds["split"] == split) & (preds["cohort"] == "tcga")]
        preds = preds.merge(msi, on="slide_id", how="left")
        unknown = int((~preds["msi_known"].fillna(False)).sum())
        preds["is_msi"] = preds["is_msi"].fillna(False).astype(bool)
        print(f"\n=== {name}: {len(preds)} TCGA slides ({unknown} without MSI annotation, "
              f"counted as non-MSI)")
        print("by subtype (n, EBV+, median prob, predicted-positive at "
              f"{args.threshold:g}):")
        by = preds.assign(called=preds["prob"] >= args.threshold).groupby(
            preds["tcga_subtype"].fillna("unknown"))
        print(pd.DataFrame({"n": by.size(), "ebv_pos": by["label"].sum(),
                            "median_prob": by["prob"].median().round(3),
                            "called_pos": by["called"].sum()}).to_string())
        row = analyse(name, preds, args.threshold)
        rows.append(row)
        print(f"FP: {row['fp_total']} total, {row['fp_msi']} MSI "
              f"({row['msi_share_of_fp']:.0%} of FPs vs {row['msi_share_of_negatives']:.0%} "
              f"of negatives); FP rate MSI {row['fp_rate_msi']:.0%} vs other "
              f"{row['fp_rate_other']:.0%}, Fisher p={row['fisher_p']:.2g}")
        print(f"median prob: EBV+ {row['median_prob_ebv_pos']:.3f}, MSI neg "
              f"{row['median_prob_msi_neg']:.3f}, other neg {row['median_prob_other_neg']:.3f} "
              f"(MWU MSI>other p={row['mwu_p_msi_gt_other']:.2g})")
        print(f"AUROC EBV+ vs MSI negatives {row['auc_ebv_vs_msi']:.3f} | vs other negatives "
              f"{row['auc_ebv_vs_other']:.3f} | overall {row['auc_all']:.3f}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
