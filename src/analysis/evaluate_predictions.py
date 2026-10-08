#!/usr/bin/env python3
"""Evaluate blinded EBV predictions against ground-truth labels.

    python evaluate_predictions.py --predictions predictions.csv --labels labels.csv

Needs only numpy, pandas and scikit-learn, so it can run anywhere.

predictions.csv  (from us)   slide_id, prob_ebv, pred_ebv, ...
labels.csv       (yours)     slide_id, label   [, case_id]
                 label: 1 = EBV-positive, 0 = EBV-negative (e.g. EBER-ISH)
                 case_id (optional): patient/case, to also score per case when
                 a case has several slides (case score = mean slide probability)

Reports, per slide and (if case_id given) per case: AUROC, AUPRC, and at the
pre-specified threshold (default 0.5, as delivered) sensitivity, specificity,
PPV, NPV, accuracy and the confusion matrix, each with a 95% bootstrap CI
(stratified by label, 2000 resamples). Writes evaluation.json next to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    call = p >= threshold
    tp, fn = int((call & (y == 1)).sum()), int((~call & (y == 1)).sum())
    tn, fp = int((~call & (y == 0)).sum()), int((call & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"n": len(y), "positives": int(y.sum()),
            "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "accuracy": div(tp + tn, len(y)),
            "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp}}


def bootstrap(y: np.ndarray, p: np.ndarray, threshold: float, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy")
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = point(y[idx], p[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items()}


def report(level: str, y: np.ndarray, p: np.ndarray, args) -> dict:
    if len(np.unique(y)) < 2:
        raise SystemExit(f"ERROR: {level}-level labels have a single class; cannot compute AUROC")
    m = point(y, p, args.threshold)
    ci = bootstrap(y, p, args.threshold, args.n_boot, args.seed)
    print(f"\n{level}-level: n={m['n']}, EBV+={m['positives']}")
    for k in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy"):
        print(f"  {k:<12} {m[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    c = m["confusion"]
    print(f"  confusion at {args.threshold:g}: TP={c['TP']} FN={c['FN']} TN={c['TN']} FP={c['FP']}")
    return {**m, "ci95": ci}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("evaluation.json"))
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions, dtype={"slide_id": str})
    labels = pd.read_csv(args.labels, dtype={"slide_id": str})
    if not {"slide_id", "label"} <= set(labels.columns):
        raise SystemExit("ERROR: labels.csv needs columns slide_id, label")
    frame = preds.merge(labels, on="slide_id", how="inner")
    missing_label = sorted(set(preds["slide_id"]) - set(labels["slide_id"]))
    missing_pred = sorted(set(labels["slide_id"]) - set(preds["slide_id"]))
    print(f"matched {len(frame)} slides; {len(missing_label)} predictions without a label, "
          f"{len(missing_pred)} labels without a prediction")
    for name, ids in (("no label", missing_label), ("no prediction", missing_pred)):
        if ids:
            print(f"  {name}: {', '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}")
    frame = frame[frame["label"].isin([0, 1])]

    result = {"threshold": args.threshold, "unmatched": {
        "predictions_without_label": missing_label, "labels_without_prediction": missing_pred},
        "slide": report("slide", frame["label"].to_numpy(int), frame["prob_ebv"].to_numpy(), args)}
    if "case_id" in frame:
        cases = frame.groupby("case_id").agg(label=("label", "max"), prob=("prob_ebv", "mean"))
        result["case"] = report("case", cases["label"].to_numpy(int), cases["prob"].to_numpy(), args)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
