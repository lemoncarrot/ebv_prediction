"""Patient-grouped, EBV-stratified cross-validation splits.

No patient spans train and held-out data — several slides can come from one
block, so slide-level splits leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Grouped, stratified outer-fold id per row. Caps folds on tiny cohorts."""
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    folds = min(folds, len(np.unique(groups)), int(min(np.bincount(labels, minlength=2)[:2])))
    if folds < 2:
        raise SystemExit(f"ERROR: too few patients/positives to cross-validate "
                         f"(class counts {np.bincount(labels, minlength=2).tolist()})")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.empty(len(frame), dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(frame, labels, groups)):
        assignment[test_index] = fold
    return assignment


def inner_split(frame: pd.DataFrame, val_frac: float, seed: int,
                restarts: int = 8, verbose: bool = False) -> np.ndarray:
    """Boolean mask of the inner (early-stopping) validation set, grouped by patient.

    StratifiedGroupKFold only balances on average; a single emitted fold can be
    badly skewed with few positives. So generate every candidate fold across
    several shuffles and keep the one whose positive rate is closest to the whole
    cohort's, requiring both classes present. Deterministic given `seed`.
    """
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    n_splits = min(max(2, round(1 / val_frac)), len(np.unique(groups)),
                   int(min(np.bincount(labels, minlength=2)[:2])))
    if n_splits < 2:
        return np.zeros(len(frame), dtype=bool)  # too little structure; validate on train

    target = float(labels.mean())
    best = None
    for restart in range(max(1, restarts)):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed + restart * 1009)
        for _, val_index in splitter.split(frame, labels, groups):
            if len(np.unique(labels[val_index])) < 2:
                continue
            skew = abs(float(labels[val_index].mean()) - target)
            size_error = abs(len(val_index) / len(frame) - val_frac)
            if best is None or (skew, size_error) < (best[0], best[1]):
                best = (skew, size_error, val_index)

    is_val = np.zeros(len(frame), dtype=bool)
    if best is None:  # every candidate single-class; fall back to the first fold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_index = next(iter(splitter.split(frame, labels, groups)))
        is_val[val_index] = True
        return is_val
    is_val[best[2]] = True
    if verbose:
        print(f"  inner split: validation prevalence {100 * labels[best[2]].mean():.1f}% vs "
              f"{100 * target:.1f}% overall (skew {100 * best[0]:.2f} pp)", flush=True)
    return is_val
