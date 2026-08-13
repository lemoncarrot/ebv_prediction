"""Slide- and patient-level scoring for MIL runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, or NaN when a split is single-class (common in tiny folds)."""
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def score_block(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    positives, negatives = labels == 1, labels == 0
    two_class = len(np.unique(labels)) > 1
    return {
        "n": int(len(labels)), "n_pos": int(positives.sum()),
        "auroc": safe_auc(labels, scores),
        "auprc": float(average_precision_score(labels, scores)) if two_class else float("nan"),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if two_class else float("nan"),
        "sensitivity": float(predictions[positives].mean()) if positives.any() else float("nan"),
        "specificity": float(1 - predictions[negatives].mean()) if negatives.any() else float("nan"),
    }


def summarise(frame: pd.DataFrame, threshold: float) -> dict:
    """Slide-level metrics plus patient-level metrics (mean probability per patient)."""
    slide = score_block(frame["label"].to_numpy(), frame["prob"].to_numpy(), threshold)
    patient_frame = frame.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "mean"))
    patient = score_block(patient_frame["label"].to_numpy(), patient_frame["prob"].to_numpy(), threshold)
    return {"slide": slide, "patient": patient}
