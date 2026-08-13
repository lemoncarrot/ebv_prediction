#!/usr/bin/env python3
"""Train the gated-attention MIL + MLP classifier for EBV status on UNI2-h bags.

One bag = one slide = the `N x 1536` feature matrix written by `01_call_uni.py`.
Labels come from `metadata/cohort_labels.csv` (slides with no EBV call are
dropped). Cross-validation is grouped by patient, so no patient ever appears in
both the training and the held-out data -- slide-level splits leak badly here
because several slides can come from one block.

Each fold is trained on k-2 groups, model-selected on an inner grouped
validation split (early stopping on validation AUROC), and scored on the
held-out fold. That held-out pass is also where attention is exported, so every
saved heatmap comes from a model that never saw the slide.

Outputs, under `--output-dir/<run-name>/`:

    config.json            exact arguments and resolved model config
    slides.csv             the resolved cohort: slide, patient, label, n_patches
    folds.csv              fold assignment and inner train/val split per slide
    history.csv            per-epoch train/val loss and val AUROC
    checkpoints/fold*.pt   model weights + model config + inner-val threshold
    oof_predictions.csv    out-of-fold slide probabilities
    metrics.json           slide- and patient-level AUROC/AUPRC/sens/spec
    attention/<slide>.h5   coords + attention weight per patch (OOF)
    external_predictions.csv   fold-ensembled scores for --external-cohort

`08_attention_heatmaps.py` turns `attention/*.h5` into per-slide CSV grids and
heatmap images.

Typical runs:

    # Chile only, what is runnable today
    python src/07_train_mil.py --cohort chile --run-name chile_5fold

    # Once the TCGA embeddings land: pooled training, and train->test transfer
    python src/07_train_mil.py --features-dir 02_uni2_embeddings 02_uni2_embeddings_tcga \\
        --cohort chile tcga --run-name pooled_5fold
    python src/07_train_mil.py --features-dir 02_uni2_embeddings 02_uni2_embeddings_tcga \\
        --cohort chile --external-cohort tcga --run-name chile_train_tcga_ext
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mil_model import ARCHITECTURES, AdditiveMIL, GatedAttentionMIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = [ROOT / "02_uni2_embeddings"]
DEFAULT_LABELS = ROOT / "metadata" / "cohort_labels.csv"
DEFAULT_OUTPUT = ROOT / "04_mil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--features-dir", type=Path, nargs="+", default=DEFAULT_FEATURES,
                      help="Directories of per-slide feature HDF5 files (searched in order).")
    data.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    data.add_argument("--cohort", nargs="+", default=None,
                      help="Cohorts to train on (default: every cohort with features).")
    data.add_argument("--external-cohort", nargs="+", default=None,
                      help="Held-out cohort(s), never trained on, scored by the fold ensemble.")
    data.add_argument("--ignore-qc-exclude", action="store_true",
                      help="Keep slides flagged qc_exclude=1 in the label table "
                           "(pen-contaminated slides are dropped by default).")
    data.add_argument("--min-patches", type=int, default=16,
                      help="Drop slides with fewer patches than this.")
    data.add_argument("--limit", type=int, default=None, help="Keep only the first N slides (debug).")
    data.add_argument("--feature-norm", choices=("none", "l2"), default="none",
                      help="l2 normalises each patch vector; no statistics are fitted, so no leakage.")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", choices=tuple(ARCHITECTURES), default=GatedAttentionMIL.arch,
                       help="gated_attention pools then predicts; additive predicts per patch "
                            "then sums, giving exact signed per-patch contributions "
                            "(Javed et al. 2022).")
    model.add_argument("--additive-pool", choices=("mean", "sum"), default="mean",
                       help="--arch additive only. mean divides per-patch logits by bag size "
                            "so the logit does not drift with patch count; sum is the paper's "
                            "Equation 4 verbatim. See mil_model.AdditiveMIL.")
    model.add_argument("--hidden-dim", type=int, default=512)
    model.add_argument("--attn-dim", type=int, default=256)
    model.add_argument("--head-hidden", type=int, nargs="*", default=[256],
                       help="MLP widths after attention pooling; pass none for a linear head.")
    model.add_argument("--dropout", type=float, default=0.25)

    train = parser.add_argument_group("training")
    train.add_argument("--single-split", action="store_true",
                       help="Train one model on a stratified train/val split of --cohort and "
                            "score --external-cohort once. No cross-validation, no fold "
                            "ensembling: the external cohort is the test set.")
    train.add_argument("--balance", choices=("none", "downsample"), default="none",
                       help="Class-imbalance handling on the training bags. downsample draws a "
                            "fresh 1:1 positive/negative subset each epoch. Combine with "
                            "--pos-weight 1 to test downsampling on its own.")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--val-frac", type=float, default=0.2,
                       help="Fraction of training groups held out for early stopping.")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=10, help="Early stopping patience, epochs.")
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--max-patches", type=int, default=4096,
                       help="Patches sampled per bag during training. Evaluation always uses "
                            "the full bag.")
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--pos-weight", type=float, default=None,
                       help="BCE positive class weight (default: negatives/positives).")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device", default=None)
    train.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA.")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--run-name", default=None)
    out.add_argument("--no-attention", action="store_true", help="Skip attention export.")
    out.add_argument("--overwrite", action="store_true", help="Reuse an existing run directory.")
    out.add_argument("--dry-run", action="store_true",
                     help="Resolve slides and splits, write slides.csv/folds.csv, then stop.")

    args = parser.parse_args()
    if args.folds < 2 and not args.single_split:
        parser.error("--folds must be at least 2")
    if args.single_split and not args.external_cohort:
        parser.error("--single-split needs --external-cohort: without a held-out cohort there "
                     "is nothing to test on")
    if not 0.0 < args.val_frac < 0.5:
        parser.error("--val-frac must be in (0, 0.5)")
    if args.batch_size < 1 or args.max_patches < 1:
        parser.error("--batch-size and --max-patches must be positive")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp and not str(args.device).startswith("cuda"):
        args.amp = False
    if args.run_name is None:
        tag = "-".join(args.cohort) if args.cohort else "all"
        args.run_name = f"{tag}_{args.folds}fold_{time.strftime('%Y%m%d-%H%M%S')}"
    return args


# --------------------------------------------------------------------------- data


def index_features(dirs: list[Path]) -> dict[str, Path]:
    """slide_id -> feature file. Earlier directories win on collision."""
    index: dict[str, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            print(f"WARNING: features dir does not exist: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.h5")):
            index.setdefault(path.stem, path)
    return index


def patch_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["features"].shape[0])


def resolve_cohort(args: argparse.Namespace) -> pd.DataFrame:
    """Join labels to feature files and report what was dropped and why."""
    labels = pd.read_csv(args.labels)
    required = {"slide_id", "cohort", "patient_id", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise SystemExit(f"ERROR: {args.labels} is missing columns: {sorted(missing)}")

    features = index_features(args.features_dir)
    if not features:
        raise SystemExit(
            f"ERROR: no .h5 features found in {', '.join(str(d) for d in args.features_dir)}"
        )

    total = len(labels)
    labels = labels[labels["label"].notna()].copy()
    no_label = total - len(labels)
    labels["label"] = labels["label"].astype(int)

    # QC exclusions are recorded in the label table by
    # src/11_annotate_qc_exclusions.py so that every run reads one authoritative
    # cohort definition instead of each caller passing its own slide list.
    qc_excluded = 0
    if "qc_exclude" in labels.columns and not args.ignore_qc_exclude:
        flagged = labels["qc_exclude"].fillna(0).astype(int) == 1
        qc_excluded = int(flagged.sum())
        labels = labels[~flagged].copy()

    wanted = set(args.cohort) if args.cohort else None
    external = set(args.external_cohort) if args.external_cohort else set()
    if wanted is not None:
        keep = wanted | external
        unknown = keep - set(labels["cohort"].unique())
        if unknown:
            raise SystemExit(f"ERROR: unknown cohort(s) {sorted(unknown)}; "
                             f"available: {sorted(labels['cohort'].unique())}")
        labels = labels[labels["cohort"].isin(keep)].copy()

    labels["feature_path"] = labels["slide_id"].map(lambda s: features.get(str(s)))
    no_features = int(labels["feature_path"].isna().sum())
    labels = labels[labels["feature_path"].notna()].copy()
    if labels.empty:
        raise SystemExit(
            "ERROR: no slide in the label table has a feature file. Check that "
            "--features-dir points at the embeddings and that slide_id matches the "
            "HDF5 file stem."
        )

    labels["n_patches"] = [patch_count(p) for p in labels["feature_path"]]
    too_small = int((labels["n_patches"] < args.min_patches).sum())
    labels = labels[labels["n_patches"] >= args.min_patches].copy()

    labels["is_external"] = labels["cohort"].isin(external)
    labels = labels.sort_values(["cohort", "slide_id"]).reset_index(drop=True)
    if args.limit is not None:
        labels = labels.groupby("is_external", group_keys=False).head(args.limit)

    print(f"labels: {total} rows -> {len(labels)} usable slides "
          f"(dropped {no_label} unlabelled, {qc_excluded} qc_exclude, "
          f"{no_features} without features, "
          f"{too_small} with <{args.min_patches} patches)")
    for (cohort, external_flag), group in labels.groupby(["cohort", "is_external"]):
        role = "external" if external_flag else "train/cv"
        print(f"  {cohort:<8} {role:<9} {len(group):>4} slides  "
              f"{group['patient_id'].nunique():>4} patients  "
              f"pos={int(group['label'].sum()):<4} neg={int((1 - group['label']).sum()):<4} "
              f"patches={group['n_patches'].sum():,}")
    return labels


class BagDataset(Dataset):
    """Serves one slide per item. HDF5 handles are opened lazily per worker."""

    def __init__(self, frame: pd.DataFrame, max_patches: int | None, feature_norm: str,
                 seed: int = 0) -> None:
        self.paths = [Path(p) for p in frame["feature_path"]]
        self.slide_ids = list(frame["slide_id"].astype(str))
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.max_patches = max_patches
        self.feature_norm = feature_norm
        self.seed = seed
        self.epoch = 0
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """Changes which patches subsampling draws, deterministically."""
        self.epoch = epoch

    def _handle(self, index: int) -> h5py.File:
        handle = self._handles.get(index)
        if handle is None:
            handle = h5py.File(self.paths[index], "r")
            self._handles[index] = handle
        return handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        handle = self._handle(index)
        dataset = handle["features"]
        n = dataset.shape[0]
        if self.max_patches is not None and n > self.max_patches:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            take = np.sort(rng.choice(n, self.max_patches, replace=False))
            features = dataset[take]
        else:
            features = dataset[:]
        features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        if self.feature_norm == "l2":
            features = torch.nn.functional.normalize(features, dim=1)
        return features, torch.tensor(self.labels[index]), index

    def __del__(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # interpreter teardown
                pass


class BalancedNegativeSampler(Sampler):
    """Each epoch: every positive bag plus an equally sized fresh draw of negatives.

    Downsampling the majority class instead of reweighting the loss. Redrawing
    the negative subset every epoch matters at this cohort size -- a fixed 1:1
    subset of TCGA would throw away ~290 of 321 negative slides permanently,
    whereas resampling shows the model all of them across training while keeping
    every individual gradient step balanced.

    Call `set_epoch` so the draw changes, and so it changes reproducibly.
    """

    def __init__(self, labels: np.ndarray, seed: int = 0) -> None:
        self.positives = np.flatnonzero(labels == 1)
        self.negatives = np.flatnonzero(labels == 0)
        self.seed = seed
        self.epoch = 0
        self.per_class = min(len(self.positives), len(self.negatives))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        # Falls back to the full set if a split is single-class.
        return 2 * self.per_class if self.per_class else len(self.positives) + len(self.negatives)

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        if not self.per_class:
            order = np.concatenate([self.positives, self.negatives])
        else:
            order = np.concatenate([
                rng.choice(self.positives, self.per_class, replace=False),
                rng.choice(self.negatives, self.per_class, replace=False),
            ])
        rng.shuffle(order)
        return iter(int(index) for index in order)


def collate(batch):
    """Zero-pad bags to the longest in the batch and build the attention mask."""
    features, labels, indices = zip(*batch)
    lengths = [len(f) for f in features]
    longest = max(lengths)
    padded = features[0].new_zeros(len(batch), longest, features[0].shape[1])
    mask = torch.zeros(len(batch), longest, dtype=torch.bool)
    for i, item in enumerate(features):
        padded[i, : len(item)] = item
        mask[i, : len(item)] = True
    return padded, mask, torch.stack(labels), torch.tensor(indices)


# ------------------------------------------------------------------------ splits


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Grouped, stratified fold ids. Falls back gracefully on tiny cohorts."""
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    folds = min(folds, n_groups, int(min(np.bincount(labels, minlength=2)[:2])))
    if folds < 2:
        raise SystemExit(
            "ERROR: too few patients or too few of one class to cross-validate "
            f"({n_groups} patients, class counts {np.bincount(labels, minlength=2).tolist()})"
        )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.empty(len(frame), dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(frame, labels, groups)):
        assignment[test_index] = fold
    return assignment


def inner_split(frame: pd.DataFrame, val_frac: float, seed: int,
                restarts: int = 8, verbose: bool = False) -> np.ndarray:
    """True where a slide belongs to the inner validation set (grouped by patient).

    `StratifiedGroupKFold` only balances on a best-effort basis: it cannot split a
    patient, so with few positives its individual folds can be badly skewed even
    though the folds are balanced on average. Taking whichever fold it happens to
    emit first is therefore a real risk -- on TCGA (29 positives, 326 patients)
    the first fold held 1 positive in 69 slides, a 1.4% validation prevalence
    against 10.0% in training, which makes validation AUROC almost meaningless
    for early stopping and gives a threshold fitted to a single positive slide.

    So generate every candidate fold across several shuffles and keep the one
    whose positive rate is closest to the whole cohort's, requiring both classes
    to be present. Deterministic given `seed`.
    """
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    n_splits = max(2, int(round(1 / val_frac)))
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups, int(min(np.bincount(labels, minlength=2)[:2])))
    if n_splits < 2:
        # Not enough structure to hold anything out; validate on the training data
        # and rely on the epoch cap instead of early stopping.
        return np.zeros(len(frame), dtype=bool)

    target = float(labels.mean())
    best: tuple[float, float, np.ndarray] | None = None
    for restart in range(max(1, restarts)):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed + restart * 1009)
        for _, val_index in splitter.split(frame, labels, groups):
            val_labels = labels[val_index]
            if len(np.unique(val_labels)) < 2:
                continue  # a single-class validation split cannot score AUROC
            # Balance first, then prefer the split closest to the requested size.
            skew = abs(float(val_labels.mean()) - target)
            size_error = abs(len(val_index) / len(frame) - val_frac)
            if best is None or (skew, size_error) < (best[0], best[1]):
                best = (skew, size_error, val_index)

    is_val = np.zeros(len(frame), dtype=bool)
    if best is None:
        # Every candidate was single-class; fall back to the unbalanced first fold
        # rather than silently training with no validation at all.
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_index = next(iter(splitter.split(frame, labels, groups)))
        is_val[val_index] = True
        return is_val

    is_val[best[2]] = True
    if verbose:
        print(f"  inner split: validation prevalence {100 * labels[best[2]].mean():.1f}% vs "
              f"{100 * target:.1f}% overall (skew {100 * best[0]:.2f} pp)", flush=True)
    return is_val


# ----------------------------------------------------------------------- metrics


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, or NaN when a split is single-class (common in tiny folds)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def score_block(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    positives = labels == 1
    negatives = labels == 0
    return {
        "n": int(len(labels)),
        "n_pos": int(positives.sum()),
        "auroc": safe_auc(labels, scores),
        "auprc": (float(average_precision_score(labels, scores))
                  if len(np.unique(labels)) > 1 else float("nan")),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (float(balanced_accuracy_score(labels, predictions))
                              if len(np.unique(labels)) > 1 else float("nan")),
        "sensitivity": float(predictions[positives].mean()) if positives.any() else float("nan"),
        "specificity": float(1 - predictions[negatives].mean()) if negatives.any() else float("nan"),
    }


def summarise(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    """Slide-level metrics plus patient-level metrics (mean probability)."""
    slide = score_block(frame["label"].to_numpy(), frame["prob"].to_numpy(), threshold)
    patient_frame = frame.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "mean"))
    patient = score_block(patient_frame["label"].to_numpy(), patient_frame["prob"].to_numpy(),
                          threshold)
    return {"slide": slide, "patient": patient}


# ---------------------------------------------------------------------- training


def autocast(args: argparse.Namespace):
    # contextlib.nullcontext rather than autocast(enabled=False): constructing a
    # CUDA autocast at all trips a bf16 capability probe on CPU-only torch builds.
    if not args.amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def run_epoch(model, loader, criterion, optimizer, args) -> float:
    """One pass. Training when `optimizer` is given, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)
    total, seen = 0.0, 0
    with torch.set_grad_enabled(training):
        for features, mask, labels, _ in loader:
            features = features.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            with autocast(args):
                logits, _ = model(features, mask)
                loss = criterion(logits.squeeze(-1).float(), labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            total += float(loss) * len(labels)
            seen += len(labels)
    return total / max(seen, 1)


@torch.no_grad()
def predict(model, frame: pd.DataFrame, args, attention_dir: Path | None,
            extra_attrs: dict[str, object] | None = None) -> np.ndarray:
    """Full-bag inference, one slide at a time; optionally writes attention HDF5s."""
    model.eval()
    dataset = BagDataset(frame, max_patches=None, feature_norm=args.feature_norm)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, pin_memory=str(args.device).startswith("cuda"))
    probabilities = np.zeros(len(frame), dtype=np.float64)
    for features, mask, _, indices in loader:
        index = int(indices[0])
        features = features.to(args.device, non_blocking=True)
        mask = mask.to(args.device, non_blocking=True)
        with autocast(args):
            logits, attention, contributions = model.forward_with_contributions(features, mask)
        probability = float(torch.sigmoid(logits.squeeze(-1).float())[0])
        probabilities[index] = probability
        if attention_dir is not None:
            write_attention(
                attention_dir, dataset.slide_ids[index], dataset.paths[index],
                attention[0].float().cpu().numpy(), probability,
                frame.iloc[index], extra_attrs or {},
                # b n c -> n for the single binary logit; None for non-additive.
                contributions=(None if contributions is None
                               else contributions[0, :, 0].float().cpu().numpy()),
            )
    return probabilities


def write_attention(directory: Path, slide_id: str, feature_path: Path,
                    attention: np.ndarray, probability: float, row: pd.Series,
                    extra_attrs: dict[str, object],
                    contributions: np.ndarray | None = None) -> None:
    """One HDF5 per slide: the attention weight of every patch, with its coordinate.

    Geometry (`patch_size_level0`, `level0_width/height`) is copied from the
    feature file so a heatmap can be reconstructed without reopening the slide.

    Additive MIL runs also store `contributions`: the signed per-patch logit
    terms, which sum to the slide logit. Negative values are inhibitory, i.e.
    evidence against EBV+. Attention cannot express that sign, so the two arrays
    say different things and are both kept.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(feature_path, "r") as source:
        coords = source["coords"][:]
        attrs = dict(source.attrs)
    if len(coords) != len(attention):
        raise ValueError(
            f"{slide_id}: {len(coords)} coordinates but {len(attention)} attention weights"
        )
    if contributions is not None and len(contributions) != len(coords):
        raise ValueError(
            f"{slide_id}: {len(coords)} coordinates but {len(contributions)} contributions"
        )
    target_path = directory / f"{slide_id}.h5"
    tmp_path = target_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as target:
        target.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        target.create_dataset("attention", data=attention.astype(np.float32),
                              compression="gzip", compression_opts=4)
        if contributions is not None:
            target.create_dataset("contributions", data=contributions.astype(np.float32),
                                  compression="gzip", compression_opts=4)
            target.attrs["logit"] = float(contributions.sum())
        for key in ("slide_path", "slide_name", "patch_size", "patch_size_level0",
                    "target_downsample", "target_magnification", "level0_width",
                    "level0_height", "level0_mpp", "patch_mpp"):
            if key in attrs:
                target.attrs[key] = attrs[key]
        target.attrs["slide_id"] = slide_id
        target.attrs["label"] = int(row["label"])
        target.attrs["cohort"] = str(row["cohort"])
        target.attrs["patient_id"] = str(row["patient_id"])
        target.attrs["prob"] = probability
        target.attrs["n_patches"] = len(attention)
        for key, value in extra_attrs.items():
            target.attrs[key] = value
    tmp_path.replace(target_path)


def train_fold(fold: int, train_frame: pd.DataFrame, val_frame: pd.DataFrame,
               args: argparse.Namespace, run_dir: Path) -> tuple[GatedAttentionMIL, list[dict]]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    random.seed(args.seed + fold)

    kwargs: dict[str, object] = dict(
        input_dim=1536, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
        head_hidden=tuple(args.head_hidden), n_classes=1, dropout=args.dropout,
    )
    if args.arch == AdditiveMIL.arch:
        kwargs["additive_pool"] = args.additive_pool
    model = ARCHITECTURES[args.arch](**kwargs).to(args.device)

    positives = float(train_frame["label"].sum())
    negatives = float(len(train_frame) - positives)
    pos_weight = args.pos_weight if args.pos_weight is not None else (
        negatives / positives if positives > 0 else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=args.device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_set = BagDataset(train_frame, args.max_patches, args.feature_norm, seed=args.seed + fold)
    sampler = None
    if args.balance == "downsample":
        sampler = BalancedNegativeSampler(train_frame["label"].to_numpy(), seed=args.seed + fold)
        print(f"  balance=downsample: {sampler.per_class} positives + {sampler.per_class} "
              f"negatives per epoch, redrawn from {len(sampler.negatives)} negatives", flush=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, collate_fn=collate,
        drop_last=(len(sampler) if sampler else len(train_set)) > args.batch_size,
        pin_memory=str(args.device).startswith("cuda"),
    )
    has_val = len(val_frame) > 0
    val_loader = None
    if has_val:
        val_set = BagDataset(val_frame, args.max_patches, args.feature_norm, seed=args.seed + fold)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=collate)

    history: list[dict] = []
    best = {"auc": -np.inf, "loss": np.inf, "epoch": -1, "state": None}
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)  # a different negative draw each epoch
        train_loss = run_epoch(model, train_loader, criterion, optimizer, args)

        if has_val:
            val_loss = run_epoch(model, val_loader, criterion, None, args)
            val_probs = predict(model, val_frame, args, attention_dir=None)
            val_auc = safe_auc(val_frame["label"].to_numpy(), val_probs)
            # NaN AUROC means a single-class validation split; fall back to loss.
            better = (val_auc > best["auc"]) if np.isfinite(val_auc) else (val_loss < best["loss"])
        else:
            val_loss, val_auc = float("nan"), float("nan")
            better = train_loss < best["loss"]

        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_auroc": val_auc})
        if better:
            best = {
                "auc": val_auc if np.isfinite(val_auc) else best["auc"],
                "loss": val_loss if has_val else train_loss,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        marker = " *" if better else ""
        print(f"  fold {fold} epoch {epoch:>3}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  val AUROC {val_auc:.3f}{marker}", flush=True)
        if args.patience and epoch - best["epoch"] >= args.patience:
            print(f"  fold {fold}: no improvement for {args.patience} epochs, stopping")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    threshold = 0.5
    if has_val:
        val_probs = predict(model, val_frame, args, attention_dir=None)
        threshold = youden_threshold(val_frame["label"].to_numpy(), val_probs)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config(), "fold": fold,
         "best_epoch": best["epoch"], "threshold": threshold, "feature_norm": args.feature_norm},
        checkpoint_dir / f"fold{fold}.pt",
    )
    return model, history


def describe_split(name: str, frame: pd.DataFrame) -> dict[str, object]:
    positives = int(frame["label"].sum())
    n = len(frame)
    print(f"  {name:<14} {n:>4} slides  {frame['patient_id'].nunique():>4} patients  "
          f"pos={positives:<4} neg={n - positives:<5} "
          f"{100 * positives / n if n else float('nan'):.1f}% positive")
    return {"n": n, "n_pos": positives, "prevalence": positives / n if n else float("nan")}


def run_single_split(internal: pd.DataFrame, external: pd.DataFrame,
                     args: argparse.Namespace, run_dir: Path) -> int:
    """Train one model on the internal cohort; score the external cohort once.

    No cross-validation and no fold ensembling. The internal cohort is split
    once, grouped by patient and stratified on the label, into a training set
    and a validation set used only for early stopping and for choosing the
    decision threshold. The external cohort is the test set and is touched
    exactly once, at the end.
    """
    is_val = inner_split(internal, args.val_frac, args.seed, verbose=True)
    train_frame = internal[~is_val].reset_index(drop=True)
    val_frame = internal[is_val].reset_index(drop=True)
    if val_frame.empty or len(np.unique(val_frame["label"])) < 2:
        print("ERROR: could not build a two-class validation split; lower --val-frac or check "
              "that the cohort has enough positives", file=sys.stderr)
        return 2

    print(f"\ndevice: {args.device}   single split, no cross-validation")
    print(f"balance: {args.balance}   pos_weight: "
          f"{'auto (neg/pos)' if args.pos_weight is None else args.pos_weight}")
    splits = {
        "train": describe_split("train", train_frame),
        "val": describe_split("val", val_frame),
        "external": describe_split("external/test", external),
    }
    # Stratification is proportional, so the two prevalences should track. Say so
    # out loud, because with ~29 positives a bad draw is both possible and fatal.
    drift = abs(splits["train"]["prevalence"] - splits["val"]["prevalence"])
    print(f"  train/val prevalence gap: {100 * drift:.2f} percentage points")

    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else
               [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else v)
           for k, v in vars(args).items()},
        "mode": "single_split",
        "splits": splits,
        "torch": torch.__version__,
    }, indent=2))
    columns = ["slide_id", "cohort", "patient_id", "label", "n_patches"]
    pd.concat([
        train_frame.assign(split="train"), val_frame.assign(split="val"),
        external.assign(split="external"),
    ])[columns + ["split"]].to_csv(run_dir / "slides.csv", index=False)

    if args.dry_run:
        print(f"dry run: wrote {run_dir}/slides.csv; stopping before training")
        return 0

    model, history = train_fold(0, train_frame, val_frame, args, run_dir)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(run_dir / "checkpoints" / "fold0.pt", map_location="cpu")
    threshold = float(checkpoint["threshold"])

    val_probs = predict(model, val_frame, args, attention_dir=None)
    val_frame = val_frame.assign(prob=val_probs)
    val_frame[columns + ["prob"]].to_csv(run_dir / "val_predictions.csv", index=False)

    attention_dir = None if args.no_attention else run_dir / "attention"
    external_probs = predict(model, external, args, attention_dir,
                             extra_attrs={"run": args.run_name})
    external_frame = external.assign(prob=external_probs)
    external_frame[columns + ["prob"]].to_csv(run_dir / "external_predictions.csv", index=False)

    metrics = {
        "run": args.run_name,
        "mode": "single_split",
        "balance": args.balance,
        "pos_weight": args.pos_weight,
        "threshold": threshold,
        "splits": splits,
        "val": summarise(val_frame, threshold),
        "external": summarise(external_frame, threshold),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nvalidation (TCGA held-out slides, threshold chosen here)")
    print(f"  AUROC {metrics['val']['slide']['auroc']:.3f}   "
          f"AUPRC {metrics['val']['slide']['auprc']:.3f}   threshold {threshold:.4f}")
    print(f"external test ({'+'.join(sorted(external['cohort'].unique()))}, never seen)")
    for key in ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy"):
        print(f"  {key:<18} {metrics['external']['slide'][key]:.3f}")
    print(f"\nrun directory: {run_dir}")
    return 0


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and not args.overwrite and any(run_dir.iterdir()):
        print(f"ERROR: {run_dir} exists and is not empty; pass --overwrite or pick a new "
              f"--run-name", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    frame = resolve_cohort(args)
    internal = frame[~frame["is_external"]].reset_index(drop=True)
    external = frame[frame["is_external"]].reset_index(drop=True)
    if internal.empty:
        print("ERROR: every selected slide is in --external-cohort; nothing to train on",
              file=sys.stderr)
        return 2

    if args.single_split:
        if external.empty:
            print("ERROR: --single-split needs a non-empty --external-cohort to test on",
                  file=sys.stderr)
            return 2
        return run_single_split(internal, external, args, run_dir)

    internal["fold"] = assign_folds(internal, args.folds, args.seed)
    n_folds = int(internal["fold"].max()) + 1
    if n_folds < args.folds:
        print(f"NOTE: reduced to {n_folds} folds -- too few patients or positives for "
              f"{args.folds}")

    fold_rows = []
    for fold in range(n_folds):
        train_mask = internal["fold"] != fold
        train_frame = internal[train_mask].reset_index(drop=True)
        is_val = inner_split(train_frame, args.val_frac, args.seed + fold)
        for slide_id, flag in zip(train_frame["slide_id"], is_val):
            fold_rows.append({"fold": fold, "slide_id": slide_id,
                              "split": "inner_val" if flag else "train"})
        for slide_id in internal.loc[~train_mask, "slide_id"]:
            fold_rows.append({"fold": fold, "slide_id": slide_id, "split": "test"})
    folds_frame = pd.DataFrame(fold_rows)

    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else
               [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else v)
           for k, v in vars(args).items()},
        "n_folds": n_folds,
        "n_slides": int(len(internal)),
        "n_external": int(len(external)),
        "torch": torch.__version__,
    }, indent=2))
    internal_columns = ["slide_id", "cohort", "patient_id", "label", "n_patches", "fold",
                        "feature_path"]
    internal.assign(feature_path=internal["feature_path"].astype(str))[internal_columns].to_csv(
        run_dir / "slides.csv", index=False)
    folds_frame.to_csv(run_dir / "folds.csv", index=False)
    if not external.empty:
        external.assign(feature_path=external["feature_path"].astype(str))[
            [c for c in internal_columns if c != "fold"]].to_csv(
            run_dir / "external_slides.csv", index=False)

    if args.dry_run:
        print(f"dry run: wrote {run_dir}/slides.csv and folds.csv; stopping before training")
        return 0

    print(f"\ndevice: {args.device}  folds: {n_folds}  slides: {len(internal)}"
          f"{f' (+{len(external)} external)' if len(external) else ''}", flush=True)

    attention_dir = None if args.no_attention else run_dir / "attention"
    oof = np.full(len(internal), np.nan)
    external_scores: list[np.ndarray] = []
    histories: list[dict] = []
    fold_metrics: dict[str, object] = {}

    for fold in range(n_folds):
        test_mask = internal["fold"] == fold
        train_pool = internal[~test_mask].reset_index(drop=True)
        is_val = inner_split(train_pool, args.val_frac, args.seed + fold)
        train_frame = train_pool[~is_val].reset_index(drop=True)
        val_frame = train_pool[is_val].reset_index(drop=True)
        test_frame = internal[test_mask].reset_index(drop=True)
        print(f"\nfold {fold}: train {len(train_frame)} / val {len(val_frame)} / "
              f"test {len(test_frame)} slides "
              f"(test pos={int(test_frame['label'].sum())})", flush=True)

        model, history = train_fold(fold, train_frame, val_frame, args, run_dir)
        histories.extend(history)

        checkpoint = torch.load(run_dir / "checkpoints" / f"fold{fold}.pt", map_location="cpu")
        threshold = float(checkpoint["threshold"])
        probabilities = predict(model, test_frame, args, attention_dir,
                                extra_attrs={"fold": fold, "run": args.run_name})
        oof[np.where(test_mask.to_numpy())[0]] = probabilities
        fold_frame = test_frame.assign(prob=probabilities)
        fold_metrics[f"fold{fold}"] = summarise(fold_frame, threshold)
        print(f"  fold {fold} test AUROC {fold_metrics[f'fold{fold}']['slide']['auroc']:.3f}",
              flush=True)

        if not external.empty:
            external_scores.append(predict(model, external, args, attention_dir=None))
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    pd.DataFrame(histories).to_csv(run_dir / "history.csv", index=False)

    oof_frame = internal[["slide_id", "cohort", "patient_id", "label", "n_patches", "fold"]].copy()
    oof_frame["prob"] = oof
    oof_frame.to_csv(run_dir / "oof_predictions.csv", index=False)

    pooled_threshold = youden_threshold(oof_frame["label"].to_numpy(), oof_frame["prob"].to_numpy())
    metrics: dict[str, object] = {
        "run": args.run_name,
        "n_folds": n_folds,
        "oof": summarise(oof_frame, pooled_threshold),
        "per_fold": fold_metrics,
        "fold_auroc_mean": float(np.nanmean(
            [m["slide"]["auroc"] for m in fold_metrics.values()])),
        "fold_auroc_std": float(np.nanstd([m["slide"]["auroc"] for m in fold_metrics.values()])),
    }

    if external_scores:
        ensemble = np.mean(np.vstack(external_scores), axis=0)
        external_frame = external[["slide_id", "cohort", "patient_id", "label", "n_patches"]].copy()
        external_frame["prob"] = ensemble
        for fold, scores in enumerate(external_scores):
            external_frame[f"prob_fold{fold}"] = scores
        external_frame.to_csv(run_dir / "external_predictions.csv", index=False)
        metrics["external"] = summarise(external_frame, pooled_threshold)
        print(f"\nexternal AUROC {metrics['external']['slide']['auroc']:.3f} "
              f"({len(external_frame)} slides)")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nOOF slide AUROC   {metrics['oof']['slide']['auroc']:.3f}  "
          f"AUPRC {metrics['oof']['slide']['auprc']:.3f}")
    print(f"OOF patient AUROC {metrics['oof']['patient']['auroc']:.3f}")
    print(f"per-fold AUROC    {metrics['fold_auroc_mean']:.3f} +/- {metrics['fold_auroc_std']:.3f}")
    print(f"\nrun directory: {run_dir}")
    if attention_dir is not None:
        print(f"attention:     {attention_dir} "
              f"({len(list(attention_dir.glob('*.h5')))} slides)")
        print(f"next:          python src/08_attention_heatmaps.py --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
