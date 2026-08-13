"""Cohort resolution and the per-slide bag dataset.

One bag = one slide = the N x 1536 feature matrix from the embedding stage.
Labels come from `metadata/cohort_labels.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


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
    with h5py.File(path, "r") as h:
        return int(h["features"].shape[0])


def resolve_cohort(args) -> pd.DataFrame:
    """Join labels to feature files, apply QC/cohort/size filters, report drops."""
    labels = pd.read_csv(args.labels)
    required = {"slide_id", "cohort", "patient_id", "label"}
    if required - set(labels.columns):
        raise SystemExit(f"ERROR: {args.labels} missing columns: {sorted(required - set(labels.columns))}")

    features = index_features(args.features_dir)
    if not features:
        raise SystemExit(f"ERROR: no .h5 features in {', '.join(map(str, args.features_dir))}")

    total = len(labels)
    labels = labels[labels["label"].notna()].copy()
    no_label = total - len(labels)
    labels["label"] = labels["label"].astype(int)

    qc_excluded = 0
    if "qc_exclude" in labels.columns and not args.ignore_qc_exclude:
        flagged = labels["qc_exclude"].fillna(0).astype(int) == 1
        qc_excluded = int(flagged.sum())
        labels = labels[~flagged].copy()

    external = set(args.external_cohort) if args.external_cohort else set()
    if args.cohort:
        keep = set(args.cohort) | external
        unknown = keep - set(labels["cohort"].unique())
        if unknown:
            raise SystemExit(f"ERROR: unknown cohort(s) {sorted(unknown)}; "
                             f"available: {sorted(labels['cohort'].unique())}")
        labels = labels[labels["cohort"].isin(keep)].copy()

    labels["feature_path"] = labels["slide_id"].map(lambda s: features.get(str(s)))
    no_features = int(labels["feature_path"].isna().sum())
    labels = labels[labels["feature_path"].notna()].copy()
    if labels.empty:
        raise SystemExit("ERROR: no labelled slide has a feature file; check --features-dir and slide_id.")

    labels["n_patches"] = [patch_count(p) for p in labels["feature_path"]]
    too_small = int((labels["n_patches"] < args.min_patches).sum())
    labels = labels[labels["n_patches"] >= args.min_patches].copy()

    labels["is_external"] = labels["cohort"].isin(external)
    labels = labels.sort_values(["cohort", "slide_id"]).reset_index(drop=True)
    if args.limit is not None:
        labels = labels.groupby("is_external", group_keys=False).head(args.limit)

    print(f"labels: {total} rows -> {len(labels)} usable slides (dropped {no_label} unlabelled, "
          f"{qc_excluded} qc_exclude, {no_features} without features, "
          f"{too_small} with <{args.min_patches} patches)")
    for (cohort, ext), group in labels.groupby(["cohort", "is_external"]):
        role = "external" if ext else "train/cv"
        print(f"  {cohort:<8} {role:<9} {len(group):>4} slides  "
              f"{group['patient_id'].nunique():>4} patients  pos={int(group['label'].sum()):<4} "
              f"neg={int((1 - group['label']).sum()):<4} patches={group['n_patches'].sum():,}")
    return labels


class BagDataset(Dataset):
    """One slide per item. HDF5 handles open lazily per worker."""

    def __init__(self, frame: pd.DataFrame, max_patches: int | None, feature_norm: str, seed: int = 0):
        self.paths = [Path(p) for p in frame["feature_path"]]
        self.slide_ids = list(frame["slide_id"].astype(str))
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.max_patches, self.feature_norm, self.seed = max_patches, feature_norm, seed
        self.epoch = 0
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """Change which patches subsampling draws, deterministically."""
        self.epoch = epoch

    def _handle(self, index: int) -> h5py.File:
        if index not in self._handles:
            self._handles[index] = h5py.File(self.paths[index], "r")
        return self._handles[index]

    def __getitem__(self, index: int):
        dset = self._handle(index)["features"]
        n = dset.shape[0]
        if self.max_patches is not None and n > self.max_patches:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            features = dset[np.sort(rng.choice(n, self.max_patches, replace=False))]
        else:
            features = dset[:]
        features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        if self.feature_norm == "l2":
            features = torch.nn.functional.normalize(features, dim=1)
        return features, torch.tensor(self.labels[index]), index

    def __del__(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass


class BalancedNegativeSampler(Sampler):
    """Every positive bag plus an equally sized fresh draw of negatives per epoch.

    Downsampling the majority class, but redrawing negatives each epoch so the
    model still sees all of them across training. Call `set_epoch` to vary (and
    reproduce) the draw.
    """

    def __init__(self, labels: np.ndarray, seed: int = 0):
        self.positives = np.flatnonzero(labels == 1)
        self.negatives = np.flatnonzero(labels == 0)
        self.seed, self.epoch = seed, 0
        self.per_class = min(len(self.positives), len(self.negatives))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 2 * self.per_class if self.per_class else len(self.positives) + len(self.negatives)

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        if not self.per_class:
            order = np.concatenate([self.positives, self.negatives])
        else:
            order = np.concatenate([rng.choice(self.positives, self.per_class, replace=False),
                                    rng.choice(self.negatives, self.per_class, replace=False)])
        rng.shuffle(order)
        return iter(int(i) for i in order)


def collate(batch):
    """Zero-pad bags to the longest in the batch and build the attention mask."""
    features, labels, indices = zip(*batch)
    longest = max(len(f) for f in features)
    padded = features[0].new_zeros(len(batch), longest, features[0].shape[1])
    mask = torch.zeros(len(batch), longest, dtype=torch.bool)
    for i, item in enumerate(features):
        padded[i, : len(item)] = item
        mask[i, : len(item)] = True
    return padded, mask, torch.stack(labels), torch.tensor(indices)
