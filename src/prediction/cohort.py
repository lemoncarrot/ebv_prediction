"""Everything that happens before the model: metadata -> labelled, split bags.

The contract with abmil.py is deliberately thin. This module turns
`metadata/cohort_labels.csv` plus one or more embedding directories into a
DataFrame with one row per usable slide, a `fold` column, and a Dataset that
yields `(features (n_tiles, embed_dim), label, slide_id)`. Nothing here imports
the model, and the model knows nothing about cohorts or folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"slide_id", "cohort", "patient_id", "label"}

# 3-class scheme: EBV is scored as P(ebv); msi and other are both EBV-negative.
CLASSES = ("other", "ebv", "msi")
EBV_INDEX = CLASSES.index("ebv")
MSI_SUBTYPE = "STAD_MSI"


def is_msi(frame: pd.DataFrame) -> np.ndarray:
    """MSI negatives by TCGA molecular subtype; False wherever it is unknown."""
    subtype = frame["tcga_subtype"] if "tcga_subtype" in frame else pd.Series(index=frame.index)
    return (subtype.eq(MSI_SUBTYPE) & frame["label"].eq(0)).to_numpy()


def class_targets(frame: pd.DataFrame) -> np.ndarray:
    """(n, 3) mask of the classes each slide may belong to.

    EBV+ -> {ebv}. Negatives with a known TCGA subtype -> {msi} or {other}
    (POLE, CIN, GS are other). Negatives without a subtype (Chile) -> {other,
    msi}: they are trained only to be not-EBV, never forced into either, so
    unlabelled MSI tumours cannot corrupt the msi class.
    """
    targets = np.zeros((len(frame), len(CLASSES)), dtype=np.float32)
    positive = frame["label"].eq(1).to_numpy()
    subtype = frame["tcga_subtype"] if "tcga_subtype" in frame else pd.Series(index=frame.index)
    known = subtype.notna().to_numpy()
    msi = is_msi(frame)
    targets[positive, EBV_INDEX] = 1
    targets[~positive & msi, CLASSES.index("msi")] = 1
    targets[~positive & known & ~msi, CLASSES.index("other")] = 1
    unknown = ~positive & ~known
    targets[unknown, CLASSES.index("other")] = 1
    targets[unknown, CLASSES.index("msi")] = 1
    return targets


# --------------------------------------------------------------------------
# bag index: slide_id -> .h5 written by run_pipeline.py
# --------------------------------------------------------------------------

def index_embeddings(dirs: list[Path]) -> dict[str, Path]:
    """slide_id -> .h5 path. Earlier directories win, so a rerun directory can
    be listed first to shadow an older cohort."""
    index: dict[str, Path] = {}
    for directory in dirs:
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            print(f"WARNING: embeddings dir does not exist: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.glob("*.h5")):
            index.setdefault(path.stem, path)
    return index


def bag_shape(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as h:
        return tuple(h["features"].shape)


def read_attrs(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in h.attrs.items()}


def check_consistent(paths: list[Path]) -> int:
    """Return the shared embedding dim, refusing a mixed-encoder cohort.

    Bags from two encoders load fine one at a time and only blow up once the
    model is built, so the check belongs here rather than at the first forward.
    """
    dims: dict[int, list[str]] = {}
    encoders: set[str] = set()
    for path in paths:
        dims.setdefault(bag_shape(path)[1], []).append(path.stem)
        encoders.add(str(read_attrs(path).get("encoder", "unknown")))
    if len(dims) > 1:
        detail = "; ".join(f"{d}-d: {len(ids)} slides ({', '.join(sorted(ids)[:3])}...)"
                           for d, ids in sorted(dims.items()))
        raise SystemExit(f"ERROR: mixed embedding dimensions in one cohort -- {detail}")
    if len(encoders) > 1:
        raise SystemExit(f"ERROR: mixed encoders in one cohort: {sorted(encoders)}")
    return next(iter(dims))


# --------------------------------------------------------------------------
# metadata -> the usable cohort
# --------------------------------------------------------------------------

def build_cohort(
    labels_csv: Path,
    embeddings_dirs: list[Path],
    min_tiles: int = 1,
    cohorts: list[str] | None = None,
    external: list[str] | None = None,
    ignore_qc_exclude: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Join labels to embedding files and apply the metadata filters.

    Returns (frame, embed_dim). Every drop is counted and reported, because a
    silently shrinking cohort is the easiest way to misread a result later.
    """
    labels = pd.read_csv(labels_csv)
    missing = REQUIRED_COLUMNS - set(labels.columns)
    if missing:
        raise SystemExit(f"ERROR: {labels_csv} missing columns: {sorted(missing)}")

    index = index_embeddings(embeddings_dirs)
    if not index:
        raise SystemExit(f"ERROR: no .h5 embeddings in {', '.join(map(str, embeddings_dirs))}")

    total = len(labels)
    labels = labels[labels["label"].notna()].copy()
    no_label = total - len(labels)
    labels["label"] = labels["label"].astype(int)

    qc_excluded = 0
    if "qc_exclude" in labels.columns and not ignore_qc_exclude:
        flagged = labels["qc_exclude"].fillna(0).astype(int) == 1
        qc_excluded = int(flagged.sum())
        labels = labels[~flagged].copy()

    external = set(external or [])
    if cohorts:
        keep = set(cohorts) | external
        unknown = keep - set(labels["cohort"].unique())
        if unknown:
            raise SystemExit(f"ERROR: unknown cohort(s) {sorted(unknown)}; "
                             f"available: {sorted(labels['cohort'].unique())}")
        labels = labels[labels["cohort"].isin(keep)].copy()

    labels["feature_path"] = labels["slide_id"].astype(str).map(index.get)
    no_features = int(labels["feature_path"].isna().sum())
    labels = labels[labels["feature_path"].notna()].copy()
    if labels.empty:
        raise SystemExit("ERROR: no labelled slide has an embedding file; "
                         "check --embeddings-dir and that slide_id matches the .h5 stem.")

    labels["n_tiles"] = [bag_shape(p)[0] for p in labels["feature_path"]]
    too_small = int((labels["n_tiles"] < min_tiles).sum())
    labels = labels[labels["n_tiles"] >= min_tiles].copy()
    if labels.empty:
        raise SystemExit(f"ERROR: no slide has >={min_tiles} tiles")

    labels["is_external"] = labels["cohort"].isin(external)
    labels = labels.sort_values(["cohort", "slide_id"]).reset_index(drop=True)
    embed_dim = check_consistent(list(labels["feature_path"]))

    print(f"cohort: {total} rows -> {len(labels)} usable slides (dropped {no_label} unlabelled, "
          f"{qc_excluded} qc_exclude, {no_features} without embeddings, "
          f"{too_small} with <{min_tiles} tiles); {embed_dim}-d")
    for (name, ext), group in labels.groupby(["cohort", "is_external"]):
        role = "external" if ext else "train/cv"
        print(f"  {name:<8} {role:<9} {len(group):>4} slides  "
              f"{group['patient_id'].nunique():>4} patients  pos={int(group['label'].sum()):<4} "
              f"neg={int((1 - group['label']).sum()):<4} tiles={group['n_tiles'].sum():,}")
    return labels, embed_dim


# --------------------------------------------------------------------------
# splits: patient-grouped so no patient spans train and held-out data
# --------------------------------------------------------------------------

def strata(frame: pd.DataFrame) -> np.ndarray:
    """Label, crossed with cohort when there is more than one, so a split of a
    mixed cohort keeps each cohort's share and positive rate."""
    key = frame["label"].astype(str)
    if frame["cohort"].nunique() > 1:
        key = frame["cohort"].astype(str) + ":" + key
    return pd.factorize(key)[0]


def imbalance(y: np.ndarray, index_sets: list[np.ndarray]) -> float:
    """Total |count - expected count| over strata and index sets; 0 is perfect."""
    totals = np.bincount(y)
    return float(sum(np.abs(np.bincount(y[idx], minlength=len(totals))
                            - totals * len(idx) / len(y)).sum() for idx in index_sets))


def assign_folds(frame: pd.DataFrame, folds: int, seed: int, restarts: int = 32) -> np.ndarray:
    """Outer-fold id per row, grouped by patient and stratified on (cohort x) label.

    Several slides can come from one patient, so a slide-level split leaks. As in
    inner_split, StratifiedGroupKFold balances only loosely, so try several
    shuffles and keep the most balanced assignment. Deterministic given `seed`.
    """
    labels = frame["label"].to_numpy()
    groups = frame["patient_id"].astype(str).to_numpy()
    folds = min(folds, len(np.unique(groups)), int(min(np.bincount(labels, minlength=2)[:2])))
    if folds < 2:
        raise SystemExit(f"ERROR: too few patients/positives to cross-validate "
                         f"(class counts {np.bincount(labels, minlength=2).tolist()})")
    y = strata(frame)
    best = None
    for restart in range(max(1, restarts)):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                        random_state=seed + restart * 1009)
        test_sets = [test_index for _, test_index in splitter.split(frame, y, groups)]
        error = imbalance(y, test_sets)
        if best is None or error < best[0]:
            best = (error, test_sets)

    assignment = np.empty(len(frame), dtype=int)
    for fold, test_index in enumerate(best[1]):
        assignment[test_index] = fold
    return assignment


def holdout_split(frame: pd.DataFrame, frac: float, seed: int, restarts: int = 32) -> np.ndarray:
    """Boolean mask of a held-out test set, grouped by patient and stratified on
    cohort x label. Candidates are the folds of round(1/frac)-way splits over
    several shuffles; the one closest to every stratum's share (then to `frac`)
    wins, so its size is approximate. Deterministic given `seed`."""
    y = strata(frame)
    groups = frame["patient_id"].astype(str).to_numpy()
    best = None
    for restart in range(max(1, restarts)):
        splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / frac)), shuffle=True,
                                        random_state=seed + restart * 1009)
        for _, test_index in splitter.split(frame, y, groups):
            key = (imbalance(y, [test_index]), abs(len(test_index) / len(frame) - frac))
            if best is None or key < best[0]:
                best = (key, test_index)
    is_test = np.zeros(len(frame), dtype=bool)
    is_test[best[1]] = True
    if frame.loc[is_test, "label"].nunique() < 2:
        raise SystemExit("ERROR: held-out split has a single class; raise --holdout-frac")
    return is_test


def inner_split(frame: pd.DataFrame, val_frac: float, seed: int, restarts: int = 8) -> np.ndarray:
    """Boolean mask of the early-stopping validation set, grouped by patient.

    StratifiedGroupKFold only balances on average, and with few positives a
    single emitted fold can be badly skewed. Generate every candidate fold over
    several shuffles and keep the one whose positive rate is closest to the
    whole cohort's, requiring both classes present. Deterministic given `seed`.
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
    return is_val


def check_no_leak(train: pd.DataFrame, held_out: pd.DataFrame, what: str) -> None:
    """Assert no patient spans the two sides. Cheap, and catches a resplit bug."""
    shared = set(train["patient_id"].astype(str)) & set(held_out["patient_id"].astype(str))
    if shared:
        raise SystemExit(f"ERROR: {len(shared)} patient(s) in both train and {what}: "
                         f"{sorted(shared)[:5]}")


# --------------------------------------------------------------------------
# the bags themselves
# --------------------------------------------------------------------------

def grid_neighbours(coords: np.ndarray, step: float) -> np.ndarray:
    """(n, 8) indices of each tile's 8 grid neighbours (-1 where absent).

    Tiles lie on a regular lattice of pitch `step` (level-0 px), so neighbours
    are found by exact lookup of grid positions rather than a radius search.
    """
    grid = np.rint(np.asarray(coords, dtype=np.float64) / step).astype(np.int64)
    span = int(grid[:, 1].max()) + 3 if len(grid) else 1
    keys = (grid[:, 0] + 1) * span + (grid[:, 1] + 1)          # +1: neighbours of row/col 0 stay >= 0
    order = np.argsort(keys)
    sorted_keys = keys[order]
    out = np.full((len(grid), 8), -1, dtype=np.int64)
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    for k, (dx, dy) in enumerate(offsets):
        target = keys + dx * span + dy
        pos = np.clip(np.searchsorted(sorted_keys, target), 0, max(len(keys) - 1, 0))
        hit = sorted_keys[pos] == target
        out[hit, k] = order[pos[hit]]
    return out


class BagDataset(Dataset):
    """One slide per item -> (features (n_tiles, embed_dim), label, slide_id)
    [, neighbours (n_tiles, 8) when built with neighbours=True, for graph layers].

    Bags are not padded or batched: ABMIL here consumes one bag at a time, so
    use `DataLoader(..., batch_size=None)`, which yields items unmodified.
    HDF5 handles open lazily so the dataset pickles into worker processes.
    """

    def __init__(self, frame: pd.DataFrame, feature_norm: str = "none",
                 max_tiles: int | None = None, seed: int = 0,
                 targets: np.ndarray | None = None, weights: np.ndarray | None = None,
                 neighbours: bool = False):
        """`targets` replaces the binary label as the training target (e.g. the
        (n, 3) mask from class_targets); `labels` stays binary for metrics.
        `weights` is a per-slide loss weight, looked up by slide_id."""
        if feature_norm not in ("none", "l2"):
            raise ValueError(f"feature_norm must be none or l2, got {feature_norm!r}")
        self.paths = [Path(p) for p in frame["feature_path"]]
        self.slide_ids = list(frame["slide_id"].astype(str))
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.targets = self.labels if targets is None else np.asarray(targets, dtype=np.float32)
        self.weight_by_id = ({} if weights is None
                             else dict(zip(self.slide_ids, np.asarray(weights, dtype=float))))
        self.feature_norm, self.max_tiles, self.seed = feature_norm, max_tiles, seed
        self.neighbours = neighbours
        self.epoch = 0
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """Change which tiles subsampling draws, deterministically."""
        self.epoch = epoch

    def _handle(self, i: int) -> h5py.File:
        if i not in self._handles:
            self._handles[i] = h5py.File(self.paths[i], "r")
        return self._handles[i]

    def __getitem__(self, i: int) -> tuple:
        handle = self._handle(i)
        dset = handle["features"]
        n = dset.shape[0]
        keep = slice(None)
        if self.max_tiles is not None and n > self.max_tiles:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            keep = np.sort(rng.choice(n, self.max_tiles, replace=False))
        features = dset[keep]
        features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        if self.feature_norm == "l2":
            features = torch.nn.functional.normalize(features, dim=1)
        item = (features, torch.tensor(self.targets[i]), self.slide_ids[i])
        if self.neighbours:                  # on the kept tiles, after any subsampling
            nb = grid_neighbours(handle["coords"][keep], float(handle.attrs["patch_size_level0"]))
            item += (torch.from_numpy(nb),)
        return item

    def __del__(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
