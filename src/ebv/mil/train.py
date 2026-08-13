#!/usr/bin/env python3
"""Train a MIL classifier for EBV status on UNI2-h bags, patient-grouped k-fold CV.

Each fold trains on k-1 outer groups, model-selects on an inner grouped
validation split (early stopping + Youden threshold), and is scored on the
held-out fold — which is also where attention is exported, so every heatmap
comes from a model that never saw the slide. `--external-cohort` is never
trained on and is scored by the fold ensemble.

    python -m ebv.mil.train --cohort chile --run-name chile_5fold
    python -m ebv.mil.train --features-dir 02_uni2_embeddings 02_uni2_embeddings_tcga \\
        --cohort chile --external-cohort tcga --run-name chile_train_tcga_ext
    python -m ebv.mil.train --cohort tcga --arch additive --run-name tcga_additive
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from ebv.mil.data import resolve_cohort
from ebv.mil.metrics import summarise, youden_threshold
from ebv.mil.models import ARCHITECTURES, build_model
from ebv.mil.splits import assign_folds, inner_split
from ebv.mil.trainer import Trainer
from ebv.paths import ROOT

DEFAULT_FEATURES = [ROOT / "02_uni2_embeddings"]
DEFAULT_LABELS = ROOT / "metadata" / "cohort_labels.csv"
DEFAULT_OUTPUT = ROOT / "04_mil"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--features-dir", type=Path, nargs="+", default=DEFAULT_FEATURES)
    data.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    data.add_argument("--cohort", nargs="+", default=None, help="Cohorts to train on (default: all).")
    data.add_argument("--external-cohort", nargs="+", default=None,
                      help="Held-out cohort(s), scored by the fold ensemble, never trained on.")
    data.add_argument("--ignore-qc-exclude", action="store_true",
                      help="Keep slides flagged qc_exclude=1 (dropped by default).")
    data.add_argument("--min-patches", type=int, default=16)
    data.add_argument("--limit", type=int, default=None, help="Keep only the first N slides (debug).")
    data.add_argument("--feature-norm", choices=("none", "l2"), default="none")

    model = p.add_argument_group("model")
    model.add_argument("--arch", choices=tuple(ARCHITECTURES), default="gated_attention")
    model.add_argument("--additive-pool", choices=("mean", "sum"), default="mean",
                       help="--arch additive only; mean keeps the logit bag-size stable.")
    model.add_argument("--hidden-dim", type=int, default=512)
    model.add_argument("--attn-dim", type=int, default=256)
    model.add_argument("--head-hidden", type=int, nargs="*", default=[256],
                       help="MLP widths after attention pooling; pass none for a linear head.")
    model.add_argument("--dropout", type=float, default=0.25)

    train = p.add_argument_group("training")
    train.add_argument("--balance", choices=("none", "downsample"), default="none",
                       help="downsample draws a fresh 1:1 pos/neg subset each epoch.")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--val-frac", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--max-patches", type=int, default=4096,
                       help="Patches sampled per bag while training; evaluation uses the full bag.")
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--pos-weight", type=float, default=None, help="BCE positive weight (default neg/pos).")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device", default=None)
    train.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA.")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--run-name", default=None)
    out.add_argument("--no-attention", action="store_true")
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--dry-run", action="store_true", help="Resolve slides/folds, write CSVs, stop.")

    args = p.parse_args()
    if args.folds < 2:
        p.error("--folds must be at least 2")
    if not 0.0 < args.val_frac < 0.5:
        p.error("--val-frac must be in (0, 0.5)")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp and not str(args.device).startswith("cuda"):
        args.amp = False
    if args.run_name is None:
        tag = "-".join(args.cohort) if args.cohort else "all"
        args.run_name = f"{tag}_{args.folds}fold_{time.strftime('%Y%m%d-%H%M%S')}"
    return args


def model_config(args) -> dict:
    cfg = dict(arch=args.arch, input_dim=1536, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
               head_hidden=list(args.head_hidden), n_classes=1, dropout=args.dropout)
    if args.arch == "additive":
        cfg["additive_pool"] = args.additive_pool
    return cfg


def pos_weight_for(frame: pd.DataFrame, override: float | None) -> float:
    if override is not None:
        return override
    positives = float(frame["label"].sum())
    return (len(frame) - positives) / positives if positives > 0 else 1.0


def serialisable(args) -> dict:
    def conv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list) and v and isinstance(v[0], Path):
            return [str(x) for x in v]
        return v
    return {k: conv(v) for k, v in vars(args).items()}


def train_one_fold(fold: int, train_frame, val_frame, test_frame, args, run_dir: Path,
                   attention_dir: Path | None):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    random.seed(args.seed + fold)

    model = build_model(model_config(args))
    trainer = Trainer(model, args)
    history = trainer.fit(train_frame, val_frame, pos_weight_for(train_frame, args.pos_weight),
                          seed=args.seed + fold)
    threshold = trainer.select_threshold(val_frame)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": model.config(), "fold": fold,
                "best_epoch": trainer.best_epoch, "threshold": threshold,
                "feature_norm": args.feature_norm}, checkpoint_dir / f"fold{fold}.pt")

    probs = trainer.predict(test_frame, attention_dir, extra_attrs={"fold": fold, "run": args.run_name})
    return trainer, history, threshold, probs


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and not args.overwrite and any(run_dir.iterdir()):
        print(f"ERROR: {run_dir} exists and is not empty; pass --overwrite or pick a new --run-name",
              file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    frame = resolve_cohort(args)
    internal = frame[~frame["is_external"]].reset_index(drop=True)
    external = frame[frame["is_external"]].reset_index(drop=True)
    if internal.empty:
        print("ERROR: every selected slide is in --external-cohort; nothing to train on", file=sys.stderr)
        return 2

    internal["fold"] = assign_folds(internal, args.folds, args.seed)
    n_folds = int(internal["fold"].max()) + 1
    if n_folds < args.folds:
        print(f"NOTE: reduced to {n_folds} folds -- too few patients or positives for {args.folds}")

    fold_rows = []
    for fold in range(n_folds):
        train_pool = internal[internal["fold"] != fold].reset_index(drop=True)
        is_val = inner_split(train_pool, args.val_frac, args.seed + fold)
        for slide_id, flag in zip(train_pool["slide_id"], is_val):
            fold_rows.append({"fold": fold, "slide_id": slide_id, "split": "inner_val" if flag else "train"})
        for slide_id in internal.loc[internal["fold"] == fold, "slide_id"]:
            fold_rows.append({"fold": fold, "slide_id": slide_id, "split": "test"})

    (run_dir / "config.json").write_text(json.dumps({
        **serialisable(args), "n_folds": n_folds, "n_slides": int(len(internal)),
        "n_external": int(len(external)), "torch": torch.__version__}, indent=2))
    columns = ["slide_id", "cohort", "patient_id", "label", "n_patches", "fold", "feature_path"]
    internal.assign(feature_path=internal["feature_path"].astype(str))[columns].to_csv(
        run_dir / "slides.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)

    if args.dry_run:
        print(f"dry run: wrote {run_dir}/slides.csv and folds.csv; stopping before training")
        return 0

    print(f"\ndevice: {args.device}  folds: {n_folds}  slides: {len(internal)}"
          f"{f' (+{len(external)} external)' if len(external) else ''}", flush=True)

    attention_dir = None if args.no_attention else run_dir / "attention"
    oof = np.full(len(internal), np.nan)
    external_scores, histories, fold_metrics = [], [], {}

    for fold in range(n_folds):
        test_mask = internal["fold"] == fold
        train_pool = internal[~test_mask].reset_index(drop=True)
        is_val = inner_split(train_pool, args.val_frac, args.seed + fold)
        train_frame = train_pool[~is_val].reset_index(drop=True)
        val_frame = train_pool[is_val].reset_index(drop=True)
        test_frame = internal[test_mask].reset_index(drop=True)
        print(f"\nfold {fold}: train {len(train_frame)} / val {len(val_frame)} / test {len(test_frame)} "
              f"(test pos={int(test_frame['label'].sum())})", flush=True)

        trainer, history, threshold, probs = train_one_fold(
            fold, train_frame, val_frame, test_frame, args, run_dir, attention_dir)
        for row in history:
            row["fold"] = fold
        histories.extend(history)
        oof[np.where(test_mask.to_numpy())[0]] = probs
        fold_metrics[f"fold{fold}"] = summarise(test_frame.assign(prob=probs), threshold)
        print(f"  fold {fold} test AUROC {fold_metrics[f'fold{fold}']['slide']['auroc']:.3f}", flush=True)

        if not external.empty:
            external_scores.append(trainer.predict(external))
        del trainer
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    pd.DataFrame(histories).to_csv(run_dir / "history.csv", index=False)

    oof_frame = internal[["slide_id", "cohort", "patient_id", "label", "n_patches", "fold"]].copy()
    oof_frame["prob"] = oof
    oof_frame.to_csv(run_dir / "oof_predictions.csv", index=False)

    pooled_threshold = youden_threshold(oof_frame["label"].to_numpy(), oof_frame["prob"].to_numpy())
    fold_aurocs = [m["slide"]["auroc"] for m in fold_metrics.values()]
    metrics = {"run": args.run_name, "n_folds": n_folds,
               "oof": summarise(oof_frame, pooled_threshold), "per_fold": fold_metrics,
               "fold_auroc_mean": float(np.nanmean(fold_aurocs)),
               "fold_auroc_std": float(np.nanstd(fold_aurocs))}

    if external_scores:
        ensemble = np.mean(np.vstack(external_scores), axis=0)
        external_frame = external[["slide_id", "cohort", "patient_id", "label", "n_patches"]].copy()
        external_frame["prob"] = ensemble
        for fold, scores in enumerate(external_scores):
            external_frame[f"prob_fold{fold}"] = scores
        external_frame.to_csv(run_dir / "external_predictions.csv", index=False)
        metrics["external"] = summarise(external_frame, pooled_threshold)
        print(f"\nexternal AUROC {metrics['external']['slide']['auroc']:.3f} ({len(external_frame)} slides)")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nOOF slide AUROC   {metrics['oof']['slide']['auroc']:.3f}  "
          f"AUPRC {metrics['oof']['slide']['auprc']:.3f}")
    print(f"OOF patient AUROC {metrics['oof']['patient']['auroc']:.3f}")
    print(f"per-fold AUROC    {metrics['fold_auroc_mean']:.3f} +/- {metrics['fold_auroc_std']:.3f}")
    print(f"\nrun directory: {run_dir}")
    if attention_dir is not None:
        print(f"next: python -m ebv.mil.heatmaps --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
