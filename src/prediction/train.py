#!/usr/bin/env python3
"""Train and evaluate ABMIL over per-slide UNI2-h embeddings.

Splits are resolved from metadata first (cohort.py). The external set is either
whole cohorts (--external-cohort) or a patient-grouped slice of the training
cohorts (--holdout-frac); it is never trained on or used for any choice.

  1. k-fold CV on the training slides: per-fold models, early-stopped on an
     inner validation split, give out-of-fold metrics (predictions.csv).
  2. A final model is trained on all training slides for the median CV best
     epoch, then scores the external set once (external_predictions.csv).
     That final score is the headline result.

    python train.py --embeddings-dir ../../data/tcga_embeddings ../../data/chile_embeddings \
                    --labels ../../metadata/cohort_labels.csv \
                    --out-dir ../../results/run1 --external-cohort chile

    python train.py ... --holdout-frac 0.2      # mixed cohorts, 20% held out
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abmil import ABMIL, logodds
from cohort import (CLASSES, EBV_INDEX, BagDataset, assign_folds, build_cohort, check_no_leak,
                    class_targets, holdout_split, inner_split, is_msi)
from export_attention import write_attention
from tracking import Tracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embeddings-dir", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cohort", nargs="*", default=None, help="Restrict to these cohorts.")
    p.add_argument("--external-cohort", nargs="*", default=None,
                   help="Held out entirely; scored by the final model, never trained on.")
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="Instead of --external-cohort: hold out this fraction of the "
                        "training cohorts (patient-grouped, stratified on cohort x label).")
    p.add_argument("--no-external", action="store_true",
                   help="Train the final model on every slide with no external set; score "
                        "new cohorts afterwards with predict.py.")
    p.add_argument("--ignore-qc-exclude", action="store_true")
    p.add_argument("--min-tiles", type=int, default=16)
    # splits
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2, help="Inner early-stopping split.")
    p.add_argument("--seed", type=int, default=0)
    # model
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--attn-dim", type=int, default=256)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--feature-norm", choices=("none", "l2"), default="none")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Subsample bags larger than this each epoch.")
    # optimisation
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accumulate", type=int, default=8,
                   help="Bags per optimiser step (bags are fed one at a time).")
    p.add_argument("--patience", type=int, default=10, help="Epochs without val AUC gain.")
    p.add_argument("--final-epochs", type=int, default=None,
                   help="Epochs for the final all-data model (default: median CV best epoch + 1).")
    p.add_argument("--balance-loss", action="store_true",
                   help="Weight positives by the negative/positive ratio.")
    p.add_argument("--label-scheme", choices=("binary", "3class"), default="binary",
                   help="3class: softmax over other/ebv/msi (MSI from tcga_subtype); "
                        "negatives without a subtype may be either other or msi. "
                        "EBV is always scored as P(ebv).")
    p.add_argument("--msi-weight", type=float, default=1.0,
                   help="Loss weight on MSI negatives (tcga_subtype STAD_MSI).")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    # tracking
    p.add_argument("--run-name", default=None, help="Defaults to the out-dir name.")
    p.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    p.add_argument("--wandb-project", default="ebv-mil")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None,
                   help="Group runs of one experiment/sweep together.")
    return p.parse_args()


def loader(dataset: BagDataset, shuffle: bool, workers: int) -> DataLoader:
    # batch_size=None disables collation: items arrive exactly as the dataset
    # returns them, which is what a one-bag-at-a-time model wants.
    return DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=workers)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """AUC/AP are undefined on a single-class set; report NaN rather than crash."""
    both_classes = len(np.unique(labels)) > 1
    return {
        "n": int(len(labels)),
        "pos": int(labels.sum()),
        "auc": float(roc_auc_score(labels, probs)) if both_classes else float("nan"),
        "ap": float(average_precision_score(labels, probs)) if both_classes else float("nan"),
        "acc": float(((probs >= 0.5) == labels).mean()),
    }


@torch.no_grad()
def evaluate(model: ABMIL, dataset: BagDataset, device: str) -> pd.DataFrame:
    """Score every bag once. Returns one row per slide; `prob` is P(EBV), and a
    multi-class model adds p_<class> for every class."""
    model.eval()
    rows = []
    for i, (features, _, slide_id) in enumerate(loader(dataset, shuffle=False, workers=0)):
        logits, _ = model(features.to(device))
        row = {"slide_id": slide_id, "label": int(dataset.labels[i]),
               "prob": float(torch.sigmoid(logodds(logits, EBV_INDEX)).item())}
        if logits.numel() > 1:
            row.update({f"p_{name}": float(p) for name, p in
                        zip(CLASSES, torch.softmax(logits, dim=-1).tolist())})
        rows.append(row)
    return pd.DataFrame(rows)


def slide_loss(logits: torch.Tensor, target: torch.Tensor,
               pos_weight: torch.Tensor | None) -> torch.Tensor:
    """BCE for a single-logit head. For a softmax head, `target` is a mask of
    allowed classes and the loss is -log P(any allowed class), so a slide known
    only to be EBV-negative is never forced into msi or other. `pos_weight`
    scales EBV+ slides in both cases."""
    if logits.numel() == 1:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target.reshape(1), pos_weight=pos_weight)
    allowed = target.bool()
    loss = torch.logsumexp(logits, 0) - torch.logsumexp(logits[allowed], 0)
    if pos_weight is not None and allowed[EBV_INDEX] and int(allowed.sum()) == 1:
        loss = loss * pos_weight
    return loss.reshape(1)


def fit(model: ABMIL, train_set: BagDataset, val_set: BagDataset | None, args,
        tracker: Tracker, fold: int, epochs: int | None = None) -> dict:
    """Train with early stopping on validation AUC. Restores the best weights.

    Without a validation set it trains for exactly `epochs` and keeps the last.
    """
    device = args.device
    model.to(device)

    pos_weight = None
    if args.balance_loss:
        pos = float(train_set.labels.sum())
        neg = float(len(train_set.labels) - pos)
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"auc": float("nan"), "epoch": -1, "state": deepcopy(model.state_dict())}
    history = []

    for epoch in range(epochs or args.epochs):
        model.train()
        train_set.set_epoch(epoch)          # re-draw the tile subsample
        optimiser.zero_grad(set_to_none=True)
        total, seen = 0.0, 0

        batches = loader(train_set, True, args.workers)
        for step, (features, target, slide_id) in enumerate(batches, 1):
            logits, _ = model(features.to(device))
            loss = slide_loss(logits, target.to(device), pos_weight)
            loss = loss * train_set.weight_by_id.get(slide_id, 1.0)
            (loss / args.accumulate).backward()
            total += float(loss.item())
            seen += 1
            if step % args.accumulate == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
        optimiser.step()                    # flush the trailing partial batch
        optimiser.zero_grad(set_to_none=True)

        train_loss = total / max(seen, 1)
        if val_set is None or len(val_set) == 0:
            history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": float("nan")})
            tracker.log(fold=fold, epoch=epoch, train_loss=train_loss)
            print(f"    epoch {epoch:>3}  loss {train_loss:.4f}", flush=True)
            best = {"auc": float("nan"), "epoch": epoch, "state": deepcopy(model.state_dict())}
            continue

        scored = evaluate(model, val_set, device)
        val_auc = metrics(scored["prob"].to_numpy(), scored["label"].to_numpy())["auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        tracker.log(fold=fold, epoch=epoch, train_loss=train_loss, val_auc=val_auc)
        shown = f"{val_auc:.3f}" if np.isfinite(val_auc) else "n/a"
        print(f"    epoch {epoch:>3}  loss {train_loss:.4f}  val AUC {shown}", flush=True)

        if np.isfinite(val_auc) and not (val_auc <= best["auc"]):   # nan-safe first improvement
            best = {"auc": val_auc, "epoch": epoch, "state": deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= args.patience:
            print(f"    early stop at epoch {epoch} (best {best['auc']:.3f} "
                  f"@ {best['epoch']})", flush=True)
            break

    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_auc": best["auc"], "history": history}


def bags(rows: pd.DataFrame, args, max_tiles: int | None) -> BagDataset:
    targets = class_targets(rows) if args.label_scheme == "3class" else None
    weights = None
    if args.msi_weight != 1.0:
        weights = np.where(is_msi(rows), args.msi_weight, 1.0)
    return BagDataset(rows, args.feature_norm, max_tiles, args.seed, targets, weights)


def new_model(embed_dim: int, args) -> ABMIL:
    return ABMIL(input_dim=embed_dim, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
                 head_hidden=args.head_hidden, dropout=args.dropout,
                 n_classes=len(CLASSES) if args.label_scheme == "3class" else 1)


def score(model: ABMIL, rows: pd.DataFrame, args) -> pd.DataFrame:
    """evaluate() plus the cohort of each slide (row order is preserved)."""
    scored = evaluate(model, bags(rows, args, None), args.device)
    scored["cohort"] = rows["cohort"].to_numpy()
    return scored


def run_fold(fold: int, frame: pd.DataFrame, embed_dim: int, args,
             tracker: Tracker) -> tuple[pd.DataFrame, dict]:
    """Train on every other fold, score this fold's slides. The external set is
    left alone -- only the final model sees it."""
    test = frame[frame["fold"] == fold]
    pool = frame[frame["fold"] != fold].reset_index(drop=True)
    check_no_leak(pool, test, f"test fold {fold}")

    is_val = inner_split(pool, args.val_frac, args.seed + fold)
    train, val = pool[~is_val], pool[is_val]
    check_no_leak(train, val, "inner validation")

    print(f"  fold {fold}: train {len(train)} / val {len(val)} / test {len(test)} slides "
          f"(test pos={int(test['label'].sum())})", flush=True)

    torch.manual_seed(args.seed + fold)
    model = new_model(embed_dim, args)

    fit_info = fit(model, bags(train, args, args.max_tiles),
                   bags(val, args, None) if len(val) else None, args, tracker, fold)

    predictions = score(model, test, args)
    predictions["fold"], predictions["split"] = fold, "test"
    summary = {"fold": fold, **fit_info, **{f"test_{k}": v for k, v in
               metrics(predictions["prob"].to_numpy(), predictions["label"].to_numpy()).items()}}
    history = summary.pop("history")   # a per-epoch list; goes in the checkpoint, not the CSV

    torch.save({"config": model.config(), "state_dict": model.state_dict(),
                "fold": fold, "history": history},
               args.out_dir / f"fold{fold}.pt")
    tracker.log(**{f"fold_result/{k}": v for k, v in summary.items()})
    return predictions, summary


def train_final(internal: pd.DataFrame, external: pd.DataFrame, embed_dim: int,
                epochs: int, args, tracker: Tracker) -> tuple[pd.DataFrame, dict]:
    """Fit one model on every training slide for a fixed epoch count, then score
    the external set once. There is no validation split, so nothing to
    early-stop on: the epoch count comes from CV."""
    print(f"  final: train {len(internal)} slides for {epochs} epochs"
          + (f", then score {len(external)} external (pos={int(external['label'].sum())})"
             if len(external) else " (no external set)"), flush=True)
    torch.manual_seed(args.seed)
    model = new_model(embed_dim, args)
    fit_info = fit(model, bags(internal, args, args.max_tiles), None, args, tracker,
                   fold=-1, epochs=epochs)   # fold -1 marks the final model in the logs
    torch.save({"config": model.config(), "state_dict": model.state_dict(),
                "fold": "final", "epochs": epochs, "history": fit_info["history"],
                "feature_norm": args.feature_norm, "label_scheme": args.label_scheme,
                "classes": CLASSES if args.label_scheme == "3class" else ("ebv",)},
               args.out_dir / "final.pt")
    if external.empty:
        return pd.DataFrame(), {}

    scored = score(model, external, args)
    write_attention(model, external.assign(split="external"), args.out_dir / "attention.h5",
                    args.device, args.feature_norm)
    scored["split"] = "external"
    result = {"all": metrics(scored["prob"].to_numpy(), scored["label"].to_numpy())}
    if scored["cohort"].nunique() > 1:
        for cohort, group in scored.groupby("cohort"):
            result[cohort] = metrics(group["prob"].to_numpy(), group["label"].to_numpy())
    return scored, result


def main() -> int:
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracker = Tracker(enabled=args.wandb, project=args.wandb_project,
                      entity=args.wandb_entity, name=args.run_name or args.out_dir.name,
                      group=args.wandb_group, config=vars(args))

    frame, embed_dim = build_cohort(
        args.labels, args.embeddings_dir, min_tiles=args.min_tiles,
        cohorts=args.cohort, external=args.external_cohort,
        ignore_qc_exclude=args.ignore_qc_exclude,
    )
    external = frame[frame["is_external"]].reset_index(drop=True)
    internal = frame[~frame["is_external"]].reset_index(drop=True)
    if internal.empty:
        raise SystemExit("ERROR: every slide is in an external cohort; nothing to train on")

    if args.holdout_frac:
        if len(external):
            raise SystemExit("ERROR: pass --external-cohort or --holdout-frac, not both")
        is_test = holdout_split(internal, args.holdout_frac, args.seed)
        external = internal[is_test].reset_index(drop=True)
        internal = internal[~is_test].reset_index(drop=True)
        check_no_leak(internal, external, "held-out test split")
        print(f"held out {len(external)} slides as the external set:")
        for name, group in external.groupby("cohort"):
            print(f"  {name:<8} {len(group):>4} slides  pos={int(group['label'].sum())}")
    if external.empty and not args.no_external:
        raise SystemExit("ERROR: no external set; pass --external-cohort, --holdout-frac, "
                         "or --no-external")
    if len(external) and args.no_external:
        raise SystemExit("ERROR: --no-external conflicts with --external-cohort/--holdout-frac")

    internal["fold"] = assign_folds(internal, args.folds, args.seed)
    n_folds = int(internal["fold"].nunique())
    splits = pd.concat([internal.assign(split="train"),
                        external.assign(split="external", fold=-1)], ignore_index=True)
    splits[["slide_id", "cohort", "patient_id", "label", "split", "fold"]].to_csv(
        args.out_dir / "splits.csv", index=False)
    tracker.summary(embed_dim=embed_dim, n_folds=n_folds, n_slides=len(internal),
                    n_positive=int(internal["label"].sum()),
                    n_patients=int(internal["patient_id"].nunique()),
                    n_external=len(external))
    print(f"training on {args.device}: {n_folds} folds, {len(internal)} slides, "
          f"{len(external)} external", flush=True)
    if args.label_scheme == "3class":
        allowed = class_targets(internal).astype(bool)
        exact = allowed.sum(1) == 1
        counts = {name: int((allowed[:, k] & exact).sum()) for k, name in enumerate(CLASSES)}
        print(f"3-class targets on training slides: {counts}, "
              f"{int((~exact).sum())} negatives as other-or-msi", flush=True)
    if args.msi_weight != 1.0:
        print(f"MSI negatives weighted x{args.msi_weight:g}: "
              f"{int(is_msi(internal).sum())} training slides", flush=True)

    all_predictions, summaries = [], []
    for fold in range(n_folds):
        predictions, summary = run_fold(fold, internal, embed_dim, args, tracker)
        all_predictions.append(predictions)
        summaries.append(summary)
        print(f"  fold {fold}: test AUC {summary['test_auc']:.3f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    folds_frame = pd.DataFrame(summaries)
    folds_frame.to_csv(args.out_dir / "folds.csv", index=False)

    # Pooled out-of-fold score: every training slide predicted by the one fold
    # model that never saw it. Describes the training procedure, not the final model.
    pooled = metrics(predictions["prob"].to_numpy(), predictions["label"].to_numpy())

    # The final model: all training slides, epoch count from CV, scored once on
    # the external set. This is the headline result.
    final_epochs = args.final_epochs or int(round(np.median(folds_frame["best_epoch"]))) + 1
    external_predictions, final = train_final(internal, external, embed_dim, final_epochs,
                                              args, tracker)
    outputs = ["splits.csv", "predictions.csv", "folds.csv", "report.json"]
    if len(external_predictions):
        external_predictions.to_csv(args.out_dir / "external_predictions.csv", index=False)
        outputs[3:3] = ["external_predictions.csv", "attention.h5"]

    report = {"pooled_oof": pooled,
              "fold_mean_auc": float(folds_frame["test_auc"].mean()),
              "fold_std_auc": float(folds_frame["test_auc"].std()),
              "final_epochs": final_epochs,
              "external": final,
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    # default=str: some args are lists of Paths (e.g. --embeddings-dir)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    tracker.summary(**{f"oof_{k}": v for k, v in pooled.items()},
                    fold_mean_auc=report["fold_mean_auc"],
                    fold_std_auc=report["fold_std_auc"],
                    final_epochs=final_epochs,
                    **{f"external_{k}": v for k, v in final.get("all", {}).items()})
    for name in outputs:
        tracker.save(args.out_dir / name)
    tracker.finish()

    print(f"\npooled out-of-fold: AUC {pooled['auc']:.3f}  AP {pooled['ap']:.3f}  "
          f"n={pooled['n']} pos={pooled['pos']}")
    print(f"per-fold test AUC:  {report['fold_mean_auc']:.3f} "
          f"+/- {report['fold_std_auc']:.3f}")
    for name, m in final.items():
        print(f"external {name:<9} AUC {m['auc']:.3f}  AP {m['ap']:.3f}  acc {m['acc']:.3f}  "
              f"n={m['n']} pos={m['pos']}  (final model, {final_epochs} epochs)")
    print(f"wrote {args.out_dir}/{', '.join(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
