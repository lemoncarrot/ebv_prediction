"""A small training loop for one MIL model on one fold.

    model = build_model(config)
    trainer = Trainer(model, args)
    history = trainer.fit(train_frame, val_frame, pos_weight)
    threshold = trainer.select_threshold(val_frame)
    probs = trainer.predict(test_frame, attention_dir)
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ebv.mil.data import BagDataset, BalancedNegativeSampler, collate
from ebv.mil.metrics import safe_auc, youden_threshold


class Trainer:
    def __init__(self, model: nn.Module, args) -> None:
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        # nullcontext (not autocast(enabled=False)) so a bf16 probe never runs on CPU torch.
        return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.args.amp else contextlib.nullcontext())

    def _run_epoch(self, loader, criterion, optimizer) -> float:
        training = optimizer is not None
        self.model.train(training)
        total, seen = 0.0, 0
        with torch.set_grad_enabled(training):
            for features, mask, labels, _ in loader:
                features = features.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with self._autocast():
                    logits, _ = self.model(features, mask)
                    loss = criterion(logits.squeeze(-1).float(), labels)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.args.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    optimizer.step()
                total += float(loss) * len(labels)
                seen += len(labels)
        return total / max(seen, 1)

    # -------------------------------------------------------------------- train
    def fit(self, train_frame: pd.DataFrame, val_frame: pd.DataFrame, pos_weight: float,
            seed: int | None = None) -> list[dict]:
        """Train with early stopping on validation AUROC; leaves the best weights loaded."""
        args = self.args
        seed = args.seed if seed is None else seed
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device, dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_set = BagDataset(train_frame, args.max_patches, args.feature_norm, seed=seed)
        sampler = None
        if args.balance == "downsample":
            sampler = BalancedNegativeSampler(train_frame["label"].to_numpy(), seed=seed)
            print(f"  balance=downsample: {sampler.per_class} pos + {sampler.per_class} neg per epoch",
                  flush=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=args.workers, collate_fn=collate,
            drop_last=(len(sampler) if sampler else len(train_set)) > args.batch_size,
            pin_memory=str(self.device).startswith("cuda"))
        has_val = len(val_frame) > 0
        val_loader = (DataLoader(BagDataset(val_frame, args.max_patches, args.feature_norm, seed=seed),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 collate_fn=collate) if has_val else None)

        history: list[dict] = []
        best = {"auc": -np.inf, "loss": np.inf, "epoch": -1, "state": None}
        for epoch in range(args.epochs):
            train_set.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss = self._run_epoch(train_loader, criterion, optimizer)

            if has_val:
                val_loss = self._run_epoch(val_loader, criterion, None)
                val_auc = safe_auc(val_frame["label"].to_numpy(), self.predict(val_frame))
                better = (val_auc > best["auc"]) if np.isfinite(val_auc) else (val_loss < best["loss"])
            else:
                val_loss = val_auc = float("nan")
                better = train_loss < best["loss"]

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_auroc": val_auc})
            if better:
                best = {"auc": val_auc if np.isfinite(val_auc) else best["auc"],
                        "loss": val_loss if has_val else train_loss, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
            print(f"  epoch {epoch:>3}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"val AUROC {val_auc:.3f}{' *' if better else ''}", flush=True)
            if args.patience and epoch - best["epoch"] >= args.patience:
                print(f"  no improvement for {args.patience} epochs, stopping")
                break

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        self.best_epoch = best["epoch"]
        return history

    def select_threshold(self, val_frame: pd.DataFrame) -> float:
        if len(val_frame) == 0:
            return 0.5
        return youden_threshold(val_frame["label"].to_numpy(), self.predict(val_frame))

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, frame: pd.DataFrame, attention_dir: Path | None = None,
                extra_attrs: dict | None = None) -> np.ndarray:
        """Full-bag inference, one slide at a time; optionally writes attention HDF5s."""
        self.model.eval()
        dataset = BagDataset(frame, max_patches=None, feature_norm=self.args.feature_norm)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.args.workers,
                            collate_fn=collate, pin_memory=str(self.device).startswith("cuda"))
        probs = np.zeros(len(frame), dtype=np.float64)
        for features, mask, _, indices in loader:
            index = int(indices[0])
            features = features.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with self._autocast():
                logits, attention, contributions = self.model.forward_with_contributions(features, mask)
            prob = float(torch.sigmoid(logits.squeeze(-1).float())[0])
            probs[index] = prob
            if attention_dir is not None:
                write_attention(
                    attention_dir, dataset.slide_ids[index], dataset.paths[index],
                    attention[0].float().cpu().numpy(), prob, frame.iloc[index], extra_attrs or {},
                    contributions=(None if contributions is None
                                   else contributions[0, :, 0].float().cpu().numpy()))
        return probs


def write_attention(directory: Path, slide_id: str, feature_path: Path, attention: np.ndarray,
                    probability: float, row: pd.Series, extra_attrs: dict,
                    contributions: np.ndarray | None = None) -> None:
    """One HDF5 per slide: per-patch attention (and, for additive MIL, signed
    contributions summing to the slide logit), with coordinates and slide geometry
    copied from the feature file so a heatmap needs no slide access."""
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(feature_path, "r") as src:
        coords = src["coords"][:]
        attrs = dict(src.attrs)
    if len(coords) != len(attention):
        raise ValueError(f"{slide_id}: {len(coords)} coords but {len(attention)} attention weights")

    tmp = (directory / f"{slide_id}.h5").with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as out:
        out.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        out.create_dataset("attention", data=attention.astype(np.float32),
                           compression="gzip", compression_opts=4)
        if contributions is not None:
            out.create_dataset("contributions", data=contributions.astype(np.float32),
                               compression="gzip", compression_opts=4)
            out.attrs["logit"] = float(contributions.sum())
        for key in ("slide_path", "slide_name", "patch_size", "patch_size_level0", "target_downsample",
                    "target_magnification", "level0_width", "level0_height", "level0_mpp", "patch_mpp"):
            if key in attrs:
                out.attrs[key] = attrs[key]
        out.attrs.update(slide_id=slide_id, label=int(row["label"]), cohort=str(row["cohort"]),
                         patient_id=str(row["patient_id"]), prob=probability, n_patches=len(attention))
        for key, value in extra_attrs.items():
            out.attrs[key] = value
    tmp.replace(directory / f"{slide_id}.h5")
