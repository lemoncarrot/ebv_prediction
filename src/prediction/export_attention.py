#!/usr/bin/env python3
"""Save per-tile attention (and tile scores) from a trained ABMIL for heatmaps.

train.py calls write_attention() for the final model on the external set. Run
this directly to export from an existing run's final.pt without retraining:

    python export_attention.py --run-dir ../../results/tcga2chile_45065815 \
        --embeddings-dir ../../data/tcga_embeddings ../../data/chile_embeddings

Writes <run-dir>/attention.h5, one group per slide:
    coords      (n, 2) int32   level-0 top-left of each tile (from the embedding .h5)
    attention   (n,)   float32 softmax attention, sums to 1 over the slide
    tile_logit  (n,)   float32 EBV log-odds of the bag head applied to each tile alone -- an
                               approximate per-tile EBV score (sign = direction)
    contribution (n, K) float32 additive models only: a_i * g(h_i), the exact per-tile
                               contribution to each class logit (sums to the slide logits)
    attrs: label, prob, cohort, split, plus every attr of the embedding .h5
           (slide_path, patch_size_level0, mpp_base, ...) needed to draw it.

Attention says where the model looked, not which way that tile pushed the call;
tile_logit is the (approximate) direction. Show both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abmil import ABMIL, logodds
from cohort import EBV_INDEX, BagDataset, index_embeddings, read_attrs


@torch.no_grad()
def write_attention(model: ABMIL, rows: pd.DataFrame, out_path: Path, device: str,
                    feature_norm: str = "none") -> None:
    """Score every slide in `rows` (needs slide_id, label, cohort, feature_path)
    and write attention.h5. Overwrites groups for slides already present."""
    model.to(device).eval()
    dataset = BagDataset(rows, feature_norm, neighbours=getattr(model, "graph_layers", 0) > 0)
    splits = rows["split"] if "split" in rows else pd.Series("", index=rows.index)
    with h5py.File(out_path, "a") as out:
        for i in range(len(dataset)):
            features, label, slide_id, *nb = dataset[i]
            x = features.to(device)
            nb = nb[0].to(device) if nb else None            # graph models: neighbour index
            logits, attention = model(x, nb)                 # either aggregation mode
            prob = torch.sigmoid(logodds(logits, EBV_INDEX)).item()
            _, h = model.attend(x, nb)
            tile_logits = model.head(h)                      # (n, n_classes)
            tile_logit = logodds(tile_logits, EBV_INDEX)     # EBV vs rest, any head

            source = Path(rows["feature_path"].iloc[i])
            with h5py.File(source, "r") as emb:
                coords = emb["coords"][:]
            if slide_id in out:
                del out[slide_id]
            group = out.create_group(slide_id)
            group.create_dataset("coords", data=coords, compression="gzip")
            group.create_dataset("attention", data=attention.cpu().numpy().astype(np.float32))
            group.create_dataset("tile_logit", data=tile_logit.cpu().numpy().astype(np.float32))
            if getattr(model, "aggregation", "attention") == "additive":
                # exact: summing over tiles gives the slide's logits
                contribution = (attention[:, None] * tile_logits).cpu().numpy()
                group.create_dataset("contribution", data=contribution.astype(np.float32))
            group.attrs.update({k: v for k, v in read_attrs(source).items() if v is not None})
            group.attrs.update({"label": int(label.item()), "prob": float(prob),
                                "cohort": str(rows["cohort"].iloc[i]),
                                "split": str(splits.iloc[i])})
    print(f"wrote attention for {len(dataset)} slides to {out_path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True, help="Holds final.pt and splits.csv.")
    p.add_argument("--embeddings-dir", type=Path, nargs="+", required=True)
    p.add_argument("--split", choices=("external", "train", "all"), default="external",
                   help="Which slides of splits.csv to export (default: external).")
    p.add_argument("--feature-norm", choices=("none", "l2"), default=None,
                   help="Defaults to what final.pt recorded, else none.")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.run_dir / "final.pt", map_location="cpu", weights_only=False)
    model = ABMIL.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    feature_norm = args.feature_norm or checkpoint.get("feature_norm", "none")

    rows = pd.read_csv(args.run_dir / "splits.csv")
    if args.split != "all":
        rows = rows[rows["split"] == args.split]
    index = index_embeddings(args.embeddings_dir)
    rows = rows.assign(feature_path=rows["slide_id"].astype(str).map(index.get))
    missing = int(rows["feature_path"].isna().sum())
    if missing:
        raise SystemExit(f"ERROR: {missing} slides have no embedding in --embeddings-dir")

    write_attention(model, rows.reset_index(drop=True), args.run_dir / "attention.h5",
                    device, feature_norm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
