#!/usr/bin/env python3
"""Score an unlabelled cohort with a trained run's final model.

    python predict.py --run-dir ../../results/all_binary_s0_<jobid> \
                      --embeddings-dir ../../data/robertoherrera_embeddings \
                      --out-dir ../../results/robertoherrera_eval --cohort robertoherrera

No labels are needed or read. Writes to --out-dir:
    predictions.csv   slide_id, prob_ebv, pred_ebv (prob_ebv >= threshold), n_tiles,
                      mpp_base/objective of the source scan
    attention.h5      per-tile attention + tile scores (export_attention.py format)
    model.json        which model/run produced these, threshold, settings
Uses the same bag construction and min-tiles filter as training (cohort.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abmil import ABMIL
from cohort import BagDataset, bag_shape, index_embeddings, read_attrs
from export_attention import write_attention
from train import evaluate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True, help="Holds final.pt (+ report.json).")
    p.add_argument("--embeddings-dir", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cohort", default="external", help="Name recorded in the outputs.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-tiles", type=int, default=16, help="As in training.")
    p.add_argument("--device", default=None)
    p.add_argument("--no-attention", action="store_true")
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.run_dir / "final.pt", map_location="cpu", weights_only=False)
    model = ABMIL.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    feature_norm = checkpoint.get("feature_norm", "none")

    index = index_embeddings(args.embeddings_dir)
    if not index:
        raise SystemExit(f"ERROR: no .h5 embeddings in {args.embeddings_dir}")
    rows = pd.DataFrame({"slide_id": sorted(index)})
    rows["feature_path"] = rows["slide_id"].map(index)
    rows["n_tiles"] = [bag_shape(Path(f))[0] for f in rows["feature_path"]]
    rows["embed_dim"] = [bag_shape(Path(f))[1] for f in rows["feature_path"]]
    if (rows["embed_dim"] != model.input_dim).any():
        raise SystemExit(f"ERROR: embeddings are not {model.input_dim}-d (model input)")
    small = rows[rows["n_tiles"] < args.min_tiles]
    if len(small):
        print(f"excluded {len(small)} slides with <{args.min_tiles} tiles: "
              f"{', '.join(small['slide_id'])}")
    rows = rows[rows["n_tiles"] >= args.min_tiles].reset_index(drop=True)
    rows["label"] = -1                      # unknown; BagDataset needs the column
    rows["cohort"] = args.cohort

    scored = evaluate(model, BagDataset(rows, feature_norm,
                                        neighbours=getattr(model, "graph_layers", 0) > 0), device)
    attrs = [read_attrs(Path(f)) for f in rows["feature_path"]]
    out = pd.DataFrame({
        "slide_id": scored["slide_id"],
        "prob_ebv": scored["prob"].round(6),
        "pred_ebv": (scored["prob"] >= args.threshold).astype(int),
        "n_tiles": rows["n_tiles"],
        "mpp_base": [a.get("mpp_base") for a in attrs],
        "mpp_achieved": [a.get("mpp_achieved") for a in attrs],
    })
    out.to_csv(args.out_dir / "predictions.csv", index=False)
    if not args.no_attention:
        write_attention(model, rows.assign(split=args.cohort),
                        args.out_dir / "attention.h5", device, feature_norm)

    report = json.loads((args.run_dir / "report.json").read_text()) \
        if (args.run_dir / "report.json").exists() else {}
    (args.out_dir / "model.json").write_text(json.dumps({
        "run_dir": str(args.run_dir.resolve()), "cohort": args.cohort,
        "threshold": args.threshold, "label_scheme": checkpoint.get("label_scheme", "binary"),
        "final_epochs": checkpoint.get("epochs"), "feature_norm": feature_norm,
        "model_config": checkpoint["config"], "training_cv": {
            k: report.get(k) for k in ("pooled_oof", "fold_mean_auc", "fold_std_auc")},
        "n_slides_scored": len(out), "excluded_small": list(small["slide_id"]),
    }, indent=2, default=str))
    print(f"scored {len(out)} slides: {int(out['pred_ebv'].sum())} called EBV+ at "
          f"{args.threshold:g}; prob median {out['prob_ebv'].median():.3f}")
    print(f"wrote {args.out_dir}/predictions.csv, model.json"
          + ("" if args.no_attention else ", attention.h5"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
