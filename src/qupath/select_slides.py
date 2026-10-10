#!/usr/bin/env python3
"""Pick a stratified, blinded review set from one run's external predictions.

    python select_slides.py --run-dir ../../results/tcga2chile_3class_msiw3_s0_46204792

Writes to <run-dir>/review/ (or --out-dir):
    review_set.csv        code, slide_id, svs_name   (our side; drives export/packaging)
    KEY_unblinding.csv    code, slide_id, cohort, label, prob, group, tcga_subtype
                          (kept by us; never copied into the package)

Groups, from the final model at config.THRESHOLD:
    tp, fn                          every EBV+ slide
    fp, tn                          EBV- slides without an MSI annotation (Chile)
    fp_msi, fp_other, tn_msi, tn_other   TCGA EBV- slides, by tcga_subtype == STAD_MSI
Counts come from config.SAMPLING_PLAN (or --plan). Groups absent from the run are
skipped and short groups are reported, never padded from other groups.
Sampling is seeded; codes are assigned after a shuffle, so code order carries
no information about group.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import config

MSI_SUBTYPE = "STAD_MSI"


def load_predictions(run_dir: Path, labels_csv: Path) -> pd.DataFrame:
    """external_predictions.csv + tcga_subtype + svs file name; adds `group`."""
    preds = pd.read_csv(run_dir / "external_predictions.csv")
    labels = pd.read_csv(labels_csv, usecols=["slide_id", "tcga_subtype"])
    preds = preds.merge(labels, on="slide_id", how="left")
    with h5py.File(run_dir / "attention.h5", "r") as h:
        missing = sorted(set(preds["slide_id"]) - set(h))
        if missing:
            raise SystemExit(f"ERROR: {len(missing)} external slides missing from attention.h5 "
                             f"(e.g. {missing[0]}); re-run export_attention.py")
        preds["svs_name"] = [Path(str(h[s].attrs["slide_path"])).name for s in preds["slide_id"]]
    preds["group"] = [assign_group(r, config.THRESHOLD) for r in preds.itertuples()]
    return preds


def assign_group(row, threshold: float) -> str:
    called = row.prob >= threshold
    if row.label == 1:
        return "tp" if called else "fn"
    outcome = "fp" if called else "tn"
    if pd.notna(row.tcga_subtype):
        outcome += "_msi" if row.tcga_subtype == MSI_SUBTYPE else "_other"
    return outcome


def sample(frame: pd.DataFrame, plan: dict[str, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked = []
    available = frame["group"].value_counts()
    for group, wanted in plan.items():
        have = int(available.get(group, 0))
        if wanted == 0 or have == 0:
            continue
        take = min(wanted, have)
        rows = frame[frame["group"] == group]
        picked.append(rows.iloc[np.sort(rng.choice(have, take, replace=False))])
        note = "" if take == wanted else f"  (short: wanted {wanted})"
        print(f"  {group:<9} {take:>3} of {have:>3} available{note}")
    unplanned = sorted(set(available.index) - {g for g, n in plan.items() if n})
    if unplanned:
        print(f"  not sampled (not in plan): {', '.join(unplanned)}")
    return pd.concat(picked, ignore_index=True)


def blind(selected: pd.DataFrame, prefix: str, seed: int) -> pd.DataFrame:
    shuffled = selected.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    width = max(2, len(str(len(shuffled))))
    shuffled.insert(0, "code", [f"{prefix}{i + 1:0{width}d}" for i in range(len(shuffled))])
    return shuffled


def parse_plan(text: str | None) -> dict[str, int]:
    if not text:
        return dict(config.SAMPLING_PLAN)
    return {k: int(v) for k, v in (item.split("=") for item in text.split(","))}


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <run-dir>/review")
    p.add_argument("--labels", type=Path,
                   default=here.parent.parent / "metadata/cohort_labels.csv")
    p.add_argument("--plan", default=None, help="e.g. tp=10,fn=7,fp=10,tn=9 (default: config)")
    args = p.parse_args()
    out_dir = args.out_dir or args.run_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(args.run_dir, args.labels)
    print(f"{args.run_dir.name}: {len(preds)} external slides; sampling:")
    selected = blind(sample(preds, parse_plan(args.plan), config.RANDOM_SEED),
                     config.REVIEW_CODE_PREFIX, config.RANDOM_SEED)

    selected[["code", "slide_id", "svs_name"]].to_csv(out_dir / "review_set.csv", index=False)
    selected[["code", "slide_id", "cohort", "label", "prob", "group", "tcga_subtype"]].to_csv(
        out_dir / "KEY_unblinding.csv", index=False)
    print(f"selected {len(selected)} slides -> {out_dir}/review_set.csv, KEY_unblinding.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
