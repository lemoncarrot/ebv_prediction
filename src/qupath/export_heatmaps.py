#!/usr/bin/env python3
"""MIL heatmaps (additive: exact; gated attention: approximate) as QuPath GeoJSON overlays.

    python export_heatmaps.py --attention ../../results/robertoherrera_3class_additive/attention.h5 \
                              --out-dir ../../qupath_export/robertoherrera_3class_additive

For a model trained with --aggregation additive, attention.h5 holds the exact
per-tile contribution a_i * g(h_i) to each class logit. Per tile this writes
(QuPath 0.7 GeoJSON, same tile format as export_geojson.py):

    ebv_evidence  n * a_i * (g_ebv(h_i) - mean of the other class logits)
                  [binary model: n * a_i * g(h_i)]. Signed: >0 pushes towards
                  EBV, <0 away. Its mean over the slide's tiles is exactly the
                  slide's EBV margin (EBV logit minus mean other logit). Scaled by
                  n (tiles) so a tile with average attention has weight 1; heavy-tailed
                  (a few tiles carry most attention). Clipped to
                  +/- a limit shared by the whole cohort (99th pct of |value|).
    tile_score    (gated-attention models, which have no exact decomposition) the head
                  applied to each tile alone, EBV log-odds; approximate, signed; clipped
                  the same way
    attn_pct      within-slide attention percentile, 0-100
    classification  attention band (top 5% / 5-20% / rest), as in the review package

Files are named <slide_id>.geojson to match the slide file name. Also writes
slides.csv (slide_id, prob, call at 0.5, label if known, tiles, evidence limit).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import scaling  # noqa: E402
from export_geojson import feature, tile_polygon  # noqa: E402

EBV, CLIP_QUANTILE = 1, 0.99


def evidence(group: h5py.Group) -> tuple[str, np.ndarray]:
    """(measurement name, per-tile values). Additive models: exact signed evidence.
    Gated-attention models have no exact decomposition, so fall back to
    tile_score: the head applied to each tile alone (approximate, signed)."""
    if "contribution" not in group:
        return "tile_score", group["tile_logit"][:].astype(np.float64)
    c = group["contribution"][:].astype(np.float64)              # (n, K)
    margin = c[:, 0] if c.shape[1] == 1 else c[:, EBV] - np.delete(c, EBV, axis=1).mean(1)
    return "ebv_evidence", margin * len(c)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--attention", type=Path, required=True, help="attention.h5 of an additive model")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--slides", nargs="+", default=None,
                   help="Only these slide_ids (colour limit still uses the whole cohort).")
    args = p.parse_args()
    overlays = args.out_dir / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.attention, "r") as h:
        kind, _ = evidence(h[next(iter(h))])
        ev = {s: evidence(h[s])[1] for s in h}
        slides = sorted(args.slides or h)
        missing = sorted(set(slides) - set(h))
        if missing:
            raise SystemExit(f"ERROR: not in attention.h5: {missing}")
        limit = float(np.quantile(np.abs(np.concatenate(list(ev.values()))), CLIP_QUANTILE))
        rows = []
        for s in slides:
            g = h[s]
            coords, size = g["coords"][:].astype(np.int64), int(g.attrs["patch_size_level0"])
            pct = scaling.percentile(g["attention"][:].astype(np.float64))
            bands = scaling.band(pct, config.ATTENTION_BANDS)
            clipped = np.clip(ev[s], -limit, limit)
            d = config.MEASUREMENT_DECIMALS
            feats = [feature(tile_polygon(int(x), int(y), size),
                             {kind: round(float(e), d), "attn_pct": round(float(a), d)}, b)
                     for (x, y), e, a, b in zip(coords, clipped, pct, bands)]
            (overlays / f"{s}.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
            label = int(g.attrs.get("label", -1))
            rows.append({"slide_id": s, "prob_ebv": round(float(g.attrs["prob"]), 4),
                         "call_at_0.5": "EBV+" if g.attrs["prob"] >= 0.5 else "EBV-",
                         "label": {1: "EBV+", 0: "EBV-"}.get(label, "unknown"),
                         "n_tiles": len(coords), "measure": kind,
                         "colour_limit": round(limit, 3)})
    with open(args.out_dir / "slides.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mb = sum(p.stat().st_size for p in overlays.glob("*.geojson")) / 1e6
    print(f"{len(rows)} overlays -> {overlays} ({mb:.0f} MB); {kind} clipped to "
          f"+/-{limit:.2f}; {sum(r['call_at_0.5'] == 'EBV+' for r in rows)} slides called EBV+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
