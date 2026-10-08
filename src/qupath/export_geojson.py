#!/usr/bin/env python3
"""attention.h5 -> one QuPath-importable GeoJSON per review slide (QuPath 0.7.0).

    python export_geojson.py --run-dir ../../results/tcga2chile_3class_msiw3_s0_46204792

Reads <run-dir>/attention.h5 and <run-dir>/review/review_set.csv; writes
<run-dir>/review/package/overlays/<code>.geojson plus overlays_manifest.csv.

One Feature per model tile: a square Polygon in level-0 pixel coordinates
(QuPath's image space for .svs; origin top-left), side = patch_size_level0.
Properties, in the format QuPath 0.7 reads (QuPathTypeAdapters /
GsonTools.PathClassTypeAdapter):
    objectType      "tile"  -- lightweight, separate from the reviewer's annotations
    isLocked        true    -- cannot be moved or edited by accident
    classification  {"name": <attention band>, "color": [r, g, b]}
    measurements    {"attn_pct": .., "attn_rel_log2": .., "tile_score": ..}
Files are named by blinded code, never by slide_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import config
import scaling

BAND_COLOURS = {name: list(rgb) for name, _, _, rgb in config.ATTENTION_BANDS}


def read_slide(attention_h5: h5py.File, slide_id: str) -> dict:
    group = attention_h5[slide_id]
    return {"coords": group["coords"][:].astype(np.int64),
            "attention": group["attention"][:].astype(np.float64),
            "tile_logit": group["tile_logit"][:].astype(np.float64),
            "size": int(group.attrs["patch_size_level0"])}


def tile_polygon(x0: int, y0: int, size: int) -> list:
    x1, y1 = x0 + size, y0 + size
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def feature(polygon: list, measurements: dict[str, float], band: str) -> dict:
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": polygon},
            "properties": {"objectType": "tile", "isLocked": True,
                           "classification": {"name": band, "color": BAND_COLOURS[band]},
                           "measurements": measurements}}


def slide_features(slide: dict, score_limit: float) -> dict:
    pct = scaling.percentile(slide["attention"])
    rel = scaling.rel_log2(slide["attention"], config.ATTN_REL_LOG2_CLIP)
    score = scaling.clip_tile_score(slide["tile_logit"], score_limit)
    bands = scaling.band(pct, config.ATTENTION_BANDS)
    d = config.MEASUREMENT_DECIMALS
    features = [feature(tile_polygon(int(x), int(y), slide["size"]),
                        {"attn_pct": round(float(p), d), "attn_rel_log2": round(float(r), d),
                         "tile_score": round(float(s), d)}, b)
                for (x, y), p, r, s, b in zip(slide["coords"], pct, rel, score, bands)]
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True, help="Holds attention.h5.")
    p.add_argument("--review-dir", type=Path, default=None, help="Default: <run-dir>/review")
    args = p.parse_args()
    review_dir = args.review_dir or args.run_dir / "review"
    out_dir = review_dir / "package" / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    review = pd.read_csv(review_dir / "review_set.csv")
    with h5py.File(args.run_dir / "attention.h5", "r") as h:
        slides = {row.code: read_slide(h, row.slide_id) for row in review.itertuples()}

    limit = scaling.tile_score_limit([s["tile_logit"] for s in slides.values()],
                                     config.TILE_SCORE_CLIP_QUANTILE)
    print(f"tile_score clipped to +/-{limit:.3f} "
          f"(|tile_logit| q{config.TILE_SCORE_CLIP_QUANTILE:g} over {len(slides)} slides)")

    manifest = []
    for code, slide in slides.items():
        if (slide["coords"] < 0).any():
            raise SystemExit(f"ERROR: {code} has negative tile coordinates")
        collection = slide_features(slide, limit)
        path = out_dir / f"{code}.geojson"
        path.write_text(json.dumps(collection, separators=(",", ":")))
        manifest.append({"code": code, "n_tiles": len(slide["coords"]),
                         "tile_size_level0": slide["size"],
                         "tile_score_limit": round(limit, 4),
                         "mb": round(path.stat().st_size / 1e6, 2)})
    pd.DataFrame(manifest).to_csv(review_dir / "overlays_manifest.csv", index=False)
    print(f"wrote {len(manifest)} overlays to {out_dir} "
          f"({sum(m['mb'] for m in manifest):.0f} MB total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
