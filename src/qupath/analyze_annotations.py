#!/usr/bin/env python3
"""Score the pathologist's annotations against the model's attention.

    python analyze_annotations.py --run-dir ../../results/tcga2chile_3class_msiw3_s0_46204792 \
        --annotations <returned>/annotations --form <returned>/review_form.csv

A tile belongs to an annotation class when its centre lies inside an annotation
of that class (holes respected; no shapely needed). Where annotations overlap,
the class listed first in config.ANNOTATION_CLASSES wins. Then, per slide: the
share of top-attention tiles (attn_pct >= config.TOP_ATTENTION_PCT) in each
class, next to the share of all tiles (the baseline the top tiles should beat).
The unblinding key is joined only at the end.

Writes <run-dir>/review/results/
    tile_annotation_overlap.csv   code, tile, attn_pct, tile_logit, class
    summary_by_slide.csv          per slide: top-tile and all-tile share per class + form
    summary_by_group.csv          per group (tp, fn, fp, ...): mean of the slide shares
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

import config
import scaling

CLASS_ORDER = [name for name, _ in config.ANNOTATION_CLASSES]
UNANNOTATED = "unannotated"


def read_annotations(path: Path) -> list[tuple[str, list[list[np.ndarray]]]]:
    """(class name, polygons) per annotation; each polygon is [exterior, *holes].
    Reads QuPath 0.7 GeoJSON: properties.classification.name (or .names)."""
    data = json.loads(path.read_text())
    features = data["features"] if data.get("type") == "FeatureCollection" else data
    out = []
    for f in features:
        cls = (f.get("properties") or {}).get("classification") or {}
        name = cls.get("name") or " : ".join(cls.get("names", [])) or None
        geom = f.get("geometry") or {}
        if geom.get("type") == "Polygon":
            polys = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue  # points/lines carry no area
        out.append((name, [[np.asarray(ring, dtype=float) for ring in poly] for poly in polys]))
    return out


def inside(points: np.ndarray, polygons: list[list[np.ndarray]]) -> np.ndarray:
    hit = np.zeros(len(points), dtype=bool)
    for exterior, *holes in polygons:
        mask = MplPath(exterior).contains_points(points)
        for hole in holes:
            mask &= ~MplPath(hole).contains_points(points)
        hit |= mask
    return hit


def tile_classes(centres: np.ndarray, annotations) -> np.ndarray:
    classes = np.full(len(centres), UNANNOTATED, dtype=object)
    ranked = sorted(annotations, key=lambda a: CLASS_ORDER.index(a[0])
                    if a[0] in CLASS_ORDER else len(CLASS_ORDER))
    for name, polygons in reversed(ranked):        # earlier classes overwrite later ones
        classes[inside(centres, polygons)] = name or "unclassified"
    return classes


def shares(classes: pd.Series, prefix: str) -> dict[str, float]:
    counts = classes.value_counts(normalize=True)
    return {f"{prefix}{c}": float(counts.get(c, 0.0)) for c in CLASS_ORDER + [UNANNOTATED]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True, help="Holds attention.h5.")
    p.add_argument("--review-dir", type=Path, default=None, help="Default: <run-dir>/review")
    p.add_argument("--annotations", type=Path, required=True,
                   help="The returned annotations/ folder (<code>.geojson).")
    p.add_argument("--form", type=Path, required=True, help="The returned review_form.csv.")
    args = p.parse_args()
    review_dir = args.review_dir or args.run_dir / "review"
    out_dir = review_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    review = pd.read_csv(review_dir / "review_set.csv")
    tiles, slides, not_returned = [], [], []
    with h5py.File(args.run_dir / "attention.h5", "r") as h:
        for row in review.itertuples():
            path = args.annotations / f"{row.code}.geojson"
            if not path.exists():
                not_returned.append(row.code)
                continue
            group = h[row.slide_id]
            size = int(group.attrs["patch_size_level0"])
            centres = group["coords"][:].astype(float) + size / 2
            pct = scaling.percentile(group["attention"][:].astype(float))
            classes = tile_classes(centres, read_annotations(path))
            frame = pd.DataFrame({"code": row.code, "tile": np.arange(len(pct)),
                                  "attn_pct": pct, "tile_logit": group["tile_logit"][:],
                                  "class": classes})
            tiles.append(frame)
            top = frame[frame["attn_pct"] >= config.TOP_ATTENTION_PCT]
            slides.append({"code": row.code, "n_tiles": len(frame), "n_top": len(top),
                           **shares(top["class"], "top_share_"),
                           **shares(frame["class"], "all_share_")})
    if not_returned:
        print(f"no annotations returned for {len(not_returned)} slides: {', '.join(not_returned)}")
    if not slides:
        raise SystemExit("ERROR: no annotation files matched review_set.csv codes")

    pd.concat(tiles).to_csv(out_dir / "tile_annotation_overlap.csv", index=False)
    key = pd.read_csv(review_dir / "KEY_unblinding.csv")
    by_slide = (pd.DataFrame(slides).merge(pd.read_csv(args.form), on="code", how="left")
                .merge(key, on="code", how="left"))
    by_slide.to_csv(out_dir / "summary_by_slide.csv", index=False)
    share_cols = [c for c in by_slide if c.startswith(("top_share_", "all_share_"))]
    by_group = by_slide.groupby("group")[share_cols].mean()
    by_group.insert(0, "n_slides", by_slide.groupby("group").size())
    by_group.to_csv(out_dir / "summary_by_group.csv")

    lymph = "Lymphoid stroma"
    print(f"\n{lymph}: share of top-{100 - config.TOP_ATTENTION_PCT:g}% attention tiles "
          "vs share of all tiles, by group")
    for g, r in by_group.iterrows():
        print(f"  {g:<9} n={int(r['n_slides']):>2}  top {r['top_share_' + lymph]:.2f}  "
              f"all {r['all_share_' + lymph]:.2f}")
    print(f"\nwrote {out_dir}/tile_annotation_overlap.csv, summary_by_slide.csv, "
          "summary_by_group.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
