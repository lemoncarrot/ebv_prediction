#!/usr/bin/env python3
"""Assemble the folder that ships to the pathologist.

    python build_package.py --run-dir ../../results/tcga2chile_3class_s0_46167285 \
                            --slides-dir ../../data/original_chile --copy-slides

    <run-dir>/review/package/
        README.md               from templates/, filled from config.py
        review_form.csv         one row per code
        overlays/R01.geojson    from export_geojson.py
        scripts/import_overlays.groovy, export_annotations.groovy
        slides/R01.svs ...      with --copy-slides: hard-linked (or copied) under the code

Without --copy-slides, <review-dir>/slides_to_copy.csv lists code -> original
file, kept on our side (original names stay out of the package). Copying tens of
GB of .svs is heavy I/O: run this in a job, not on a login node. On the same
filesystem slides are hard-linked instead (instant, no extra space; the originals
are untouched, and transferring the package still sends full files).
Refuses to finish if any code lacks an overlay or slide, or if the unblinding
key ended up inside the package.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

import config

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
GROOVY = HERE / "groovy"


def groovy_classes() -> str:
    rows = ",\n    ".join(f'["{name}", {r}, {g}, {b}]' for name, (r, g, b)
                          in config.ANNOTATION_CLASSES)
    return f"[\n    {rows}\n]"


def render_scripts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(GROOVY.glob("*.groovy")):
        text = path.read_text().replace("{{ANNOTATION_CLASSES_GROOVY}}", groovy_classes())
        (out_dir / path.name).write_text(text)


def render_readme(out: Path, n_slides: int, score_limit: float) -> None:
    fields = {
        "{{N_SLIDES}}": str(n_slides),
        "{{QUPATH_VERSION}}": config.QUPATH_VERSION,
        "{{BANDS}}": ", ".join(f'"{name}"' for name, *_ in config.ATTENTION_BANDS),
        "{{TILE_SCORE_LIMIT}}": f"{score_limit:.2f}",
        "{{ANNOTATION_CLASSES}}": "\n".join(f"   - {name}" for name, _ in
                                             config.ANNOTATION_CLASSES),
    }
    text = (TEMPLATES / "README_pathologist.md").read_text()
    for key, value in fields.items():
        text = text.replace(key, value)
    if "{{" in text:
        raise SystemExit("ERROR: unfilled field in README template")
    out.write_text(text)


def render_form(out: Path, codes: list[str]) -> None:
    header = (TEMPLATES / "review_form.csv").read_text().strip()
    width = header.count(",")
    out.write_text("\n".join([header] + [code + "," * width for code in codes]) + "\n")


def link_or_copy(source: Path, target: Path) -> None:
    """Hard link when source and target share a filesystem, else copy."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def place_slides(review: pd.DataFrame, slides_dir: Path, package: Path, copy: bool,
                 review_dir: Path) -> None:
    sources = {row.code: slides_dir / row.svs_name for row in review.itertuples()}
    missing = [f"{c}: {p.name}" for c, p in sources.items() if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} slides not in {slides_dir}, e.g. {missing[:3]}")
    if not copy:
        review.assign(source=[str(p) for p in sources.values()])[["code", "svs_name", "source"]]\
            .to_csv(review_dir / "slides_to_copy.csv", index=False)
        print(f"slides found; list written to {review_dir / 'slides_to_copy.csv'}")
        return
    out = package / "slides"
    out.mkdir(exist_ok=True)
    for code, source in sources.items():
        target = out / f"{code}{source.suffix}"
        if not (target.exists() and target.stat().st_size == source.stat().st_size):
            link_or_copy(source, target)
    total = sum(p.stat().st_size for p in out.iterdir()) / 1e9
    print(f"placed {len(sources)} slides in {out} ({total:.1f} GB, hard-linked where possible)")


def check_package(package: Path, codes: list[str]) -> None:
    missing = [c for c in codes if not (package / "overlays" / f"{c}.geojson").exists()]
    if missing:
        raise SystemExit(f"ERROR: no overlay for {missing}; run export_geojson.py")
    if any(package.rglob("KEY_unblinding*")):
        raise SystemExit("ERROR: the unblinding key is inside the package; remove it")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--review-dir", type=Path, default=None, help="Default: <run-dir>/review")
    p.add_argument("--slides-dir", type=Path, default=None,
                   help="Folder holding the original .svs files (matched by file name).")
    p.add_argument("--copy-slides", action="store_true",
                   help="Place slides in package/slides/ under their code (hard link on the "
                        "same filesystem: no extra space; else a copy).")
    args = p.parse_args()
    review_dir = args.review_dir or args.run_dir / "review"
    package = review_dir / "package"

    review = pd.read_csv(review_dir / "review_set.csv")
    codes = list(review["code"])
    manifest = pd.read_csv(review_dir / "overlays_manifest.csv")
    check_package(package, codes)

    render_scripts(package / "scripts")
    render_readme(package / "README.md", len(codes), float(manifest["tile_score_limit"].iloc[0]))
    render_form(package / "review_form.csv", codes)
    if args.slides_dir:
        place_slides(review, args.slides_dir, package, args.copy_slides, review_dir)
    elif args.copy_slides:
        raise SystemExit("ERROR: --copy-slides needs --slides-dir")
    else:
        print("no --slides-dir: slides not checked or copied")
    check_package(package, codes)
    print(f"package ready: {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
