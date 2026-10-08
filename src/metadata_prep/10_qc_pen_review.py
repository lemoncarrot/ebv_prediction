#!/usr/bin/env python
"""Subsample preprocessed slides for visual pen-mark review.

Joins `metadata/cohort_labels.csv` against the preprocessing manifest and the
per-slide HDF5 attributes, keeps only slides that are in the metadata *and*
carry an EBV call, then writes a small inspection folder: downscaled copies of
the preprocessing QC images (tissue contours in green, excluded holes in red,
removed surgical-marker ink tinted magenta) plus contact sheets covering the
whole eligible set.

`pen_fraction_removed` -- the share of segmented tissue area the pen filter
deleted -- is read straight from each patch HDF5, so the ranking here is the
pipeline's own measurement, not a re-estimate. It only counts ink the filter
*caught*; a slide whose ink slipped through scores 0. That is precisely why the
contact sheets cover every eligible slide and not just the top of the ranking.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
EBV_CALLS = ("Positive", "Negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=REPO / "metadata/cohort_labels.csv")
    parser.add_argument(
        "--preprocessed-dir", type=Path, nargs="+", default=[REPO / "01_preprocessed_tcga"],
        help="Preprocessing output directories, each holding patches/, qc/ and manifest.csv.",
    )
    parser.add_argument(
        "--cohort", nargs="+", default=["tcga"],
        help="Cohort values in the labels file to review. Use 'all' for every cohort.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "03_qc_pen_review")
    parser.add_argument(
        "--n", type=int, default=60,
        help="Size of the stratified subsample written to thumbs/. 0 writes every eligible slide.",
    )
    parser.add_argument(
        "--all-positives", action=argparse.BooleanOptionalAction, default=True,
        help="Force every EBV-positive slide into the subsample (default: on -- they are the scarce class).",
    )
    parser.add_argument("--thumb-width", type=int, default=1400, help="Long-edge width of a subsample thumbnail.")
    parser.add_argument("--cell-width", type=int, default=420, help="Cell width in the contact sheets.")
    parser.add_argument("--cols", type=int, default=6, help="Contact-sheet columns.")
    parser.add_argument("--rows", type=int, default=5, help="Contact-sheet rows per page.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-leak-scan", dest="leak_scan", action="store_false",
        help="Skip the missed-ink measurement (it re-reads every QC image).",
    )
    parser.add_argument("--violet-hue-lo", type=float, default=125.0, help="Violet-ink hue band, OpenCV scale.")
    parser.add_argument("--violet-hue-hi", type=float, default=175.0)
    parser.add_argument(
        "--violet-sat", type=float, default=180.0,
        help="Saturation above which a violet pixel is marker rather than hematoxylin.",
    )
    parser.add_argument("--dark-value", type=float, default=70.0, help="Value below which a low-saturation pixel is black marker.")
    parser.add_argument(
        "--no-contact-sheets", dest="contact_sheets", action="store_false",
        help="Skip the full-cohort contact sheets and write only the subsample.",
    )
    return parser.parse_args()


def read_labels(path: Path, cohorts: list[str]) -> dict[str, dict[str, str]]:
    keep_all = "all" in cohorts
    rows = {}
    for row in csv.DictReader(path.open()):
        if keep_all or row["cohort"] in cohorts:
            rows[row["slide_id"]] = row
    return rows


def read_manifests(dirs: list[Path]) -> dict[str, dict[str, str]]:
    """Manifest rows keyed by slide stem, remembering which directory owned them."""
    rows = {}
    for directory in dirs:
        manifest = directory / "manifest.csv"
        if not manifest.exists():
            print(f"warning: no manifest at {manifest}", file=sys.stderr)
            continue
        for row in csv.DictReader(manifest.open()):
            stem = Path(row["slide"]).stem
            row["_dir"] = str(directory)
            rows[stem] = row
    return rows


def slide_attrs(patch_path: Path) -> dict[str, float]:
    with h5py.File(patch_path, "r") as handle:
        attrs = handle.attrs
        return {
            "pen_filter": bool(attrs.get("pen_filter", False)),
            "pen_fraction_removed": float(attrs.get("pen_fraction_removed", float("nan"))),
            "level0_width": int(attrs["level0_width"]),
            "level0_height": int(attrs["level0_height"]),
            "patch_mpp": float(attrs.get("patch_mpp", float("nan"))),
        }


def hsv_like_opencv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSV on OpenCV's scales (H 0-179, S/V 0-255) without requiring cv2 here."""
    scaled = rgb.astype(np.float32) / 255.0
    high = scaled.max(2)
    delta = high - scaled.min(2)
    red, green, blue = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    hue = np.zeros_like(high)
    lit = delta > 0
    sel = lit & (high == red)
    hue[sel] = ((green - blue)[sel] / delta[sel]) % 6
    sel = lit & (high == green)
    hue[sel] = ((blue - red)[sel] / delta[sel]) + 2
    sel = lit & (high == blue)
    hue[sel] = ((red - green)[sel] / delta[sel]) + 4
    sat = np.where(high > 0, delta / np.maximum(high, 1e-6), 0) * 255
    return hue * 30.0, sat, high * 255


def missed_ink_fractions(patch_path: Path, qc_path: Path, args: argparse.Namespace) -> dict[str, float]:
    """Share of *retained* patches whose centre sits on ink the pen filter kept.

    `pen_fraction_removed` only scores ink the filter caught, so it cannot reveal
    a colour band the filter does not look at. This re-reads the QC image -- where
    removed ink is already tinted magenta, hence excluded by construction -- and
    asks how many surviving patch centres land on untinted violet or black marker.
    It is a lower bound (centre-point test, at segmentation resolution), but it is
    measured on the patches that actually reach the encoder.
    """
    with h5py.File(patch_path, "r") as handle:
        coords = handle["coords"][:]
        attrs = handle.attrs
        ds_x = float(attrs["segmentation_downsample_x"])
        ds_y = float(attrs["segmentation_downsample_y"])
        patch_level0 = int(attrs["patch_size_level0"])
    if not len(coords):
        return {"frac_patches_on_violet_ink": 0.0, "frac_patches_on_dark_ink": 0.0}

    image = np.asarray(Image.open(qc_path).convert("RGB"))
    hue, sat, val = hsv_like_opencv(image)
    # Violet marker overlaps hematoxylin in hue, so saturation is what separates
    # them: ink is near-pure pigment, nuclei are not. Black marker is dark and
    # nearly colourless. The magenta QC tint lands outside both bands.
    violet = (hue >= args.violet_hue_lo) & (hue < args.violet_hue_hi) & (sat >= args.violet_sat) & (val >= 60)
    dark = (val < args.dark_value) & (sat < 110)

    cx = np.clip(((coords[:, 0] + patch_level0 / 2) / ds_x).astype(int), 0, image.shape[1] - 1)
    cy = np.clip(((coords[:, 1] + patch_level0 / 2) / ds_y).astype(int), 0, image.shape[0] - 1)
    return {
        "frac_patches_on_violet_ink": round(float(violet[cy, cx].mean()), 5),
        "frac_patches_on_dark_ink": round(float(dark[cy, cx].mean()), 5),
    }


def build_records(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    labels = read_labels(args.labels, args.cohort)
    manifest = read_manifests(args.preprocessed_dir)

    tally = Counter()
    records = []
    for slide_id, row in sorted(labels.items()):
        tally["in_metadata"] += 1
        has_call = row["ebv_status"] in EBV_CALLS
        tally["with_ebv_call" if has_call else "no_ebv_call"] += 1
        if not has_call:
            continue

        man = manifest.get(slide_id)
        status = man["status"] if man else "not_preprocessed"
        tally[f"status_{status}"] += 1
        if status != "ok":
            records.append({
                "slide_id": slide_id, "cohort": row["cohort"], "patient_id": row["patient_id"],
                "ebv_status": row["ebv_status"], "label": row["label"], "status": status,
                "message": (man or {}).get("message", "no manifest row"),
                "patches": "", "pen_fraction_removed": "", "pen_filter": "",
                "level0_width": "", "level0_height": "", "qc_image": "", "eligible": 0,
                "frac_patches_on_violet_ink": "", "frac_patches_on_dark_ink": "",
            })
            continue

        directory = Path(man["_dir"])
        patch_path = directory / "patches" / f"{slide_id}.h5"
        qc_path = directory / "qc" / f"{slide_id}.jpg"
        if not patch_path.exists():
            tally["missing_h5"] += 1
            continue
        attrs = slide_attrs(patch_path)
        if not qc_path.exists():
            tally["missing_qc_image"] += 1
        leak = {"frac_patches_on_violet_ink": "", "frac_patches_on_dark_ink": ""}
        if args.leak_scan and qc_path.exists():
            leak = missed_ink_fractions(patch_path, qc_path, args)
        records.append({
            "slide_id": slide_id, "cohort": row["cohort"], "patient_id": row["patient_id"],
            "ebv_status": row["ebv_status"], "label": row["label"], "status": status,
            "message": man.get("message", ""), "patches": man.get("patches", ""),
            "pen_fraction_removed": round(attrs["pen_fraction_removed"], 6),
            "pen_filter": int(attrs["pen_filter"]),
            "level0_width": attrs["level0_width"], "level0_height": attrs["level0_height"],
            "qc_image": str(qc_path) if qc_path.exists() else "",
            "eligible": int(bool(qc_path.exists())),
            **leak,
        })
        tally["eligible"] += int(bool(qc_path.exists()))

    records.sort(key=lambda r: (-(r["pen_fraction_removed"] or 0) if r["eligible"] else 1, r["slide_id"]))
    for rank, record in enumerate(r for r in records if r["eligible"]):
        record["pen_rank"] = rank
    for record in records:
        record.setdefault("pen_rank", "")
    return records, tally


def choose_subsample(records: list[dict], args: argparse.Namespace) -> None:
    """Mark `selected` on a stratified slice across the pen-fraction range.

    Pen artifacts are not uniformly distributed, so a uniform random draw would
    be dominated by the clean majority. Sampling within pen-fraction bins keeps
    the heavy, marginal, and clean regimes all represented -- the marginal ones
    decide whether the current filter threshold needs to move.
    """
    eligible = [r for r in records if r["eligible"]]
    for record in records:
        record["selected"] = 0
    if args.n <= 0 or args.n >= len(eligible):
        for record in eligible:
            record["selected"] = 1
        return

    rng = random.Random(args.seed)
    picked: set[str] = set()
    if args.all_positives:
        picked.update(r["slide_id"] for r in eligible if r["ebv_status"] == "Positive")

    # Slides where the filter *missed* ink are the ones a threshold change would
    # act on, and they can score 0 on pen_fraction_removed, so the pen-fraction
    # bins below would never surface them. Force the worst offenders in.
    leaked = [r for r in eligible if isinstance(r.get("frac_patches_on_violet_ink"), float)]
    leaked.sort(key=lambda r: -r["frac_patches_on_violet_ink"])
    picked.update(r["slide_id"] for r in leaked[:12] if r["frac_patches_on_violet_ink"] > 0.005)

    # Bins in pen fraction: heavy / moderate / trace / none. Upper bins are small
    # in count but carry the whole question, so they are sampled hardest.
    bins = [
        ("heavy", lambda p: p >= 0.10),
        ("moderate", lambda p: 0.01 <= p < 0.10),
        ("trace", lambda p: 0.0 < p < 0.01),
        ("none", lambda p: p <= 0.0),
    ]
    weights = {"heavy": 0.45, "moderate": 0.25, "trace": 0.15, "none": 0.15}
    remaining = max(0, args.n - len(picked))
    for name, test in bins:
        members = [r for r in eligible if test(r["pen_fraction_removed"]) and r["slide_id"] not in picked]
        quota = min(len(members), int(round(remaining * weights[name])))
        if name == "heavy":
            members.sort(key=lambda r: -r["pen_fraction_removed"])  # worst offenders first, not random
            picked.update(r["slide_id"] for r in members[:quota])
        else:
            picked.update(r["slide_id"] for r in rng.sample(members, quota))

    # Top up from anything still unpicked if rounding left the quota short.
    leftover = [r for r in eligible if r["slide_id"] not in picked]
    rng.shuffle(leftover)
    for record in leftover[: max(0, args.n - len(picked))]:
        picked.add(record["slide_id"])

    for record in eligible:
        record["selected"] = int(record["slide_id"] in picked)


def short_name(slide_id: str) -> str:
    """TCGA barcodes carry a long UUID suffix that is noise in a filename."""
    head = slide_id.split(".")[0]
    return head[:48]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def caption(image: Image.Image, text: str, font: ImageFont.ImageFont, height: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "black")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).text((6, image.height + 2), text, fill="white", font=font)
    return canvas


def fit(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image
    return image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)


def write_thumbs(records: list[dict], out_dir: Path, args: argparse.Namespace) -> int:
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for stale in thumbs.glob("*.jpg"):
        stale.unlink()
    font = load_font(20)
    written = 0
    for record in records:
        if not record.get("selected"):
            continue
        pen = record["pen_fraction_removed"]
        tag = "POS" if record["ebv_status"] == "Positive" else "NEG"
        # Rank prefix so a plain filename sort puts the worst pen first.
        name = f"r{record['pen_rank']:03d}_pen{pen * 100:05.1f}_{tag}_{short_name(record['slide_id'])}.jpg"
        image = fit(Image.open(record["qc_image"]).convert("RGB"), args.thumb_width)
        leak = record.get("frac_patches_on_violet_ink")
        leak_text = f"  missed-ink patches={leak * 100:.2f}%" if isinstance(leak, float) else ""
        label = (
            f"{short_name(record['slide_id'])}  EBV={record['ebv_status']}  "
            f"pen removed={pen * 100:.2f}%{leak_text}  patches={record['patches']}"
        )
        caption(image, label, font, 26).save(thumbs / name, quality=88)
        written += 1
    return written


def write_contact_sheets(records: list[dict], out_dir: Path, args: argparse.Namespace) -> int:
    sheets = out_dir / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for stale in sheets.glob("*.jpg"):
        stale.unlink()
    font = load_font(15)
    eligible = [r for r in records if r["eligible"]]
    per_page = args.cols * args.rows
    cell_w = args.cell_width
    cell_h = round(cell_w * 0.8) + 22
    pages = math.ceil(len(eligible) / per_page) if eligible else 0
    for page in range(pages):
        chunk = eligible[page * per_page : (page + 1) * per_page]
        sheet = Image.new("RGB", (args.cols * cell_w, args.rows * cell_h), "black")
        for index, record in enumerate(chunk):
            image = Image.open(record["qc_image"]).convert("RGB")
            image.thumbnail((cell_w - 4, cell_h - 26), Image.LANCZOS)
            tag = "POS" if record["ebv_status"] == "Positive" else "NEG"
            cell = Image.new("RGB", (cell_w, cell_h), "black")
            cell.paste(image, ((cell_w - image.width) // 2, 2))
            text = f"#{record['pen_rank']} {tag} pen={record['pen_fraction_removed'] * 100:.1f}%  {short_name(record['slide_id'])[:34]}"
            ImageDraw.Draw(cell).text((4, cell_h - 20), text, fill="#ffe066" if tag == "POS" else "white", font=font)
            sheet.paste(cell, ((index % args.cols) * cell_w, (index // args.cols) * cell_h))
        sheet.save(sheets / f"contact_sheet_{page + 1:02d}_of_{pages:02d}.jpg", quality=85)
    return pages


def write_report(records: list[dict], tally: Counter, out_dir: Path, args: argparse.Namespace, n_thumbs: int, n_sheets: int) -> str:
    eligible = [r for r in records if r["eligible"]]
    pen = np.array([r["pen_fraction_removed"] for r in eligible], dtype=float)
    lines = [
        "Pen-mark QC review",
        "==================",
        f"labels file      : {args.labels}",
        f"preprocessed dirs: {', '.join(str(d) for d in args.preprocessed_dir)}",
        f"cohorts          : {', '.join(args.cohort)}",
        "",
        "Metadata join",
        "-------------",
        f"slides in metadata           : {tally['in_metadata']}",
        f"  with an EBV call           : {tally['with_ebv_call']}",
        f"  without an EBV call (drop) : {tally['no_ebv_call']}",
    ]
    for key in sorted(k for k in tally if k.startswith("status_")):
        lines.append(f"  preprocessing {key[7:]:<16}: {tally[key]}")
    if tally["missing_h5"]:
        lines.append(f"  manifest ok but no HDF5    : {tally['missing_h5']}")
    if tally["missing_qc_image"]:
        lines.append(f"  manifest ok but no QC jpg  : {tally['missing_qc_image']}")
    lines += [
        f"eligible for review          : {len(eligible)}",
        "",
        "EBV balance of the reviewable set",
        "---------------------------------",
    ]
    for status, count in sorted(Counter(r["ebv_status"] for r in eligible).items()):
        lines.append(f"  {status:<9}: {count}")

    if len(pen):
        lines += [
            "",
            "pen_fraction_removed (share of segmented tissue deleted as ink)",
            "---------------------------------------------------------------",
            f"  median  : {np.median(pen) * 100:.3f}%",
            f"  p90     : {np.percentile(pen, 90) * 100:.2f}%",
            f"  p99     : {np.percentile(pen, 99) * 100:.2f}%",
            f"  max     : {pen.max() * 100:.2f}%",
            f"  > 0%    : {(pen > 0).sum()} slides",
            f"  >= 1%   : {(pen >= 0.01).sum()} slides",
            f"  >= 10%  : {(pen >= 0.10).sum()} slides",
            "",
            "Pen burden by EBV status (mean fraction removed)",
            "-----------------------------------------------",
        ]
        for status in EBV_CALLS:
            subset = np.array([r["pen_fraction_removed"] for r in eligible if r["ebv_status"] == status])
            if subset.size:
                lines.append(f"  {status:<9}: mean {subset.mean() * 100:5.2f}%   >=10%: {(subset >= 0.10).sum()}/{subset.size}")

        # Pen use is a property of the submitting site, so a site-skewed artifact
        # is a candidate confounder for any classifier trained on these slides.
        site_pen: dict[str, list[float]] = {}
        for record in eligible:
            site = record["slide_id"][5:7]
            site_pen.setdefault(site, []).append(record["pen_fraction_removed"])
        ranked = sorted(site_pen.items(), key=lambda kv: -float(np.mean(kv[1])))
        lines += ["", "Top tissue-source sites by mean pen fraction", "-------------------------------------------"]
        for site, values in ranked[:8]:
            arr = np.array(values)
            lines.append(f"  TCGA-{site}: n={arr.size:<4} mean {arr.mean() * 100:5.2f}%  >=10%: {(arr >= 0.10).sum()}")

    leaked = [r for r in eligible if isinstance(r.get("frac_patches_on_violet_ink"), float)]
    if leaked:
        # The violet band overlaps dark hematoxylin, so on a slide with no
        # confirmed ink a high score is more likely heavy staining than a miss.
        # Split on whether the filter already proved ink is present.
        inked = [r for r in leaked if r["pen_fraction_removed"] >= 0.01]
        clean = [r for r in leaked if r["pen_fraction_removed"] < 0.01]
        violet = np.array([r["frac_patches_on_violet_ink"] for r in leaked])
        dark = np.array([r["frac_patches_on_dark_ink"] for r in leaked])
        lines += [
            "",
            "Ink the filter MISSED (retained patch centres landing on untinted ink)",
            "----------------------------------------------------------------------",
            f"  violet band H{args.violet_hue_lo:.0f}-{args.violet_hue_hi:.0f} S>={args.violet_sat:.0f}, "
            f"black band V<{args.dark_value:.0f}",
            f"  slides with >1% of patches on violet ink : {(violet > 0.01).sum()}",
            f"  slides with >2% of patches on violet ink : {(violet > 0.02).sum()}",
            f"  slides with >1% of patches on black ink  : {(dark > 0.01).sum()}",
            f"  worst slide                              : {violet.max() * 100:.2f}% of patches",
            "",
            "  CAVEAT: this band overlaps dark hematoxylin. Read it as a genuine miss",
            "  only where the filter already removed ink on the same slide; on an",
            "  ink-free slide a high score usually means heavy staining. Split:",
        ]
        for name, group in (("confirmed ink (removed >=1%)", inked), ("no ink detected (removed <1%)", clean)):
            if not group:
                continue
            arr = np.array([r["frac_patches_on_violet_ink"] for r in group])
            lines.append(
                f"    {name:<30} n={arr.size:<4} median {np.median(arr) * 100:5.2f}%  "
                f">2%: {(arr > 0.02).sum()}  max {arr.max() * 100:.2f}%"
            )
        lines.append("")
        lines.append("  worst 12 among slides with confirmed ink:")
        for record in sorted(inked, key=lambda r: -r["frac_patches_on_violet_ink"])[:12]:
            lines.append(
                f"    {record['frac_patches_on_violet_ink'] * 100:5.2f}% missed  "
                f"{record['ebv_status']:<8} removed={record['pen_fraction_removed'] * 100:5.1f}%  "
                f"{short_name(record['slide_id'])}"
            )

    lines += [
        "",
        "Outputs",
        "-------",
        f"  pen_review.csv     : one row per metadata slide with an EBV call",
        f"  thumbs/            : {n_thumbs} captioned QC thumbnails, filename-sorted worst pen first",
        f"  contact_sheets/    : {n_sheets} pages covering all {len(eligible)} eligible slides",
        "",
        "Reading the images: green = retained tissue contour, red = excluded hole,",
        "magenta tint = ink the pen filter removed. Ink still showing untinted is ink",
        "the filter missed; magenta over plain tissue is the filter over-reaching.",
    ]
    text = "\n".join(lines)
    (out_dir / "coverage_report.txt").write_text(text + "\n")
    return text


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, tally = build_records(args)
    choose_subsample(records, args)

    fields = [
        "pen_rank", "slide_id", "cohort", "patient_id", "ebv_status", "label", "status",
        "patches", "pen_fraction_removed", "frac_patches_on_violet_ink", "frac_patches_on_dark_ink",
        "pen_filter", "level0_width", "level0_height",
        "eligible", "selected", "qc_image", "message",
    ]
    with (args.output_dir / "pen_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    n_thumbs = write_thumbs(records, args.output_dir, args)
    n_sheets = write_contact_sheets(records, args.output_dir, args) if args.contact_sheets else 0
    print(write_report(records, tally, args.output_dir, args, n_thumbs, n_sheets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
