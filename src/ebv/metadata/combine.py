#!/usr/bin/env python3
"""Combine Chile and TCGA-STAD slide labels into one training table.

The two cohorts are labelled by different assays, and the TCGA table is messy in
three specific ways this script handles explicitly:

1. Its `EBV` column is ~52% NA, while `Subtype` (STAD_EBV) is complete. Both are
   carried through; `--tcga-label-source` picks which one becomes `label`.
2. 59 samples appear on more than one row. The duplicates never disagree on
   EBV/Subtype, so they are collapsed, but any future disagreement is an error
   rather than a silent first-wins.
3. Slide barcodes carry a vial letter (`-01Z`) that the metadata `Sample` column
   does not (`-01`), so the join needs the letter stripped.

Patients can contribute several slides, so `patient_id` is emitted for grouped
train/test splitting -- splitting on slide would leak.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHILE = ROOT / "metadata" / "ebv_status.csv"
DEFAULT_TCGA_TABLE = ROOT / "metadata" / "STAD_merged_table.csv"
DEFAULT_TCGA_SLIDES = ROOT / "00_tcga_WSI_raw"
DEFAULT_CHILE_SLIDES = ROOT / "00_new_HE"
DEFAULT_OUTPUT = ROOT / "metadata" / "cohort_labels.csv"

FIELDS = [
    "slide_id",
    "cohort",
    "patient_id",
    "sample_id",
    "ebv_status",
    "label",
    "label_source",
    "tcga_subtype",
    "tcga_ebv_column",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chile-labels", type=Path, default=DEFAULT_CHILE)
    parser.add_argument("--chile-slide-dir", type=Path, default=DEFAULT_CHILE_SLIDES)
    parser.add_argument("--tcga-table", type=Path, default=DEFAULT_TCGA_TABLE)
    parser.add_argument("--tcga-slide-dir", type=Path, default=DEFAULT_TCGA_SLIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tcga-label-source",
        choices=("subtype", "ebv_column"),
        default="subtype",
        help="subtype: STAD_EBV vs rest (complete coverage). ebv_column: the "
        "explicit EBV call, closer to Chile's EBER-ISH but ~52%% missing.",
    )
    parser.add_argument(
        "--drop-unlabelled",
        action="store_true",
        help="Omit slides with no usable label instead of writing them with a blank label.",
    )
    return parser.parse_args()


def blank(value: str) -> bool:
    return value.strip() in {"", "NA", "NaN", "nan", "None", "#N/A"}


def sample_from_slide(slide_id: str) -> str:
    """`TCGA-3M-AB46-01Z-00-DX1.<uuid>` -> `TCGA-3M-AB46-01` (vial letter dropped)."""
    fields = slide_id.split(".")[0].split("-")
    if len(fields) < 4:
        raise ValueError(f"unrecognised TCGA slide barcode: {slide_id!r}")
    return "-".join(fields[:3] + [re.sub(r"[^0-9]", "", fields[3])])


def patient_from_slide(slide_id: str) -> str:
    return "-".join(slide_id.split(".")[0].split("-")[:3])


def read_chile(labels_path: Path, slide_dir: Path) -> list[dict]:
    if not labels_path.exists():
        print(f"WARNING: no Chile labels at {labels_path}; skipping cohort", file=sys.stderr)
        return []
    present = {path.stem for path in slide_dir.glob("*.svs")} if slide_dir.is_dir() else None
    out: list[dict] = []
    for row in csv.DictReader(labels_path.open()):
        slide_id = row["slide_id"]
        note = ""
        if present is not None and slide_id not in present:
            note = "slide file not found"
        out.append(
            {
                "slide_id": slide_id,
                "cohort": "chile",
                # Scanner IDs look like `140-A4`; the patient is the part before the dash.
                "patient_id": row.get("source_idscanner", "").split("-")[0].strip(),
                "sample_id": row.get("source_idscanner", ""),
                "ebv_status": row["ebv_status"],
                "label": row["label"],
                "label_source": "eber_ish",
                "tcga_subtype": "",
                "tcga_ebv_column": "",
                "note": note,
            }
        )
    return out


def read_tcga(table_path: Path, slide_dir: Path, source: str) -> tuple[list[dict], dict]:
    if not table_path.exists():
        print(f"WARNING: no TCGA table at {table_path}; skipping cohort", file=sys.stderr)
        return [], {}
    rows = list(csv.DictReader(table_path.open()))
    required = {"Sample", "Patient.ID", "Subtype", "EBV", "SampleCode"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"TCGA table missing column(s): {', '.join(sorted(missing))}")

    # Collapse duplicate rows per sample, but refuse to guess if they disagree.
    by_sample: dict[str, dict] = {}
    clashes: list[str] = []
    for row in rows:
        if row["SampleCode"].strip() != "1":
            continue  # keep primary solid tumour only; 11 is matched normal
        key = row["Sample"]
        keep = (row["Subtype"], row["EBV"])
        if key in by_sample:
            if (by_sample[key]["Subtype"], by_sample[key]["EBV"]) != keep:
                clashes.append(key)
            continue
        by_sample[key] = row
    if clashes:
        raise ValueError(
            f"{len(clashes)} sample(s) have conflicting EBV/Subtype across duplicate "
            f"rows, e.g. {clashes[:3]}"
        )

    slides = sorted(path.stem for path in slide_dir.glob("*.svs")) if slide_dir.is_dir() else []
    out: list[dict] = []
    stats = Counter()
    for slide_id in slides:
        sample = sample_from_slide(slide_id)
        row = by_sample.get(sample)
        if row is None:
            stats["unmapped"] += 1
            out.append(
                {
                    "slide_id": slide_id,
                    "cohort": "tcga",
                    "patient_id": patient_from_slide(slide_id),
                    "sample_id": sample,
                    "ebv_status": "",
                    "label": "",
                    "label_source": "",
                    "tcga_subtype": "",
                    "tcga_ebv_column": "",
                    "note": "no metadata row for this sample",
                }
            )
            continue

        subtype = row["Subtype"].strip()
        ebv_col = "" if blank(row["EBV"]) else row["EBV"].strip()
        note = ""
        if source == "subtype":
            status = "Positive" if subtype == "STAD_EBV" else "Negative"
            label_source = "tcga_subtype"
        else:
            status = ebv_col
            label_source = "tcga_ebv_column"
        # Flag the disagreement rather than hiding it behind whichever source won.
        if ebv_col and ((ebv_col == "Positive") != (subtype == "STAD_EBV")):
            note = f"EBV column ({ebv_col}) disagrees with Subtype ({subtype})"
            stats["conflict"] += 1

        if not status:
            stats["unlabelled"] += 1
            note = note or "EBV column is NA"
        else:
            stats["positive" if status == "Positive" else "negative"] += 1
        out.append(
            {
                "slide_id": slide_id,
                "cohort": "tcga",
                "patient_id": row["Patient.ID"],
                "sample_id": sample,
                "ebv_status": status,
                "label": {"Positive": 1, "Negative": 0}.get(status, ""),
                "label_source": label_source if status else "",
                "tcga_subtype": subtype,
                "tcga_ebv_column": ebv_col,
                "note": note,
            }
        )
    return out, stats


def summarise(rows: list[dict]) -> None:
    by_cohort = defaultdict(Counter)
    patients = defaultdict(set)
    for row in rows:
        by_cohort[row["cohort"]][row["ebv_status"] or "unlabelled"] += 1
        if row["ebv_status"]:
            patients[row["cohort"]].add(row["patient_id"])
    print(f"\n{'cohort':<8} {'slides':>7} {'pos':>6} {'neg':>6} {'unlab':>7} {'patients':>9}")
    for cohort in sorted(by_cohort):
        counts = by_cohort[cohort]
        total = sum(counts.values())
        print(
            f"{cohort:<8} {total:7d} {counts['Positive']:6d} {counts['Negative']:6d} "
            f"{counts['unlabelled']:7d} {len(patients[cohort]):9d}"
        )
    labelled = [r for r in rows if r["ebv_status"]]
    pos = sum(1 for r in labelled if r["ebv_status"] == "Positive")
    print(
        f"{'TOTAL':<8} {len(rows):7d} {pos:6d} {len(labelled) - pos:6d} "
        f"{len(rows) - len(labelled):7d} "
        f"{len({(r['cohort'], r['patient_id']) for r in labelled}):9d}"
    )
    if labelled:
        print(f"\npositive rate: {100 * pos / len(labelled):.1f}% of labelled slides")


def main() -> int:
    args = parse_args()
    try:
        chile = read_chile(args.chile_labels.resolve(), args.chile_slide_dir.resolve())
        tcga, stats = read_tcga(
            args.tcga_table.resolve(), args.tcga_slide_dir.resolve(), args.tcga_label_source
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = chile + tcga
    if not rows:
        print("ERROR: no slides found in either cohort", file=sys.stderr)
        return 2

    duplicated = [k for k, v in Counter(r["slide_id"] for r in rows).items() if v > 1]
    if duplicated:
        print(f"ERROR: duplicate slide_id across cohorts: {duplicated[:5]}", file=sys.stderr)
        return 2

    if args.drop_unlabelled:
        before = len(rows)
        rows = [r for r in rows if r["ebv_status"]]
        print(f"dropped {before - len(rows)} unlabelled slide(s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} slides to {args.output}")
    print(f"TCGA label source: {args.tcga_label_source}")
    if stats.get("unmapped"):
        print(f"  {stats['unmapped']} TCGA slide(s) have no metadata row")
    if stats.get("conflict"):
        print(f"  {stats['conflict']} TCGA slide(s) where EBV column and Subtype disagree")
    if stats.get("unlabelled"):
        print(f"  {stats['unlabelled']} TCGA slide(s) unlabelled under this source")
    summarise(rows)

    multi = [k for k, v in Counter(
        (r["cohort"], r["patient_id"]) for r in rows if r["ebv_status"]
    ).items() if v > 1]
    if multi:
        print(f"\n{len(multi)} patient(s) contribute >1 labelled slide -- group by "
              f"patient_id when splitting, or the split will leak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
