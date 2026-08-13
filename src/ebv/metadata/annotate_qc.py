#!/usr/bin/env python
"""Mark pen-contaminated slides as excluded in `metadata/cohort_labels.csv`.

Adds two columns, in place and idempotently:

- `qc_exclude`        1 if the slide must be dropped from training/evaluation
- `qc_exclude_reason` free text, empty when `qc_exclude` is 0

The exclusion list is derived from `03_qc_pen_review/pen_review.csv`, which ranks
every reviewable slide by `pen_fraction_removed` (descending, rank 0 = worst).
Selecting by rank reproduces exactly what was inspected in
`03_qc_pen_review/thumbs/`, where the filenames carry the same `rNNN` prefix.

Re-running recomputes the flags from scratch rather than accumulating, so
changing `--max-rank` and re-running is safe.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXCLUDE_COLUMNS = ("qc_exclude", "qc_exclude_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=REPO / "metadata/cohort_labels.csv")
    parser.add_argument("--review", type=Path, default=REPO / "03_qc_pen_review/pen_review.csv")
    parser.add_argument(
        "--max-rank", type=int, default=34,
        help="Exclude pen_rank 0..MAX_RANK inclusive (default 34, i.e. r000-r034).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the change without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    excluded: dict[str, str] = {}
    for row in csv.DictReader(args.review.open()):
        if row["eligible"] != "1" or row["pen_rank"] == "":
            continue
        rank = int(row["pen_rank"])
        if rank <= args.max_rank:
            pen = float(row["pen_fraction_removed"]) * 100
            excluded[row["slide_id"]] = f"pen_review r{rank:03d}: pen_fraction_removed={pen:.2f}%"

    with args.labels.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = [f for f in reader.fieldnames if f not in EXCLUDE_COLUMNS] + list(EXCLUDE_COLUMNS)
        rows = list(reader)

    hits = 0
    for row in rows:
        reason = excluded.get(row["slide_id"], "")
        row["qc_exclude"] = "1" if reason else "0"
        row["qc_exclude_reason"] = reason
        hits += bool(reason)

    print(f"{args.labels}: {len(rows)} rows, {hits} marked qc_exclude=1 "
          f"(pen_rank 0-{args.max_rank})")
    unmatched = set(excluded) - {r["slide_id"] for r in rows}
    if unmatched:
        print(f"WARNING: {len(unmatched)} reviewed slide_ids are absent from the label table: "
              f"{sorted(unmatched)[:3]}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    tmp = args.labels.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.labels)
    print(f"wrote {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
