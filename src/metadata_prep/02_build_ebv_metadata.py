#!/usr/bin/env python3
"""Build slide-level EBV labels from the cohort XLSX and preprocessed HDF5 files.

This uses only the Python standard library so it works in the preprocessing
environment without pandas or openpyxl.
"""

from __future__ import annotations

import argparse
import csv
import posixpath
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "00_new_HE" / "Chile set_Training_n_132.xlsx"
DEFAULT_PATCH_DIR = ROOT / "01_preprocessed" / "patches"
DEFAULT_OUTPUT = ROOT / "metadata" / "ebv_status.csv"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match preprocessed slides to EBER-ISH labels in an XLSX workbook."
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any preprocessed slide lacks a workbook match (default: exclude it).",
    )
    return parser.parse_args()


def cell_column(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference.upper())
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    number = 0
    for character in match.group():
        number = number * 26 + ord(character) - ord("A") + 1
    return number - 1


def first_worksheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("workbook contains no worksheets")
    relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            if target.startswith("xl/"):
                return posixpath.normpath(target)
            return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"worksheet relationship {relationship_id!r} was not found")


def read_first_xlsx_sheet(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{MAIN_NS}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                )

        sheet_root = ET.fromstring(archive.read(first_worksheet_path(archive)))
        values: list[dict[int, str]] = []
        for row in sheet_root.findall(f".//{{{MAIN_NS}}}row"):
            parsed_row: dict[int, str] = {}
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                column = cell_column(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
                    )
                else:
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    value = value_node.text if value_node is not None and value_node.text else ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                parsed_row[column] = value.strip()
            values.append(parsed_row)

    if not values:
        raise ValueError("worksheet is empty")
    headers = {column: value.strip().lower() for column, value in values[0].items()}
    if len(set(headers.values())) != len(headers):
        raise ValueError("worksheet contains duplicate headers")
    return [
        {header: row.get(column, "") for column, header in headers.items()}
        for row in values[1:]
        if any(row.values())
    ]


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def patient_id(value: str) -> str:
    return normalized(value).split("-", 1)[0]


def sample_id_from_slide(slide_id: str) -> str:
    # Scanner filenames append a six-digit scan time to the cohort scanner ID.
    match = re.fullmatch(r"(.+)_([0-9]{6})", slide_id)
    if match is None:
        raise ValueError(f"slide ID does not end in an underscore and six digits: {slide_id!r}")
    return match.group(1)


def build_metadata(
    workbook: Path, patch_dir: Path
) -> tuple[list[dict[str, object]], list[str]]:
    source_rows = read_first_xlsx_sheet(workbook)
    required = {"idscanner", "id", "eber_ish"}
    available = set(source_rows[0]) if source_rows else set()
    if not required <= available:
        missing = ", ".join(sorted(required - available))
        raise ValueError(f"workbook is missing required column(s): {missing}")

    by_scanner: dict[str, dict[str, str]] = {}
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        scanner_key = normalized(row["idscanner"])
        patient_key = normalized(row["id"])
        if not scanner_key or not patient_key:
            raise ValueError("workbook has a blank idscanner or id")
        if scanner_key in by_scanner:
            raise ValueError(f"duplicate workbook idscanner: {row['idscanner']!r}")
        by_scanner[scanner_key] = row
        by_patient.setdefault(patient_key, []).append(row)

    patch_files = sorted(patch_dir.glob("*.h5"), key=lambda path: path.stem)
    if not patch_files:
        raise ValueError(f"no .h5 preprocessing outputs found in {patch_dir}")

    output: list[dict[str, object]] = []
    unmatched: list[str] = []
    used_source_ids: set[str] = set()
    for patch_path in patch_files:
        slide_id = patch_path.stem
        scanner_id = sample_id_from_slide(slide_id)
        row = by_scanner.get(normalized(scanner_id))
        method = "idscanner_exact"
        if row is None:
            candidates = by_patient.get(patient_id(scanner_id), [])
            if len(candidates) == 1:
                row = candidates[0]
                method = "patient_id_unique"
            else:
                unmatched.append(slide_id)
                continue

        status = row["eber_ish"].strip().lower()
        labels = {"negative": 0, "positive": 1}
        if status not in labels:
            raise ValueError(
                f"unsupported EBER-ISH value {row['eber_ish']!r} for {row['idscanner']!r}"
            )
        source_key = normalized(row["idscanner"])
        if source_key in used_source_ids:
            raise ValueError(
                f"workbook record {row['idscanner']!r} matched more than one preprocessed slide"
            )
        used_source_ids.add(source_key)
        output.append(
            {
                "slide_id": slide_id,
                "source_idscanner": row["idscanner"],
                "ebv_status": status.capitalize(),
                "label": labels[status],
                "match_method": method,
            }
        )

    if len(output) + len(unmatched) != len(patch_files):
        raise AssertionError("preprocessing outputs were not fully accounted for")
    return output, unmatched


def main() -> int:
    args = parse_args()
    try:
        rows, unmatched = build_metadata(args.workbook.resolve(), args.patch_dir.resolve())
        if args.strict and unmatched:
            raise ValueError("unmatched preprocessed slide(s): " + ", ".join(unmatched))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "slide_id",
                    "source_idscanner",
                    "ebv_status",
                    "label",
                    "match_method",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    positives = sum(int(row["label"]) for row in rows)
    fallbacks = sum(row["match_method"] != "idscanner_exact" for row in rows)
    print(f"Wrote {len(rows)} slides to {args.output}")
    print(f"EBV status: {positives} Positive, {len(rows) - positives} Negative")
    print(f"Matches: {len(rows) - fallbacks} exact idscanner, {fallbacks} unique patient-ID fallback")
    if unmatched:
        print(f"Excluded {len(unmatched)} preprocessed slides with no workbook record:")
        print("  " + ", ".join(unmatched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
