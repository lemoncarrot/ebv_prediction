#!/usr/bin/env python3
"""Download TCGA whole-slide images from the GDC.

TCGA slide images are open access: no token, no dbGaP approval, works entirely
from the CLI. Three ways to choose what to fetch:

    --project TCGA-STAD          every open-access diagnostic slide in a project
    <slide names...>             named slides (with or without .svs/.h5 suffix)
    --auto N                     N slides drawn from a precomputed feature dir

Downloads run concurrently, skip files already present, verify md5, and record
per-slide status in a manifest so an interrupted run can simply be re-run.

Standard library only, so it runs in the preprocessing environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = ROOT / "TCGA_STAD_precomputed_features" / "TCGA"
DEFAULT_OUTPUT = ROOT / "00_tcga_WSI_raw"
GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
CHUNK = 8 << 20
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "slides", nargs="*", help="Slide names, with or without .svs/.h5 suffix."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Fetch every open-access slide in a GDC project, e.g. TCGA-STAD.",
    )
    parser.add_argument(
        "--strategy",
        default="Diagnostic Slide",
        help="Experimental strategy for --project. 'Diagnostic Slide' is FFPE "
        "(what you want for H&E morphology); 'Tissue Slide' is frozen section.",
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--auto",
        type=int,
        default=None,
        help="Pick the N smallest slides from --feature-dir, balanced across "
        "native magnifications.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent downloads (GDC tolerates ~4-8)."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Attempts per slide before giving up."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-check md5 of files that already exist (slow; size is checked regardless).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.slides and args.auto is None and args.project is None:
        parser.error("give slide names, or --project, or --auto")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def say(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def slide_name(value: str) -> str:
    for suffix in (".svs", ".h5"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def gdc_query(filters: dict, fields: str, size: int) -> list[dict]:
    payload = {"filters": filters, "fields": fields, "format": "JSON", "size": str(size)}
    request = urllib.request.Request(
        GDC_FILES, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["data"]["hits"]


def resolve_project(project: str, strategy: str) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": [strategy]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    hits = gdc_query(filters, "file_id,file_name,file_size,md5sum,access", 20000)
    return sorted(hits, key=lambda hit: hit["file_name"])


def resolve_names(names: list[str]) -> dict[str, dict]:
    filters = {
        "op": "in",
        "content": {"field": "file_name", "value": [f"{n}.svs" for n in names]},
    }
    hits = gdc_query(filters, "file_id,file_name,file_size,md5sum,access", len(names) * 4)
    return {slide_name(hit["file_name"]): hit for hit in hits}


def auto_select(feature_dir: Path, count: int) -> list[str]:
    """Pick the smallest precomputed slides, alternating native magnification."""
    import h5py

    by_magnification: dict[int, list[tuple[int, str]]] = {}
    for path in sorted(feature_dir.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            attrs = handle["coords"].attrs
            if "level0_magnification" not in attrs:
                continue
            by_magnification.setdefault(int(attrs["level0_magnification"]), []).append(
                (int(handle["features"].shape[1]), str(attrs["name"]))
            )
    if not by_magnification:
        raise ValueError(f"no usable precomputed features under {feature_dir}")
    for entries in by_magnification.values():
        entries.sort()
    order = sorted(by_magnification)
    picked: list[str] = []
    index = 0
    while len(picked) < count and any(by_magnification[m] for m in order):
        magnification = order[index % len(order)]
        if by_magnification[magnification]:
            patches, name = by_magnification[magnification].pop(0)
            say(f"  selected {name}  ({patches} patches, {magnification}x native)")
            picked.append(name)
        index += 1
    return picked


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(hit: dict, output_dir: Path, args: argparse.Namespace) -> dict:
    name = slide_name(hit["file_name"])
    destination = output_dir / hit["file_name"]
    size = int(hit["file_size"])
    row = {"slide": name, "file_id": hit["file_id"], "bytes": size}

    if destination.exists() and not args.overwrite:
        if destination.stat().st_size != size:
            say(f"  {name}: size mismatch on disk, re-downloading")
        elif args.verify and md5_of(destination) != hit["md5sum"]:
            say(f"  {name}: md5 mismatch on disk, re-downloading")
        else:
            return {**row, "status": "skipped", "message": "exists"}

    tmp_path = destination.with_suffix(".svs.part")
    for attempt in range(1, args.retries + 1):
        digest = hashlib.md5()
        written = 0
        try:
            with urllib.request.urlopen(f"{GDC_DATA}/{hit['file_id']}", timeout=300) as response:
                with tmp_path.open("wb") as stream:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        stream.write(block)
                        digest.update(block)
                        written += len(block)
            if written != size:
                raise ValueError(f"size mismatch: got {written}, expected {size}")
            if digest.hexdigest() != hit["md5sum"]:
                raise ValueError("md5 mismatch")
            tmp_path.replace(destination)
            return {**row, "status": "ok", "message": ""}
        except (OSError, ValueError) as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt == args.retries:
                return {**row, "status": "failed", "message": f"{type(exc).__name__}: {exc}"}
            say(f"  {name}: attempt {attempt} failed ({exc}); retrying")
    return {**row, "status": "failed", "message": "exhausted retries"}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    try:
        if args.project:
            say(f"Querying GDC for {args.project} / {args.strategy} ...")
            hits = resolve_project(args.project, args.strategy)
            if not hits:
                print("ERROR: query returned no files", file=sys.stderr)
                return 2
        else:
            if args.auto is not None:
                feature_dir = args.feature_dir.resolve()
                if not feature_dir.is_dir():
                    print(f"ERROR: no feature directory: {feature_dir}", file=sys.stderr)
                    return 2
                say(f"Selecting {args.auto} slide(s) from {feature_dir}:")
                names = auto_select(feature_dir, args.auto)
            else:
                names = [slide_name(value) for value in args.slides]
            found = resolve_names(names)
            missing = [name for name in names if name not in found]
            if missing:
                print(f"ERROR: GDC has no file named: {', '.join(missing)}", file=sys.stderr)
                return 2
            hits = [found[name] for name in names]
    except OSError as exc:
        print(f"ERROR: GDC API request failed: {exc}", file=sys.stderr)
        return 2

    controlled = [h["file_name"] for h in hits if h.get("access") != "open"]
    if controlled:
        print(f"ERROR: controlled-access file(s): {', '.join(controlled[:5])}", file=sys.stderr)
        return 2

    total = sum(int(hit["file_size"]) for hit in hits)
    present = sum(
        1
        for hit in hits
        if (output_dir / hit["file_name"]).exists()
        and (output_dir / hit["file_name"]).stat().st_size == int(hit["file_size"])
    )
    say(f"{len(hits)} slide(s), {total / 1e12:.2f} TB total; {present} already on disk.")
    if args.dry_run:
        for hit in hits[:20]:
            say(f"  {hit['file_id']}  {int(hit['file_size']) / 1e6:9.1f} MB  {hit['file_name']}")
        if len(hits) > 20:
            say(f"  ... and {len(hits) - 20} more")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, hit, output_dir, args): hit for hit in hits}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            done += 1
            say(f"[{done}/{len(hits)}] {row['slide'][:44]:<44} {row['status']}")

    manifest = output_dir / "manifest.csv"
    rows.sort(key=lambda item: item["slide"])
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["slide", "file_id", "bytes", "status", "message"]
        )
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] == "failed" for row in rows)
    ok = sum(row["status"] == "ok" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    say(f"\nok {ok}, skipped {skipped}, failed {failures}. Manifest: {manifest}")
    if failures:
        say("Re-run the same command to retry the failures; completed files are skipped.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
