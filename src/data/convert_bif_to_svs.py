#!/usr/bin/env python3
"""Convert Roche/Ventana .bif whole-slide images to Aperio-style .svs.

    python convert_bif_to_svs.py --input ../../data/robertoherrera/IC-18-10201.bif \
                                 --out-dir ../../data/robertoherrera_svs
    python convert_bif_to_svs.py --summarize --out-dir ../../data/robertoherrera_svs

Per slide, writes <out-dir>/<stem>.svs plus <stem>.json recording the source's
scan metadata (mpp, objective power, levels, scanned bounds) and a validation of
the output. --summarize merges every .json into conversion_report.csv.

The .svs is a tiled, JPEG-compressed BigTIFF pyramid whose first ImageDescription
follows the Aperio convention ("Aperio ...|AppMag = ..|MPP = .."), which is what
makes OpenSlide (and QuPath) read it as format "aperio" with the source's
resolution. Level 0 is read from the .bif at full resolution in 256 px tile rows;
lower levels (downsample 4, 16, 64) come from the .bif's own pyramid when it has
that exact downsample, else are resized from the nearest finer level. Areas
outside the scanned region (transparent in OpenSlide) become white.

Validation: reopen the .svs with OpenSlide, require format "aperio", identical
level-0 size and mpp, and a small mean pixel difference to the source on
sampled tissue-sized regions (JPEG re-encoding is lossy, so not zero).
Skips slides whose .svs and .json already exist unless --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import openslide
import tifffile
from PIL import Image

TILE = 256
JPEG_QUALITY = 90
LEVEL_DOWNSAMPLES = (4, 16, 64)
MIN_LEVEL_DIM = 512          # stop adding levels below this size
READ_CHUNK = 8192            # columns per read_region call within a tile row


class BifReader:
    """Minimal OpenSlide-like reader for Ventana BIF via tifffile.

    Used when OpenSlide cannot open the file: VENTANA DP 200/600 scans record
    tile joints with Direction="LEFT", which no released OpenSlide accepts
    (openslide issue #759 / PR #760). Reading the TIFF tiles directly is only
    correct when every joint has zero overlap (tiles abut exactly), so the
    constructor refuses files with any non-zero OverlapX/OverlapY.
    """

    def __init__(self, path: Path, cache_tiles: int = 96):
        self._tif = tifffile.TiffFile(path)
        self._lock = threading.Lock()
        levels = []
        for page in self._tif.pages:
            m = re.match(r"level=(\d+)", page.description or "")
            if m and page.is_tiled:
                levels.append((int(m.group(1)), page))
        if not levels:
            raise ValueError(f"{path.name}: no 'level=N' tiled pages; not a Ventana BIF pyramid")
        self._pages = [p for _, p in sorted(levels, key=lambda t: t[0])]
        xmp = ""
        for page in self._tif.pages:
            if "XMP" in page.tags:
                value = page.tags["XMP"].value
                xmp += value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
        self.xmp = xmp
        overlaps = re.findall(r'OverlapX="(-?\d+)"\s+OverlapY="(-?\d+)"', xmp)
        nonzero = sum(1 for ox, oy in overlaps if int(ox) or int(oy))
        if nonzero:
            raise ValueError(f"{path.name}: {nonzero}/{len(overlaps)} tile joints overlap; "
                             "direct TIFF read would mis-stitch. Needs an OpenSlide with "
                             "LEFT-joint support.")
        self.n_joints = len(overlaps)
        iscan = re.search(r"<iScan[^>]*>", xmp)          # scanner settings (ScanRes, Magnification, ...)
        self.scan = dict(re.findall(r'(\w+)="([^"]*)"', iscan.group(0))) if iscan else {}
        base = self._pages[0]
        self.dimensions = (base.imagewidth, base.imagelength)
        self.level_dimensions = [(p.imagewidth, p.imagelength) for p in self._pages]
        self.level_downsamples = [base.imagewidth / p.imagewidth for p in self._pages]
        self.level_count = len(self._pages)
        self._cache: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()
        self._cache_size = cache_tiles

    def get_best_level_for_downsample(self, downsample: float) -> int:
        best = 0
        for i, d in enumerate(self.level_downsamples):
            if d <= downsample:
                best = i
        return best

    def _tile(self, level: int, index: int) -> np.ndarray:
        key = (level, index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        page = self._pages[level]
        with self._lock:
            fh = self._tif.filehandle
            fh.seek(page.dataoffsets[index])
            data = fh.read(page.databytecounts[index])
        segment, _, _ = page.decode(data, index, jpegtables=page.jpegtables)
        tile = np.asarray(segment).reshape(page.tilelength, page.tilewidth, -1)[..., :3]
        self._cache[key] = tile
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return tile

    def read_region(self, location: tuple[int, int], level: int,
                    size: tuple[int, int]) -> Image.Image:
        """Same contract as OpenSlide: location in level-0 pixels, RGBA out,
        transparent outside the image."""
        page = self._pages[level]
        ds = self.level_downsamples[level]
        x0, y0 = int(location[0] / ds), int(location[1] / ds)
        w, h = size
        out = np.zeros((h, w, 4), dtype=np.uint8)
        tw, tl = page.tilewidth, page.tilelength
        across = -(-page.imagewidth // tw)
        for ty in range(max(0, y0 // tl), min(-(-page.imagelength // tl), (y0 + h - 1) // tl + 1)):
            for tx in range(max(0, x0 // tw), min(across, (x0 + w - 1) // tw + 1)):
                tile = self._tile(level, ty * across + tx)
                # overlap of this tile with the requested window, clipped to the image
                gx0, gy0 = max(x0, tx * tw), max(y0, ty * tl)
                gx1 = min(x0 + w, (tx + 1) * tw, page.imagewidth)
                gy1 = min(y0 + h, (ty + 1) * tl, page.imagelength)
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                out[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0, :3] = \
                    tile[gy0 - ty * tl:gy1 - ty * tl, gx0 - tx * tw:gx1 - tx * tw]
                out[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0, 3] = 255
        return Image.fromarray(out, "RGBA")

    def close(self) -> None:
        self._tif.close()


def open_slide(path: Path):
    """OpenSlide when it can read the file, else the tifffile BIF reader."""
    try:
        return openslide.OpenSlide(str(path)), "openslide"
    except openslide.OpenSlideError as error:
        if path.suffix.lower() != ".bif":
            raise
        print(f"OpenSlide cannot open {path.name} ({error}); using tifffile BIF reader")
        return BifReader(path), "tifffile-bif"


def scan_metadata(slide, path: Path, reader: str = "openslide") -> dict:
    """What the scanner recorded, before any conversion."""
    if isinstance(slide, BifReader):
        mpp = float(slide.scan["ScanRes"]) if slide.scan.get("ScanRes") else None
        objective = float(slide.scan["Magnification"]) if slide.scan.get("Magnification") else None
        return {
            "slide_id": path.stem, "source": str(path), "source_bytes": path.stat().st_size,
            "format": "ventana-bif", "reader": reader, "vendor": "ventana",
            "scanner": slide.scan.get("ScannerModel"), "mpp_x": mpp, "mpp_y": mpp,
            "objective_power": objective,
            "magnification_from_mpp": round(10 / mpp, 1) if mpp else None,
            "width": slide.dimensions[0], "height": slide.dimensions[1],
            "levels": slide.level_count,
            "level_downsamples": [round(d, 3) for d in slide.level_downsamples],
            "bounds": None, "tile_joints_checked": slide.n_joints,
            "ventana": slide.scan,
        }
    p = slide.properties
    def num(key):
        try:
            return float(p[key])
        except (KeyError, ValueError):
            return None
    mpp_x, mpp_y = num(openslide.PROPERTY_NAME_MPP_X), num(openslide.PROPERTY_NAME_MPP_Y)
    objective = num(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    return {
        "slide_id": path.stem, "source": str(path), "source_bytes": path.stat().st_size,
        "format": openslide.OpenSlide.detect_format(str(path)),
        "slide_id": path.stem, "source": str(path), "source_bytes": path.stat().st_size,
        "reader": reader,
        "mpp_x": mpp_x, "mpp_y": mpp_y, "objective_power": objective,
        # what the slide scale corresponds to, independent of the recorded objective
        "magnification_from_mpp": round(10 / mpp_x, 1) if mpp_x else None,
        "width": slide.dimensions[0], "height": slide.dimensions[1],
        "levels": slide.level_count,
        "level_downsamples": [round(d, 3) for d in slide.level_downsamples],
        "bounds": [p.get(f"openslide.bounds-{k}") for k in ("x", "y", "width", "height")],
        "ventana": {k: v for k, v in p.items() if k.startswith("ventana.")},
    }


def rgb(region: Image.Image) -> np.ndarray:
    """RGBA from OpenSlide -> RGB on white (transparent = outside the scan)."""
    background = Image.new("RGB", region.size, (255, 255, 255))
    background.paste(region, mask=region.split()[3])
    return np.asarray(background)


def level_tiles(slide: openslide.OpenSlide, level: int, size: tuple[int, int],
                scale: float):
    """Yield TILE x TILE RGB tiles of one output level, row-major, edge tiles
    padded white. `size` is the output level's (w, h); the source is OpenSlide
    `level`, resized by `scale` (= 1 when the source level matches exactly)."""
    w, h = size
    src_ds = slide.level_downsamples[level]
    for y0 in range(0, h, TILE):
        rows = min(TILE, h - y0)
        strip = np.full((TILE, w, 3), 255, dtype=np.uint8)
        for x0 in range(0, w, READ_CHUNK):
            cols = min(READ_CHUNK, w - x0)
            # source-level region covering this output chunk
            sx, sy = int(round(x0 * scale)), int(round(y0 * scale))
            sw, sh = max(1, int(round(cols * scale))), max(1, int(round(rows * scale)))
            region = slide.read_region((int(sx * src_ds), int(sy * src_ds)), level, (sw, sh))
            block = rgb(region)
            if scale != 1:
                block = np.asarray(Image.fromarray(block).resize((cols, rows), Image.BOX))
            strip[:rows, x0:x0 + cols] = block[:rows, :cols]
        for x0 in range(0, w, TILE):
            tile = strip[:, x0:x0 + TILE]
            if tile.shape[1] < TILE:
                tile = np.pad(tile, ((0, 0), (0, TILE - tile.shape[1]), (0, 0)),
                              constant_values=255)
            yield tile


def aperio_description(w: int, h: int, mpp: float, mag: float | None, slide_id: str) -> str:
    mag_text = f"|AppMag = {mag:g}" if mag else ""
    return (f"Aperio Image Library v12.0.0 \r\n{w}x{h} [0,0 {w}x{h}] ({TILE}x{TILE}) "
            f"JPEG/RGB Q={JPEG_QUALITY}{mag_text}|MPP = {mpp:.6f}|Filename = {slide_id}"
            f"|Converted = from Ventana BIF by convert_bif_to_svs.py")


def convert(path: Path, out: Path) -> dict:
    slide, reader = open_slide(path)
    meta = scan_metadata(slide, path, reader)
    if not meta["mpp_x"]:
        raise SystemExit(f"ERROR: {path.name} has no mpp; refusing to write an SVS without scale")
    w, h = slide.dimensions
    mag = meta["objective_power"] or meta["magnification_from_mpp"]
    started = time.time()
    tmp = out.with_suffix(".svs.tmp")
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
        levels_written = [(1, w, h)]
        tw.write(level_tiles(slide, 0, (w, h), 1.0), shape=(h, w, 3), dtype=np.uint8,
                 tile=(TILE, TILE), photometric="rgb", compression="jpeg",
                 compressionargs={"level": JPEG_QUALITY}, subfiletype=0,
                 description=aperio_description(w, h, meta["mpp_x"], mag, path.stem),
                 resolution=(1e4 / meta["mpp_x"], 1e4 / meta["mpp_y"]), resolutionunit="CENTIMETER",
                 metadata=None)
        for ds in LEVEL_DOWNSAMPLES:
            lw, lh = w // ds, h // ds
            if min(lw, lh) < MIN_LEVEL_DIM:
                break
            src = slide.get_best_level_for_downsample(ds + 1e-6)
            scale = ds / slide.level_downsamples[src]
            tw.write(level_tiles(slide, src, (lw, lh), scale), shape=(lh, lw, 3), dtype=np.uint8,
                     tile=(TILE, TILE), photometric="rgb", compression="jpeg",
                     compressionargs={"level": JPEG_QUALITY}, subfiletype=0,
                     description=f"Aperio Image Library v12.0.0 \r\n{w}x{h} -> {lw}x{lh}",
                     metadata=None)
            levels_written.append((ds, lw, lh))
    tmp.rename(out)
    meta.update({"output": str(out), "output_bytes": out.stat().st_size,
                 "output_levels": [ds for ds, _, _ in levels_written],
                 "seconds": round(time.time() - started, 1)})
    meta.update(validate(slide, out, meta))
    slide.close()
    return meta


def validate(source: openslide.OpenSlide, out: Path, meta: dict, n: int = 8) -> dict:
    """Reopen the .svs and compare it with the source on regions sampled where
    the source has tissue-like (non-white) content."""
    svs = openslide.OpenSlide(str(out))
    fmt = openslide.OpenSlide.detect_format(str(out))
    mpp = float(svs.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan"))
    rng = np.random.default_rng(0)
    w, h = source.dimensions
    diffs = []
    for _ in range(200):
        if len(diffs) >= n:
            break
        x, y = int(rng.integers(0, max(1, w - 512))), int(rng.integers(0, max(1, h - 512)))
        a = rgb(source.read_region((x, y), 0, (512, 512))).astype(float)
        if a.mean() > 220:          # mostly background; keep looking for tissue
            continue
        b = rgb(svs.read_region((x, y), 0, (512, 512))).astype(float)
        diffs.append(float(np.abs(a - b).mean()))
    result = {"out_format": fmt, "out_mpp_x": mpp, "out_width": svs.dimensions[0],
              "out_height": svs.dimensions[1], "out_levels": svs.level_count,
              "pixel_mae": round(float(np.mean(diffs)), 2) if diffs else None,
              "regions_checked": len(diffs)}
    ok = (fmt == "aperio" and svs.dimensions == source.dimensions
          and abs(mpp - meta["mpp_x"]) < 1e-4 and (not diffs or np.mean(diffs) < 8))
    result["valid"] = bool(ok)
    svs.close()
    return result


def summarize(out_dir: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not rows:
        raise SystemExit(f"ERROR: no .json reports in {out_dir}")
    columns = [k for k in rows[0] if k != "ventana"]
    with open(out_dir / "conversion_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def counts(key, digits=None):
        values = [r.get(key) if digits is None or r.get(key) is None else round(r[key], digits)
                  for r in rows]
        return dict(Counter(values))

    maes = sorted(r["pixel_mae"] for r in rows if r.get("pixel_mae") is not None)
    print(f"{len(rows)} slides; valid: {sum(bool(r.get('valid')) for r in rows)}/{len(rows)}")
    print("magnification (objective power) :", counts("objective_power"))
    print("mpp_x                           :", counts("mpp_x", 4))
    print("magnification implied by mpp    :", counts("magnification_from_mpp"))
    print(f"source {sum(r['source_bytes'] for r in rows) / 1e9:.1f} GB -> svs "
          f"{sum(r['output_bytes'] for r in rows) / 1e9:.1f} GB; pixel MAE median "
          f"{maes[len(maes) // 2] if maes else float('nan'):.2f}")
    print(f"wrote {out_dir / 'conversion_report.csv'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, help="One .bif file.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize", action="store_true",
                   help="Merge the per-slide .json reports into conversion_report.csv.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_dir)
        return 0
    if not args.input:
        raise SystemExit("ERROR: --input is required unless --summarize")

    out = args.out_dir / f"{args.input.stem}.svs"
    report = args.out_dir / f"{args.input.stem}.json"
    if out.exists() and report.exists() and not args.overwrite:
        print(f"skip {args.input.name}: already converted")
        return 0
    meta = convert(args.input, out)
    report.write_text(json.dumps(meta, indent=2, default=str))
    print(f"{args.input.name}: {meta['width']}x{meta['height']} mpp {meta['mpp_x']} "
          f"objective {meta['objective_power']} -> {out.name} "
          f"({meta['output_bytes'] / 1e9:.2f} GB, {meta['seconds']}s) "
          f"format={meta['out_format']} pixel_mae={meta['pixel_mae']} valid={meta['valid']}")
    return 0 if meta["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
