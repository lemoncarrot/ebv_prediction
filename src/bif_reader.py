"""OpenSlide-compatible reader for Ventana BIF that OpenSlide cannot open.

VENTANA DP 200/600 scans record tile joints with Direction="LEFT", which no
released OpenSlide accepts (openslide issue #759 / PR #760). This reader lays
the scanner frames (TIFF tiles) on their grid and IGNORES the joint overlaps.
On the robertoherrera DP 600 slides that means ~25% of joints keep a duplicated
24 px (11 um) strip and rows drift by up to ~1% -- acceptable for a quick
embedding pass, not a faithful image. `overlaps_ignored` records how many.

Provides the subset of openslide.OpenSlide that SlideReader uses: properties
(mpp, objective power), level_count/dimensions/downsamples, read_region,
get_thumbnail, close. Pure tifffile (tile index only) + Pillow (JPEG decode),
so it needs neither OpenSlide nor imagecodecs. Thread-safe for read_region.
"""

from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

MPP_X, MPP_Y, OBJECTIVE = "openslide.mpp-x", "openslide.mpp-y", "openslide.objective-power"


class BifSlide:
    backend = "tifffile-bif (tile overlaps ignored)"

    def __init__(self, path: str | Path, cache_tiles: int = 128):
        self.path = Path(path)
        self._tif = tifffile.TiffFile(self.path)
        self._io_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        levels = []
        for page in self._tif.pages:
            m = re.match(r"level=(\d+)", page.description or "")
            if m and page.is_tiled:
                levels.append((int(m.group(1)), page))
        if not levels:
            raise ValueError(f"{self.path.name}: no 'level=N' tiled pages; not a Ventana BIF")
        self._pages = [p for _, p in sorted(levels, key=lambda t: t[0])]

        xmp = "".join(pg.tags["XMP"].value.decode("utf-8", "ignore")
                      if isinstance(pg.tags["XMP"].value, bytes) else str(pg.tags["XMP"].value)
                      for pg in self._tif.pages if "XMP" in pg.tags)
        iscan = re.search(r"<iScan[^>]*>", xmp)
        self.scan = dict(re.findall(r'(\w+)="([^"]*)"', iscan.group(0))) if iscan else {}
        overlaps = re.findall(r'OverlapX="(-?\d+)"\s+OverlapY="(-?\d+)"', xmp)
        self.joints = len(overlaps)
        self.overlaps_ignored = sum(1 for ox, oy in overlaps if int(ox) or int(oy))
        self.blank_tiles = 0                 # empty/undecodable frames read as white

        base = self._pages[0]
        self.level_count = len(self._pages)
        self.level_dimensions = [(p.imagewidth, p.imagelength) for p in self._pages]
        self.dimensions = self.level_dimensions[0]
        self.level_downsamples = [base.imagewidth / p.imagewidth for p in self._pages]
        self.properties = {k: v for k, v in (
            (MPP_X, self.scan.get("ScanRes")), (MPP_Y, self.scan.get("ScanRes")),
            (OBJECTIVE, self.scan.get("Magnification")),
            ("openslide.vendor", "ventana"),
            ("bif.scanner", self.scan.get("ScannerModel")),
            ("bif.joints", str(self.joints)),
            ("bif.overlaps_ignored", str(self.overlaps_ignored))) if v is not None}
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cache_size = cache_tiles

    def _tile(self, level: int, index: int) -> np.ndarray:
        key = (level, index)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        page = self._pages[level]
        with self._io_lock:
            fh = self._tif.filehandle
            fh.seek(page.dataoffsets[index])
            data = fh.read(page.databytecounts[index])
        tables = page.jpegtables
        if data and tables:                          # abbreviated stream: splice in the tables
            data = tables[:-2] + data[2:]
        try:
            tile = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            # empty/unreadable frame (unscanned area): white, never tissue; counted
            tile = np.full((page.tilelength, page.tilewidth, 3), 255, dtype=np.uint8)
            with self._cache_lock:
                self.blank_tiles += 1
        with self._cache_lock:
            self._cache[key] = tile
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return tile

    def read_region(self, location: tuple[int, int], level: int,
                    size: tuple[int, int]) -> Image.Image:
        """Location in level-0 pixels, RGBA out, transparent outside the image."""
        page = self._pages[level]
        ds = self.level_downsamples[level]
        x0, y0 = int(location[0] / ds), int(location[1] / ds)
        w, h = int(size[0]), int(size[1])
        out = np.zeros((h, w, 4), dtype=np.uint8)
        tw, tl = page.tilewidth, page.tilelength
        across, down = -(-page.imagewidth // tw), -(-page.imagelength // tl)
        for ty in range(max(0, y0 // tl), min(down, (y0 + h - 1) // tl + 1)):
            for tx in range(max(0, x0 // tw), min(across, (x0 + w - 1) // tw + 1)):
                gx0, gy0 = max(x0, tx * tw), max(y0, ty * tl)
                gx1 = min(x0 + w, (tx + 1) * tw, page.imagewidth)
                gy1 = min(y0 + h, (ty + 1) * tl, page.imagelength)
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                tile = self._tile(level, ty * across + tx)
                out[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0, :3] = \
                    tile[gy0 - ty * tl:gy1 - ty * tl, gx0 - tx * tw:gx1 - tx * tw]
                out[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0, 3] = 255
        return Image.fromarray(out, "RGBA")

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """Whole slide, fitted within `size`, from the smallest level that covers it."""
        target = max(self.dimensions[0] / size[0], self.dimensions[1] / size[1])
        level = max([i for i, d in enumerate(self.level_downsamples) if d <= target] or [0])
        lw, lh = self.level_dimensions[level]
        region = self.read_region((0, 0), level, (lw, lh))
        white = Image.new("RGB", region.size, (255, 255, 255))
        white.paste(region, mask=region.split()[3])
        scale = min(size[0] / lw, size[1] / lh, 1.0)
        return white.resize((max(1, int(lw * scale)), max(1, int(lh * scale))), Image.BOX)

    def close(self) -> None:
        self._tif.close()
