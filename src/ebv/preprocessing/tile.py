"""Lay a non-overlapping patch grid over the tissue contours (CLAM four-point rule)."""

from __future__ import annotations

import cv2
import numpy as np

from ebv.preprocessing.config import Config


def _inside(contour: np.ndarray, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(contour, point, False) >= 0


def coordinates_for_contours(contours, holes_by_tissue, target_downsample: float, cfg: Config) -> np.ndarray:
    """Level-0 [x, y] corner of every retained patch.

    The grid is anchored at the slide origin (0, 0) with step `patch_size *
    downsample`, so every coordinate is an exact multiple of the level-0 patch
    size — the same lattice the published TCGA features use. A tile is kept when
    at least one of four points around its centre is inside the tissue contour
    and its centre is not inside an excluded hole.
    """
    footprint = max(1, round(cfg.patch_size * target_downsample))
    step = max(1, round((cfg.patch_size - cfg.overlap) * target_downsample))
    coords: list[tuple[int, int]] = []

    for contour, holes in zip(contours, holes_by_tissue):
        box_x, box_y, width, height = cv2.boundingRect(contour)
        start_x, start_y = box_x - (box_x % step), box_y - (box_y % step)
        for x in range(start_x, box_x + width, step):
            for y in range(start_y, box_y + height, step):
                cx, cy, q = x + footprint / 2, y + footprint / 2, footprint / 4
                probes = ((cx - q, cy - q), (cx + q, cy + q), (cx + q, cy - q), (cx - q, cy + q))
                if not any(_inside(contour, p) for p in probes):
                    continue
                if any(_inside(hole, (cx, cy)) for hole in holes):
                    continue
                coords.append((x, y))

    return np.asarray(coords, dtype=np.int64) if coords else np.empty((0, 2), dtype=np.int64)
