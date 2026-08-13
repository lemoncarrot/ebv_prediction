"""Tissue segmentation on a low-res thumbnail: HSV saturation mask + pen removal."""

from __future__ import annotations

import cv2
import numpy as np
import openslide
from PIL import Image

from ebv.preprocessing.config import Config


def level_downsamples_xy(slide: openslide.OpenSlide) -> list[tuple[float, float]]:
    w0, h0 = slide.level_dimensions[0]
    return [(w0 / w, h0 / h) for w, h in slide.level_dimensions]


def closest_level(slide: openslide.OpenSlide, wanted: float) -> int:
    """Pyramid level whose downsample is closest (in log space) to `wanted`."""
    ds = level_downsamples_xy(slide)
    return min(range(len(ds)), key=lambda i: abs(np.log(max(np.mean(ds[i]), 1e-9) / wanted)))


def pen_mask(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask of surgical-marker ink. H&E occupies a narrow hue slice
    (eosin pink, hematoxylin purple); ink sits outside it, or inside at a
    saturation stain never reaches. Blue 100-125 is excluded — it overlaps real
    hematoxylin.
    """
    hue, sat, val = (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, i].astype(np.int16) for i in range(3))
    ink = (
        ((hue >= 35) & (hue < 85) & (sat >= cfg.pen_green_sat))       # green
        | ((hue >= 85) & (hue < 100) & (sat >= cfg.pen_cyan_sat))     # cyan
        | (((hue < 20) | (hue >= 175)) & (sat >= cfg.pen_red_sat))    # red
        | ((val < cfg.pen_dark_value) & (sat < 100))                 # black cores
    ).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop speckles
    if cfg.pen_dilate > 0:  # grow so a tile cannot clip an ink edge
        ink = cv2.dilate(ink, np.ones((cfg.pen_dilate, cfg.pen_dilate), np.uint8))
    return ink.astype(bool)


def segment_tissue(slide: openslide.OpenSlide, level: int, cfg: Config):
    """Return (thumbnail rgb, level-0 tissue contours, per-contour holes, seg
    downsample, pen fraction removed)."""
    rgb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))

    # 1. saturation channel -> median blur -> threshold -> morphological close
    sat = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1], cfg.mthresh)
    ttype = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if cfg.use_otsu else 0)
    _, mask = cv2.threshold(sat, 0 if cfg.use_otsu else cfg.sthresh, 255, ttype)
    if cfg.close > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.close, cfg.close), np.uint8))

    # 2. remove pen ink after the close (else the close refills its holes)
    pen_fraction = 0.0
    if cfg.pen_filter:
        ink = pen_mask(rgb, cfg)
        before = int((mask > 0).sum())
        mask[ink] = 0
        pen_fraction = (before - int((mask > 0).sum())) / before if before else 0.0

    # 3. contours: outer = tissue, children = holes; scale both back to level 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2:]
    ds_x, ds_y = level_downsamples_xy(slide)[level]
    if hierarchy is None:
        return rgb, [], [], (ds_x, ds_y), pen_fraction
    hierarchy = hierarchy[0]
    ref_area = int(512**2 / (ds_x * ds_y))  # CLAM areas are in 512-px units
    min_tissue, min_hole = cfg.area_tissue * ref_area, cfg.area_hole * ref_area
    scale = np.array([[[ds_x, ds_y]]], dtype=np.float64)

    tissue, holes_by_tissue = [], []
    for idx in np.flatnonzero(hierarchy[:, 3] == -1):
        child_ids, child = [], hierarchy[idx, 2]
        while child != -1:
            child_ids.append(int(child))
            child = hierarchy[child, 0]
        area = cv2.contourArea(contours[idx]) - sum(cv2.contourArea(contours[j]) for j in child_ids)
        if area <= min_tissue:
            continue
        kept = sorted((contours[j] for j in child_ids if cv2.contourArea(contours[j]) > min_hole),
                      key=cv2.contourArea, reverse=True)[: cfg.max_holes]
        tissue.append(np.rint(contours[idx] * scale).astype(np.int32))
        holes_by_tissue.append([np.rint(h * scale).astype(np.int32) for h in kept])
    return rgb, tissue, holes_by_tissue, (ds_x, ds_y), pen_fraction


def save_qc(rgb, contours, holes, seg_ds, path, ink=None) -> None:
    """Green = kept tissue, red = excluded holes, magenta tint = removed ink."""
    canvas = rgb.copy()
    if ink is not None and ink.any():
        canvas[ink] = (0.45 * canvas[ink] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    scale = np.array([[[1.0 / seg_ds[0], 1.0 / seg_ds[1]]]])
    cv2.drawContours(canvas, [np.rint(c * scale).astype(np.int32) for c in contours], -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, [np.rint(h * scale).astype(np.int32) for g in holes for h in g], -1, (255, 0, 0), 2)
    Image.fromarray(canvas).save(path, quality=90)
