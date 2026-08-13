"""Preprocessing parameters, defaulting to the CLAM TCGA HSV segmentation preset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # tiling
    patch_size: int = 256          # patch edge in pixels, at the target scale
    target_mag: float = 20.0       # recorded label only; physical scale drives sampling
    target_mpp: float = 0.5        # extracted patch scale in um/px
    overlap: int = 0
    base_magnification: float | None = None  # fallback objective for metadata
    # tissue segmentation (CLAM TCGA preset)
    seg_downsample: float = 64.0
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    area_tissue: float = 16.0
    area_hole: float = 4.0
    max_holes: int = 8
    # surgical-marker (pen) removal
    pen_filter: bool = True
    pen_green_sat: int = 60
    pen_cyan_sat: int = 100
    pen_red_sat: int = 200
    pen_dark_value: int = 50
    pen_dilate: int = 5
    overwrite: bool = False
