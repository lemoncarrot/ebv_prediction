# implement CLAM segmentation (mahmood [UNI] lab tool)
# opencv library

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from SlideReaderClass import SlideReader

def segment_tissue(
    thumb: np.ndarray,
    thumb_mpp: float,
    sat_thresh: int | None = None,
    median_blur: int = 7,
    close_kernel: int = 4,
    min_tissue_um2: float = 4e6,
    min_hole_um2: float = 1e6,
    max_holes_per_contour: int = 8,
    max_dark_value: int | None = None,
    ) -> np.ndarray:
    # like-CLAM tissue segmentation (on lower resolution thumbnail)
    
    if median_blur % 2 == 0:
        raise ValueError("median_blur must be odd")
    
    # separate by saturation
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    
    sat = cv2.medianBlur(sat, median_blur)
    
    if sat_thresh is None:
        # otsu thresholding if not specified
        _, binary = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY)

    if max_dark_value is not None:
        # near-black pixels (scanner edges, slide borders) are never tissue
        dark = cv2.medianBlur(hsv[..., 2], median_blur) <= max_dark_value
        binary[dark] = 0

    # close pinholes (one piece of tissue stays connected)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # outer contours?
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros(thumb.shape[:2], dtype=bool)
    
    hierarchy = hierarchy[0]
    px_area_um2 = thumb_mpp ** 2
    mask = np.zeros(thumb.shape[:2], dtype=np.uint8)
    
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1: 
            continue
        
        holes = [j for j in range(len(contours)) if hierarchy[j][3] == i]
        hole_areas = sorted(
            ((cv2.contourArea(contours[j]), j) for j in holes), reverse=True
        )
        
        net_area = (cv2.contourArea(cnt) - sum(a for a, _ in hole_areas)) * px_area_um2
        if net_area < min_tissue_um2:
            continue
        
        cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
        
        for area, j in hole_areas[:max_holes_per_contour]:
            if area * px_area_um2 < min_hole_um2:
                cv2.drawContours(mask, [contours[j]], -1, 0, thickness=cv2.FILLED)

    return mask.astype(bool)

def _params_key(params: dict) -> str:
    blob = json.dumps({**params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]

def _write_overlay(path: Path, thumb: np.ndarray, mask: np.ndarray) -> None:
    # save thumbnail with tissue mask overlaid
    
    overlay = thumb.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)
    
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), params):
        # imwrite reports failure by return value, not by raising
        raise OSError(f"could not write overlay to {path}")


def segment_slide(
    reader: SlideReader,
    max_dim: int = 4096,
    qc_path: Path | str | None = None,
    **seg_params,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Segment tissue from an open slide, in memory.
    
    This is the entry point for the one-pass pipeline: nothing is written except
    the QC overlay at `qc_path` (skipped when None), and the mask is handed
    straight to the caller instead of round-tripping through mask.png.
    
    Returns (mask, thumb, thumb_downsample, thumb_mpp).
    """
    thumb, thumb_ds = reader.thumbnail(max_dim=max_dim)
    thumb_mpp = reader.geometry.mpp_base * thumb_ds
    mask = segment_tissue(thumb, thumb_mpp=thumb_mpp, **seg_params)
    
    if qc_path is not None:
        qc_path = Path(qc_path)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        _write_overlay(qc_path, thumb, mask)
    
    return mask, thumb, thumb_ds, thumb_mpp

def segment_stage(
    slide_path: str,
    out_dir: str,
    mpp_target: float,
    tile_size: int = 256,
    max_dim: int = 4096,
    overwrite: bool = False,
    **seg_params,
) -> dict:

    # segments tissue, writing a small thumbnail, mask, overlay, metadata (json)
    
    # metadata is saved with a specific hash so work skipped if previously run
    
    slide_id = Path(slide_path).stem
    out = Path(out_dir) / slide_id
    meta_path = out / "seg.json"
    
    key = _params_key({"max_dim": max_dim, "tile_size": tile_size, **seg_params})
    
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("params_key") == key:
            return meta
        
    out.mkdir(parents=True, exist_ok = True)
    
    with SlideReader(slide_path, mpp_target=mpp_target, tile_size=tile_size) as reader:
        g = reader.geometry
        mask, thumb, thumb_ds, thumb_mpp = segment_slide(reader, max_dim=max_dim, **seg_params)
        n_tiles = sum(
            1 for _ in reader.tile_origins(mask=mask, mask_downsample=thumb_ds)
        )
        
    cv2.imwrite(str(out / "thumbnail.png"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask.png"), mask.astype(np.uint8) * 255)
    _write_overlay(out / "overlay.jpg", thumb, mask)
    
    meta = {
        "slide_id": slide_id,
        "slide_path": str(slide_path),
        "params_key": key,
        # Required to interpret mask.png — the mask is meaningless alone.
        "thumb_downsample": thumb_ds,
        "thumb_mpp": thumb_mpp,
        "thumb_shape": list(thumb.shape[:2]),
        # Geometry, for auditing what scale each slide was read at.
        "mpp_base": g.mpp_base,
        "mpp_target": g.mpp_target,
        "mpp_achieved": g.mpp_achieved,
        "mpp_source": g.mpp_source,
        "level": g.level,
        "read_size": g.read_size,
        # QC signals — sort a cohort by these to find failures fast.
        "tissue_fraction": float(mask.mean()),
        "n_tiles": n_tiles,
        "tile_size": tile_size,
        "params": {"max_dim": max_dim, "tile_size": tile_size, **seg_params},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# function to get the tissue segmentation for downstream
def load_mask(out_dir: str, slide_id: str) -> tuple[np.ndarray, float]:
    out = Path(out_dir) / slide_id
    meta = json.loads((out / "seg.json").read_text())
    mask = cv2.imread(str(out / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    return mask, meta["thumb_downsample"]


        