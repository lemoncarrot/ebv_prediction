from dataclasses import dataclass
from typing import Iterator, Literal, Optional
import numpy as np

import math
import cv2
import openslide

class MissingResolutionError(Exception):
    """no usable MPP metadata, throw big error here"""

    
_OBJECTIVE_TO_MPP = {40: 0.25, 20: 0.5, 10: 1.0}
_DS_TOL = 1e-2

@dataclass(frozen=True)
class SlideGeometry:
    mpp_base: float 
    mpp_target: float
    level: int
    level_downsample: float
    residual: float
    mpp_achieved: float
    read_size: int
    step_level0: float
    mpp_source: Literal["mpp-x", "objective-power"]
    
class SlideReader:
    # read tiles at a fixed physical resolution, reference as level 0 pixels
    
    
    def __init__(self, path: str, mpp_target: float, tile_size: int):
        self.path = str(path)
        self.mpp_target = mpp_target
        self.tile_size = int(tile_size)
        self._slide = openslide.OpenSlide(self.path)
        self.geometry = self._resolve_geometry()
        
    def _resolve_mpp(self) -> tuple[float, str]:
        # first gets MPP, if not available, fall back to objective power (maps to MPP roughly)
        
        props = self._slide.properties # get metadata of the slide
        
        mx = props.get(openslide.PROPERTY_NAME_MPP_X)
        my = props.get(openslide.PROPERTY_NAME_MPP_Y)
        
        if mx and my:
            mx, my = float(mx), float(my)
            if abs(mx - my) / min(mx, my) > 0.01:
                raise MissingResolutionError("likely a metadata error")
            
            return (mx + my ) / 2.0, "mpp-x"
        
        obj = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if obj:
            key = int(round(float(obj)))
            if key in _OBJECTIVE_TO_MPP:
                return _OBJECTIVE_TO_MPP[key], "objective-power"
            
        raise MissingResolutionError(f"{self.path}: this file sucks and has no usable metadata")
        
    def _resolve_geometry(self) -> SlideGeometry:
        mpp_base, source = self._resolve_mpp()
        target_ds = self.mpp_target / mpp_base
        
        if target_ds < 1.0 - _DS_TOL:
            raise ValueError("target MPP smaller than base, cannot increase resolution lol")
        
        level = 0
        for i, ds in enumerate(self._slide.level_downsamples):
            if ds <= target_ds + _DS_TOL:
                level = i
            else:
                break
            
        # see which level we should use to downsample (lower resolution better so it is faster)
        
        ds = self._slide.level_downsamples[level]
        residual = target_ds / ds
        read_size = max(1, int(round(self.tile_size * residual)))
        
        mpp_achieved = mpp_base * ds * (read_size / self.tile_size)
        
        return SlideGeometry(
            mpp_base=mpp_base,
            mpp_target=self.mpp_target,
            level=level,
            level_downsample=ds,
            residual=residual,
            mpp_achieved=mpp_achieved,
            read_size=read_size,
            step_level0=self.tile_size * target_ds,
            mpp_source=source,
        )
    def read_tile(self, x0: int, y0: int) -> np.ndarray:
        g = self.geometry
        
        region = self._slide.read_region((int(x0), int(y0)), g.level, (g.read_size, g.read_size))
        
        rgba = np.asarray(region, dtype=np.uint8)
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        
        if (alpha != 255).any():
            a = alpha.astype(np.float32) / 255.0
            rgb = (rgb.astype(np.float32) + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
            
        if g.read_size != self.tile_size:
            rgb = cv2.resize(rgb, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA)
            
        return np.ascontiguousarray(rgb)
    
    def tile_origins(
        self,
        mask: Optional[np.ndarray] = None,
        mask_downsample: Optional[float] = None,
        min_tissue: float = 0.10,
        drop_partial: bool = True
    ) -> Iterator[tuple[int, int]]:
        # get level 0 tile origins (top left is origin)
        g = self.geometry
        w0, h0 = self._slide.level_dimensions[0]
        span = g.step_level0
        
        ny = math.floor(h0 / span) if drop_partial else math.ceil(h0 / span)
        nx = math.floor(w0 / span) if drop_partial else math.ceil(w0 / span)
        
        for j in range(ny):
            for i in range(nx):
                x0, y0 = i * span, j * span
                if mask is not None and not self._tile_has_tissue(x0, y0, span, mask, mask_downsample, min_tissue):
                    continue
                yield int(round(x0)), int(round(y0))

    @staticmethod
    def _tile_has_tissue(x0, y0, span, mask, mask_ds, min_tissue) -> bool:
        s = span / mask_ds
        r0, c0 = int(y0 / mask_ds), int(x0 / mask_ds)
        r1, c1 = int(math.ceil(r0 + s)), int(math.ceil(c0 + s))
        patch = mask[r0:r1, c0:c1]
        return patch.size > 0 and patch.mean() >= min_tissue
    
    def thumbnail(self, max_dim: int=4096) -> tuple[np.ndarray, float]:
        # low res rgb, used downstream, also saves downsample factor
        
        w0, h0 = self._slide.level_dimensions[0]
        ds = max(max(w0, h0) / max_dim, 1.0)
        thumb = self._slide.get_thumbnail((int(w0 / ds), int(h0 / ds)))
        return np.asarray(thumb.convert("RGB"), dtype=np.uint8), ds
    
    def close(self):
        self._slide.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
        
        
