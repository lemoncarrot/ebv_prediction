"""Resolve each slide's level-0 downsample so a patch lands at a fixed physical scale.

We resample by the slide's measured MPP, not its objective power. UNI2-h is
sensitive to the field of view: on a Leica GT450 whose "40x" is 0.263 um/px
rather than 0.25, honouring the objective would shift the scale ~5% and move
patch embeddings measurably. Resampling to a true um/px is scanner-independent
and keeps two cohorts on the same scale.
"""

from __future__ import annotations

import openslide

STANDARD_OBJECTIVES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0)


def _prop_float(slide: openslide.OpenSlide, name: str) -> float | None:
    raw = slide.properties.get(name)
    value = float(raw) if raw else None
    return value if value and value > 0 else None


def level0_mpp(slide: openslide.OpenSlide) -> float | None:
    """Measured level-0 pixel spacing in um/px, or None if the slide omits it."""
    return _prop_float(slide, openslide.PROPERTY_NAME_MPP_X)


def objective_power(slide: openslide.OpenSlide, fallback: float | None = None) -> tuple[float, str]:
    """Level-0 magnification, for metadata/QC only (physical scale ignores it).

    Prefers the reported objective; otherwise derives it from MPP (0.25 um/px ~
    40x) and snaps to the nearest standard objective.
    """
    objective = _prop_float(slide, openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if objective:
        return objective, "objective-power"
    mpp = level0_mpp(slide)
    if mpp:
        mag = 10.0 / mpp
        nearest = min(STANDARD_OBJECTIVES, key=lambda v: abs(v - mag))
        return (nearest, "mpp (snapped)") if abs(nearest - mag) / nearest <= 0.15 else (mag, "mpp")
    if fallback is not None:
        return fallback, "fallback"
    raise ValueError("slide has neither objective-power nor MPP; pass --base-magnification")


def target_downsample(slide: openslide.OpenSlide, target_mpp: float) -> tuple[float, str]:
    """Level-0 downsample that yields `target_mpp` um/px patches."""
    mpp = level0_mpp(slide)
    if mpp is None:
        raise ValueError("slide has no MPP metadata; cannot resample to a physical scale")
    return target_mpp / mpp, f"mpp {mpp:g} -> {target_mpp:g} um/px"
