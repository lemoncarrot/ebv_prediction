"""Attention scaling for display. Pure numpy, no I/O, no QuPath.

Raw attention sums to 1 over a slide, so its scale depends on tile count and a
single hot tile flattens everything else. Each function answers one question:

    percentile(a)        where did the model look most, on this slide?
    rel_log2(a)          how concentrated is attention? 0 = uniform share
    tile_score_limit()   one symmetric colour limit for tile_logit across slides
    band(pct)            fixed-colour band class for a percentile
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def percentile(attention: np.ndarray) -> np.ndarray:
    """Within-slide percentile rank, 0-100 (ties share the average rank).

    Every slide gets a full 0-100 range, including slides with diffuse
    attention -- the README must say "highest on this slide", not "important".
    """
    n = len(attention)
    if n == 1:
        return np.array([100.0])
    return (rankdata(attention, method="average") - 1) / (n - 1) * 100.0


def rel_log2(attention: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """log2(attention * n_tiles), clipped. Keeps absolute concentration: a
    diffuse slide stays near 0 everywhere, a focal one lights up."""
    relative = np.maximum(attention * len(attention), np.finfo(np.float64).tiny)
    return np.clip(np.log2(relative), *clip)


def tile_score_limit(tile_logits: list[np.ndarray], quantile: float) -> float:
    """Symmetric clip for tile_logit, from |logit| pooled over ALL review slides,
    so the diverging colour means the same thing on every slide."""
    pooled = np.abs(np.concatenate([np.asarray(t, dtype=float) for t in tile_logits]))
    return float(np.quantile(pooled, quantile))


def clip_tile_score(tile_logit: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(tile_logit, -limit, limit)


def band(pct: np.ndarray, bands: list[tuple]) -> np.ndarray:
    """Band class name per tile from config.ATTENTION_BANDS (lower-inclusive;
    the top band also includes 100)."""
    names = np.empty(len(pct), dtype=object)
    for name, lower, upper, _ in bands:
        inside = (pct >= lower) & ((pct < upper) | (upper >= 100.0))
        names[inside & (names == None)] = name  # noqa: E711 -- first matching band wins
    return names
