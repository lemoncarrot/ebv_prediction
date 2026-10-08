"""Shared constants for the QuPath review. Plain data, no logic.

Everything a reviewer sees (class names, colours, bands, limits) is fixed here so
that the overlays, the package README and the analysis all agree.
Target: QuPath 0.7.0.
"""

from __future__ import annotations

QUPATH_VERSION = "0.7.0"

# --------------------------------------------------------------------------
# review set: how many slides of each outcome group (see select_slides.py)
# --------------------------------------------------------------------------
# Groups come from the final model's external predictions at THRESHOLD:
#   tp, fn, fp, tn                 -- slides with no MSI annotation (e.g. Chile)
#   fp_msi, fp_other, tn_msi, tn_other  -- TCGA negatives, split by MSI subtype
# A group missing from the run is skipped; a short group is reported, not padded.
THRESHOLD = 0.5
SAMPLING_PLAN = {
    "tp": 10, "fn": 7,
    "fp": 10, "tn": 9,                      # used when MSI status is unknown
    "fp_msi": 10, "fp_other": 5,            # used for TCGA test sets
    "tn_msi": 0, "tn_other": 9,
}
REVIEW_CODE_PREFIX = "R"   # blinded slide names: R01, R02, ...
RANDOM_SEED = 0

# --------------------------------------------------------------------------
# attention scaling (see scaling.py)
# --------------------------------------------------------------------------
ATTN_REL_LOG2_CLIP = (-2.0, 4.0)   # 0 = uniform share; +3 = 8x uniform
TILE_SCORE_CLIP_QUANTILE = 0.99    # of |tile_logit| pooled over the review set
MEASUREMENT_DECIMALS = 3           # keeps the GeoJSON small
ATTENTION_BANDS = [                # (class name, lower pct, upper pct, RGB); lower-inclusive
    ("Attention top 5%", 95.0, 100.0, (215, 48, 39)),
    ("Attention 5-20%", 80.0, 95.0, (252, 141, 89)),
    ("Attention rest", 0.0, 80.0, (200, 200, 200)),
]

# --------------------------------------------------------------------------
# pathologist annotation classes (import_overlays.groovy adds these to the project)
# --------------------------------------------------------------------------
ANNOTATION_CLASSES = [             # (name, RGB)
    ("Tumour epithelium", (200, 0, 0)),
    ("Lymphoid stroma", (60, 60, 220)),
    ("Lymphoepithelioma-like", (150, 0, 150)),
    ("Necrosis", (90, 90, 90)),
    ("Artifact", (240, 200, 0)),
    ("Normal mucosa", (0, 150, 80)),
]

# --------------------------------------------------------------------------
# analysis (see analyze_annotations.py)
# --------------------------------------------------------------------------
TOP_ATTENTION_PCT = 95.0           # "top tiles" = attn_pct >= this
# A tile belongs to an annotation when its centre lies inside it (point-in-polygon;
# no shapely dependency). With ~110 um tiles this is close to a 50% area rule.
