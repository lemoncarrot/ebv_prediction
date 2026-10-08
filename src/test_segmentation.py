"""Single-slide smoke test for the segmentation -> UNI2-h path.

Uses the cached segment_stage (which keeps mask.png etc. around to look at) --
the real cohort run is run_pipeline.py, which keeps nothing but the overlay.
"""

import numpy as np

from SlideReaderClass import SlideReader
from TissueSegmentation import segment_stage, load_mask
from encoders.UNI import UNIEncoder

slide_path = "/mnt/ix1/Projects_lite/kchen24/EBV_prediction/00_new_HE/506-A9_182329.svs"
out_dir = "/mnt/ix1/Projects_lite/kchen24/EBV_prediction/segmentation_output"

# UNI2-h sampling: a 256 px field of view at 20x, which the encoder's own
# Resize(224) then takes down to its 224 px input. Tiling at 224 here would
# shrink the field of view to 112 um and move off the validated scale.
encoder = UNIEncoder()
spec = encoder.spec
mpp_target, tile_size = spec.mpp_target, spec.tile_size

meta = segment_stage(slide_path, out_dir, mpp_target, tile_size=tile_size)
mask, thumb_ds = load_mask(out_dir, meta["slide_id"])
print(f"{meta['slide_id']}: {meta['n_tiles']} tiles, tissue {meta['tissue_fraction']:.3f}")

with SlideReader(slide_path, mpp_target=mpp_target, tile_size=tile_size) as reader:
    coords = np.array(
        list(reader.tile_origins(mask=mask, mask_downsample=thumb_ds)),
        dtype=np.int32,
    )

    embeddings = np.empty((len(coords), spec.embed_dim), dtype=np.float32)
    BatchSize = 64
    for i in range(0, len(coords), BatchSize):
        batch = coords[i: i + BatchSize]
        tiles = np.stack([reader.read_tile(x, y) for x, y in batch])
        embeddings[i: i + BatchSize] = encoder.encode(tiles)

print(coords.shape, embeddings.shape)
