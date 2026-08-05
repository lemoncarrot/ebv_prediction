# UNI2-h slide embeddings

This project extracts patch-level UNI2-h embeddings from the `.svs` slides in
`00_new_HE`. The output follows the public MahmoodLab TCGA feature format: one
HDF5 file per slide containing `coords` (`N x 2`, level-0 pixel locations) and
`features` (`N x 1536`).

## Preprocessing, concisely

1. Read each slide's microns-per-pixel (MPP), falling back to objective power,
   and compute the exact scale to 20x (0.50 um/px). The code checks X/Y spacing,
   warns if MPP and objective power disagree, and resamples each field of view;
   for example, a 40x slide contributes a 512-level-0-pixel field resized to
   256 px, while a native 20x slide contributes 256 level-0 pixels.
2. At a low-resolution pyramid level, convert the slide to HSV and segment the
   saturation channel using the [CLAM TCGA preset](https://github.com/mahmoodlab/CLAM/blob/master/presets/tcga.csv): threshold 8, median filter 7, morphological close 4, tissue-area filter 16, hole-area filter 4, and at most 8 holes per contour.
3. Tile segmented tissue without overlap at **20x** using **256 x 256 px** patches
   and CLAM's permissive four-point contour rule. Store coordinates only; also
   save a QC image with tissue (green) and excluded holes (blue).
4. Read each selected patch from the WSI at 20x. UNI2's official transform
   resizes/crops it to **224 x 224**, converts it to a tensor, and applies
   ImageNet normalization.
5. Run the UNI2-h global representation (CLS token) without test-time
   augmentation and save one **1536-dimensional** float32 vector per coordinate.

The public [UNI2-h feature dataset](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)
documents the 20x/256 px sampling and HDF5 fields, but does not publish its exact
tissue-filtering implementation. The classical mask here uses MahmoodLab's
published TCGA-specific CLAM preset and records all parameters in the HDF5
attributes. Always inspect `01_preprocessed/qc/` before feature extraction.

## Run

After completing the official [UNI installation and gated-model access](https://github.com/mahmoodlab/UNI#installation),
install the WSI-stage dependencies if they are not already present:

```bash
pip install openslide-python opencv-python h5py pillow numpy
```

On Linux, `openslide-python` also requires the system OpenSlide library (or use
`openslide-bin` where appropriate for the environment).

First smoke-test one slide:

```bash
python src/00_preprocessing.py --limit 1
python src/01_call_uni.py --limit 1 --batch-size 16
```

Then process the cohort:

```bash
python src/00_preprocessing.py
python src/01_call_uni.py --batch-size 32 --workers 4
```

Defaults and outputs:

- input slides: `00_new_HE/**/*.svs`
- coordinate files and QC: `01_preprocessed/patches/`, `01_preprocessed/qc/`
- UNI2-h embeddings: `02_uni2_embeddings/*.h5`
- failed/skipped slide status: each output directory's `manifest.csv`

Both stages resume by skipping existing outputs. Use `--overwrite` deliberately
to replace them. If slide metadata lacks both objective power and MPP, supply the
known scanner setting, for example `--base-magnification 40`.
