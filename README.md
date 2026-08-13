# UNI2-h slide embeddings

This project extracts patch-level UNI2-h embeddings from the `.svs` slides in
`00_new_HE`. The output follows the public MahmoodLab TCGA feature format: one
HDF5 file per slide containing `coords` (`N x 2`, level-0 pixel locations) and
`features` (`N x 1536`).

## Preprocessing, concisely

1. Read each slide's level-0 MPP and resample so the extracted patch is a true
   0.5 um/px: a 512-level-0-pixel field on a 0.25 um/px 40x slide (resized to
   256 px), 256 px on a 0.50 um/px 20x slide, and ~487 px on a Leica GT450's
   0.263 um/px "40x". Resampling by measured MPP rather than by objective power
   keeps the field of view scanner-independent, which is what puts two cohorts
   on one scale — UNI2-h is sensitive to the field of view, see the validation
   numbers below. Objective power is recorded for QC but does not set the scale.
2. At a low-resolution pyramid level, convert the slide to HSV and segment the
   saturation channel using the [CLAM TCGA preset](https://github.com/mahmoodlab/CLAM/blob/master/presets/tcga.csv): threshold 8, median filter 7, morphological close 4, tissue-area filter 16, hole-area filter 4, and at most 8 holes per contour.
3. Tile segmented tissue without overlap at **20x** using **256 x 256 px** patches
   and CLAM's permissive four-point contour rule. The grid is anchored at the
   slide origin, so every coordinate is an exact multiple of the level-0 patch
   size, matching the published TCGA lattice. Store coordinates only; also save
   a QC image: green outlines are retained tissue contours, red outlines are
   excluded holes, and a magenta tint marks pen ink removed from the mask.
4. Read each selected patch from the WSI at 20x. UNI2's official transform
   resizes/crops it to **224 x 224**, converts it to a tensor, and applies
   ImageNet normalization.
5. Run the UNI2-h global representation (CLS token) without test-time
   augmentation and save one **1536-dimensional** float32 vector per coordinate.

### What the CLAM-style preprocessing does

CLAM (Clustering-constrained Attention Multiple-instance learning) includes a
widely used whole-slide preprocessing pipeline. This project uses its
TCGA-specific tissue segmentation and tiling conventions; it does **not** run a
trained CLAM classifier and does not attempt to label tumor versus normal
tissue.

For each slide, preprocessing works as follows:

1. **Choose a low-resolution view.** A downsampled pyramid level near 64x
   downsampling is used so the entire slide can be segmented efficiently.
2. **Detect tissue by color.** The RGB thumbnail is converted to HSV. Glass and
   empty background generally have low saturation, whereas H&E-stained tissue
   has higher saturation. The saturation image is median-filtered, thresholded,
   and morphologically closed to remove speckles and join small gaps.
3. **Construct tissue contours.** Connected foreground regions become tissue
   contours. Very small regions are removed as debris. Large enclosed background
   regions become holes, such as empty spaces between tissue fragments.
4. **Generate a 20x coordinate grid.** A single global lattice is laid over the
   slide, anchored at level-0 `(0, 0)` with a step of `patch_size x
   target_downsample` (512 px for a 40x slide, 256 px for a 20x slide). Each
   contour's bounding box is snapped back to that lattice, so contours share one
   grid phase rather than each starting at its own bounding-box corner. CLAM's
   `four_pt` rule then tests four points around a candidate tile's center; the
   tile is retained when at least one is inside the tissue contour and its
   center is not inside an excluded hole.
5. **Store coordinates, not images.** The preprocessing HDF5 contains an
   `N x 2` array named `coords`. Every row is the level-0 `[x, y]` location of a
   tile's upper-left corner. Patch size, magnification, downsampling, slide size,
   and segmentation settings are stored as HDF5 attributes. The original `.svs`
   is still required during feature extraction.

This is deliberately a permissive tissue mask: it excludes obvious background
and small artifacts while retaining most stained tissue for downstream
learning. Review the QC contour images because folds, pen marks, weak staining,
or unusual slide backgrounds can affect an automated saturation mask.

### What 256 px at 20x to 224 px means

The selected field of view is always defined as **256 x 256 pixels at 20x**. The
number of level-0 pixels read from the `.svs` changes with its native resolution:

- a 20x / 0.50 um-per-pixel slide supplies a 256 x 256 level-0 region;
- a 40x / 0.25 um-per-pixel slide supplies a 512 x 512 level-0 region, which is
  downsampled to 256 x 256.

UNI2-h expects a 224 x 224 tensor. Its official transform applies `Resize(224)`
followed by `CenterCrop(224)`. Because these pathology patches are square,
`Resize(224)` normally changes the complete 256 x 256 patch to 224 x 224 and the
center crop removes nothing further. It therefore preserves the same tissue
field of view and 20x sampling definition; it only changes the image array size
required by the encoder. A non-square input would instead be resized by its
shorter side and center-cropped to a square.

Finally, RGB values are converted to a tensor and normalized with ImageNet mean
`(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. UNI2-h then produces one 1536-value feature vector for
each retained coordinate.


The public [UNI2-h feature dataset](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)
documents the 20x/256 px sampling and HDF5 fields, but does not publish its exact
tissue-filtering implementation. The classical mask here uses MahmoodLab's
published TCGA-specific CLAM preset and records all parameters in the HDF5
attributes. Always inspect `01_preprocessed/qc/` before feature extraction.

## Validation against the published TCGA features

`ebv.validation.fetch_tcga` downloads TCGA diagnostic slides from the GDC
(open access, no token) whose names match the precomputed feature files;
`ebv.validation.validate` then runs the encoder check below. (A one-time tiling
comparison — `scale-mode=reference` plus a coordinate overlay — is no longer
shipped; its findings are recorded further down for reference.)

**Encoder check — passes.** Reading patches at *their* coordinates with this
repo's code and running UNI2-h reproduces their published vectors:

| slide | native | cosine median | cosine min | rel. L2 median |
| --- | --- | --- | --- | --- |
| TCGA-HU-A4GY | 20x | 0.999971 | 0.999491 | 0.0076 |
| TCGA-HU-A4HB | 20x | 0.999956 | 0.999336 | 0.0094 |
| TCGA-MX-A5UG | 40x | 0.999956 | 0.998406 | 0.0096 |

Patch reading, resampling, the UNI2 transform, ImageNet normalization, and the
CLS-token forward pass are therefore all correct, on both the pass-through (20x)
and the 2x-downsample (40x) path.

**Field of view is the sensitive parameter.** Re-running TCGA-MX-A5UG at
deliberately wrong scales, against the same reference:

| target downsample | field-of-view error | cosine median | rel. L2 median |
| --- | --- | --- | --- |
| 2.000 | 0.0% (their contract) | 0.999956 | 0.0096 |
| 2.016 | +0.8% (raw MPP) | 0.986259 | 0.1675 |
| 2.097 | +4.8% | 0.952970 | 0.3059 |
| 1.903 | -4.8% | 0.949453 | 0.3174 |

A sub-1% scale error costs more than an order of magnitude in reconstruction
error. This is why objective power is preferred over raw MPP: every slide in
`00_new_HE` reports objective power 40 but MPP 0.2627 um/px (~38.07x), a 4.8%
disagreement that would have put this cohort's embeddings in a measurably
different part of UNI2's space from the TCGA reference.

**Tiling check (one-time, code removed) — aligned, but their coordinates are not
a complete tiling.** The findings below were produced by an earlier
`scale-mode=reference` tiling comparison that is no longer part of the shipped
code; they are kept as a record of why the release is not a tiling reference.
After anchoring the grid at the slide origin and snapping magnification, 96-99%
of their tiles are reproduced exactly, with zero coordinates off the lattice:

| slide | ours | theirs | shared | recall of theirs |
| --- | --- | --- | --- | --- |
| TCGA-BR-8592 | 17002 | 7332 | 7226 | 0.986 |
| TCGA-HU-A4GY | 1657 | 231 | 225 | 0.974 |
| TCGA-HU-A4HB | 929 | 255 | 245 | 0.961 |
| TCGA-MX-A5UG | 788 | 180 | 172 | 0.956 |

The remaining 2-7x surplus is not over-segmentation here. In the published
release, **299 of 398 STAD slides (75%) are missing whole blocks of x-columns**
while y stays contiguous; the median slide retains only 76% of the x-columns
spanned by its own coordinates, and one retains 12%.

This is dropout on a fixed grid, not tissue filtering:

- **Contiguous missing column runs are 8k+7 long** (7, 15, 23, 31 dominate;
  75.3% of runs on 40x slides, 77.9% on 20x), and contiguous *kept* runs are
  8k+1. The two sum to a multiple of 8, so the slide is chunked into 8-patch-wide
  vertical strips that are dropped whole, with the first column of each dropped
  strip surviving.
- **The unit is 8 patches, not a pixel width.** The same mod-8 signature appears
  at both magnifications, where 8 patches means 4096 level-0 px on a 40x slide
  and 2048 on a 20x one.
- **The dropped tissue is indistinguishable from the kept tissue.** On
  TCGA-BR-8592, columns they kept vs dropped: mean saturation 62.15 vs 60.50
  (pooled sd 14.54, so 0.11 sd apart) and tissue fraction 0.978 vs 0.979.

No tissue-content filter operates on an 8-patch raster, so this is a systematic
extraction or serialization loss on their side, not a preprocessing step to
reproduce. Across a 120-slide sample, 28.9% of columns are absent. The overlay
that showed this (`03_validation/overlay_*.jpg`, green where both pipelines
agree, red where only this pipeline has a tile, blue where only the reference
does) coloured clean vertical red bands over ordinary tumour tissue on
TCGA-BR-8592 — the signature of missing columns, not of over-segmentation.

Treat the release as a trustworthy **encoder** reference and not as a **tiling**
reference, and do not tune segmentation to match its tile counts.

## Run

After completing the official [UNI installation and gated-model access](https://github.com/mahmoodlab/UNI#installation),
install the WSI-stage dependencies if they are not already present:

```bash
pip install openslide-python opencv-python h5py pillow numpy
```

On Linux, `openslide-python` also requires the system OpenSlide library (or use
`openslide-bin` where appropriate for the environment).

The pipeline is a package under `src/ebv`, run with `python -m ebv.<stage>`. Put
`src` on the path first — `export PYTHONPATH=src` from the repository root (the
commands below assume it), or `pip install -e .` if you add a packaging file.

First smoke-test one slide:

```bash
python -m ebv.preprocessing.run --limit 1
python -m ebv.embedding.encode --limit 1 --batch-size 16
```

Then process the cohort:

```bash
python -m ebv.preprocessing.run
python -m ebv.embedding.encode --batch-size 32 --workers 4
```

Defaults and outputs:

- input slides: `00_new_HE/**/*.svs`
- coordinate files and QC: `01_preprocessed/patches/`, `01_preprocessed/qc/`
- UNI2-h embeddings: `02_uni2_embeddings/*.h5`
- failed/skipped slide status: each output directory's `manifest.csv`

Both stages resume by skipping existing outputs. Use `--overwrite` deliberately
to replace them. If slide metadata lacks both objective power and MPP, supply the
known scanner setting, for example `--base-magnification 40`.

To reproduce the validation:

```bash
python -m ebv.validation.fetch_tcga --auto 3     # or name slides explicitly
python -m ebv.validation.validate --max-patches 64
```

`--auto` picks the smallest precomputed slides, which are the cheapest to
encode but are also the ones most affected by the missing-column issue above.
Name a median-sized slide explicitly when assessing tiling.

## TCGA rerun, then MIL

The TCGA slides now live in `00_tcga_WSI_raw/`, a flat directory of symlinks
(`*.svs`) into `/mnt/ix1/Resources/TCGA_Data/GDC_portal/images/STAD/tcga_WSI_stad`,
where each slide sits in its own GDC UUID folder. Every consumer script points
there. Preprocessing and embedding must be rerun against it:

```bash
python -m ebv.preprocessing.run --input-dir 00_tcga_WSI_raw --output-dir 01_preprocessed_tcga
python -m ebv.embedding.encode --input-dir 00_tcga_WSI_raw \
    --coords-dir 01_preprocessed_tcga/patches \
    --output-dir 02_uni2_embeddings_tcga --batch-size 32 --workers 8
```

Shard the encoder across GPUs with `--shard-count 2 --shard-index 0|1`.

### MIL + MLP classifier

`ebv.mil.models` holds the gated-attention MIL (Ilse et al. 2018, with a masked
softmax for variable bag sizes and an MLP head on the pooled bag embedding) and
Additive MIL, registered by name so a new architecture is one decorated
`nn.Module`. `ebv.mil.train` trains via a small `Trainer`;
`ebv.mil.heatmaps` turns the saved attention into patch-importance tables and
heatmaps. Add `--arch additive` to train the additive variant, which also writes
signed per-patch contributions.

```bash
# Chile only (runnable now)
python -m ebv.mil.train --cohort chile --run-name chile_5fold

# Pooled, once the TCGA embeddings exist
python -m ebv.mil.train --features-dir 02_uni2_embeddings 02_uni2_embeddings_tcga \
    --cohort chile tcga --run-name pooled_5fold

# Train on Chile, hold TCGA out entirely as an external cohort
python -m ebv.mil.train --features-dir 02_uni2_embeddings 02_uni2_embeddings_tcga \
    --cohort chile --external-cohort tcga --run-name chile_train_tcga_ext

python -m ebv.mil.heatmaps --run-dir 04_mil/pooled_5fold --overlay
```

Cross-validation is grouped by `patient_id` and stratified on the EBV label, so
no patient spans the training and held-out data. Each fold carves an inner
grouped validation split out of its training data for early stopping and for the
Youden threshold; the outer fold is only ever scored. Slides without an EBV call
(59 TCGA cases in `metadata/cohort_labels.csv`) are dropped automatically.

Training subsamples `--max-patches` (default 4096) patches per bag per epoch;
evaluation and attention export always use the full bag.

Run directory (`04_mil/<run-name>/`):

- `slides.csv`, `folds.csv` — the resolved cohort and every split assignment
- `checkpoints/fold*.pt` — weights, model config, and the fold's threshold
- `history.csv`, `oof_predictions.csv`, `metrics.json` — slide- and
  patient-level AUROC/AUPRC/sensitivity/specificity
- `attention/<slide>.h5` — `coords` and `attention` per patch, written from the
  fold that held the slide out, with the slide geometry copied from the feature
  file so a heatmap needs no slide access
- `external_predictions.csv` — fold-ensembled scores for `--external-cohort`

Heatmap outputs (`05_attention_maps/<run-name>/`) are one row per patch
(`_patch_attention.csv`), the same values as a lattice matrix
(`_attention_grid.csv`), and the rendered map (`_attention_heatmap.png`).
Colour is `log2(attention x n_patches)`: 0 is an average patch, +1 got twice the
attention of an average patch. Raw attention is not comparable across slides
because it sums to 1 over a bag, so a 40k-patch slide sits on a 1/40000
baseline and a 3k-patch slide on 1/3000.

## Exact run log

Every command that produced the artifacts currently on disk, in order. All of
them run from the repository root inside the `UNI` conda environment:

```bash
cd /mnt/ix1/Projects_lite/kchen24/EBV_prediction
conda activate UNI    # /home/kchen24/.conda/envs/UNI, torch 2.13.0+cu130, 2x RTX 3090
export PYTHONPATH=src # the pipeline is the ebv package under src/
```

Both stages skip existing outputs, so a re-run resumes unless `--overwrite` is
passed. `--jobs`, `--workers`, `--batch-size`, and the shard flags affect
throughput only, never the stored values; every flag that *does* change the
output is listed explicitly below.

### Cohort 1 — Chile (`Dataset3_JCAraya`), 164 slides

Input is `00_new_HE`, a symlink to
`/mnt/ix1/Resources/Image_Ref/Collaborators/GC_Camargo/Dataset3_JCAraya`.

```bash
# 1. Segment + tile -> 01_preprocessed/{patches,qc}/ + manifest.csv
python -m ebv.preprocessing.run

# 2. UNI2-h embeddings -> 02_uni2_embeddings/*.h5, one shard per GPU
CUDA_VISIBLE_DEVICES=0 python -m ebv.embedding.encode --shard-count 2 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -m ebv.embedding.encode --shard-count 2 --shard-index 1
```

Both stages ran entirely on defaults — no flag that changes the output was
passed. Result: **164/164 preprocessed, 0 failed**; **164/164 embedded, 0
failed**, split 82 + 82 across the two shards for 2,638,899 patches
(1,318,790 + 1,320,109). Logs: `01_preprocessed_chile.log`,
`logs/uni_chile_shard{0,1}.log`.

### Cohort 2 — TCGA-STAD, 441 slides

Input is `00_tcga_WSI_raw`, a flat directory of symlinks into
`/mnt/ix1/Resources/TCGA_Data/GDC_portal/images/STAD/tcga_WSI_stad`, where each
slide sits in its own GDC UUID folder.

```bash
# 0. Discard the previous preprocessing run outright (slides were re-uploaded)
rm -rf 01_preprocessed_tcga/patches 01_preprocessed_tcga/qc 01_preprocessed_tcga/manifest.csv

# 1. Segment + tile -> 01_preprocessed_tcga/{patches,qc}/ + manifest.csv
python -m ebv.preprocessing.run \
    --input-dir 00_tcga_WSI_raw \
    --output-dir 01_preprocessed_tcga \
    --jobs 32 --overwrite \
    > 01_preprocessed_tcga.log 2>&1

# 2. UNI2-h embeddings -> 02_uni2_embeddings_tcga/*.h5, one shard per GPU
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python -m ebv.embedding.encode \
      --input-dir 00_tcga_WSI_raw \
      --coords-dir 01_preprocessed_tcga/patches \
      --output-dir 02_uni2_embeddings_tcga \
      --batch-size 32 --workers 16 \
      --shard-count 2 --shard-index $i \
      > 02_uni2_tcga_shard$i.log 2>&1 < /dev/null &
done
```

Preprocessing result: **399/441 ok, 42 failed.** All 42 failures are the same
error — `not a whole-slide image: 1600x1200, single pyramid level`. Those GDC
UUID folders hold ~0.5 MB placeholder files rather than real WSIs, so they
cannot be tiled and are absent from every downstream stage. Embedding covers
the 399 surviving slides, sharded 200 + 199 for 4,949,226 patches
(2,474,830 + 2,474,396).

The shards are detached with `setsid nohup` deliberately: the run takes hours
and must not die with the shell that launched it. Watch it with
`tail -f 02_uni2_tcga_shard0.log`, and note that `pkill -f ebv.embedding.encode`
will also match the launching shell's own command line — match on
`ebv.embedding.encode` and check `pgrep -af` first.

### Scale convention actually used

Both cohorts ran with the script default `--scale-mode physical`, which
resamples by each slide's measured MPP to a true 0.5 um/px patch:

| cohort | level-0 MPP | objective | `target_downsample` | patch scale |
| --- | --- | --- | --- | --- |
| Chile | 0.2627 | 40x | 1.9031 | 0.500 um/px |
| TCGA (e.g. TCGA-3M-AB46) | 0.2527 | 40x | 1.9786 | 0.500 um/px |

This is self-consistent across the two cohorts, which is what the pooled MIL
model needs. It is **not** the published-TCGA-feature convention, which derives
the scale from objective power and gives an exact `target_downsample` of 2.0 on
a 40x slide (the removed `scale-mode=reference` did this). The validation
section above measures that difference — on TCGA-MX-A5UG a 4.8% field-of-view
error moves cosine agreement with the published features from 0.99996 to ~0.95.
So treat these embeddings as internally comparable to each other, and not as
drop-in replacements for the MahmoodLab release.

### Reproducing the validation artifacts

```bash
python -m ebv.validation.fetch_tcga --auto 3
python -m ebv.validation.validate --max-patches 64
```

Outputs land in `03_validation/tcga_validation.json`.
