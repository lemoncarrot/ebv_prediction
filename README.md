# EBV prediction from H&E whole-slide images

Predicts EBV status of gastric adenocarcinoma from routine H&E slides:
UNI2-h tile embeddings + attention-based multiple-instance learning (ABMIL),
developed on TCGA-STAD and an internal Chilean cohort, with external cohorts
scored blind.

| cohort | slides / patients | EBV+ | EBV label |
|---|---|---|---|
| TCGA-STAD | 306 / 284 | 25 slides | molecular subtype (EBV) |
| Chile (internal) | 130 / 130 | 22 | EBER-ISH |
| CPTAC-STAD (planned external) | 159 tumours | 13 | molecular subtype (LinkedOmics `GC_metatable.tsi`) |
| robertoherrera (external, blinded) | 69 slides | collaborator-held | collaborator-held |

## Pipeline

1. **Embed** (`src/run_pipeline.py`): tissue segmentation, then 256 px tiles read
   at a fixed **0.5 µm/px** (20x-equivalent, whatever the scan magnification;
   resolution from the slide's MPP metadata), encoded with **UNI2-h** (1536-d).
   One HDF5 per slide: `coords` (level-0 tile origins) + `features`.
2. **Train** (`src/prediction/train.py`): gated-attention ABMIL on per-slide bags.
   5-fold patient-grouped CV (stratified on cohort x label) gives training
   metrics only; the **final model is retrained on all training slides** for the
   median CV best epoch and scored **once** on the external set.
   - `--label-scheme 3class` (EBV+ / MSI / other; EBV scored as P(EBV)) and
     `--msi-weight` address MSI-high tumours, the main source of false positives.
3. **Evaluate / interpret**: PR curves and confusion matrices (`src/plotting/`),
   MSI-confound and multi-seed comparisons (`src/analysis/`), per-tile attention
   (`attention.h5`) and a blinded QuPath review package (`src/qupath/`).
4. **New cohorts** (`src/data/`): format conversion (Ventana `.bif` -> `.svs`),
   embedding, and blinded prediction (`src/prediction/predict.py`) with a
   standalone scoring script for collaborators (`src/analysis/evaluate_predictions.py`).

## Layout

```
environment.yml              conda env (uni2h); torch built for CUDA 12.1 on Sherlock
metadata/                    cohort_labels.csv (labels + QC), ebv_status.csv, STAD_merged_table.csv
src/
  run_pipeline.py            segment + tile + UNI2-h embed (step 1)
  SlideReaderClass.py        fixed-µm/px tile reader (OpenSlide)
  TissueSegmentation.py      tissue mask + QC overlay
  encoders/UNI.py            UNI2-h wrapper (needs MahmoodLab `uni`, gated HF weights)
  prediction/                abmil.py, cohort.py (labels, splits, bags), train.py, predict.py,
                             export_attention.py, train.sbatch (tcga2chile|chile2tcga|mixed|all)
  plotting/                  PR curves, confusion matrices, lab-meeting slide figures
  analysis/                  msi_confound.py, compare_runs.py, cohort_table.py,
                             evaluate_predictions.py (standalone, for collaborators)
  data/                      convert_bif_to_svs.py/.sbatch, setup_encoder.sbatch,
                             check_encoder.py, embed_cohort.sbatch, download_cptac.sbatch
  qupath/                    blinded attention review for QuPath 0.7 (overlays, scripts, analysis)
  metadata_prep/             scripts that built metadata/ (as originally run; old paths)
```

## Running (Sherlock)

Everything runs as Slurm jobs; see the header of each `.sbatch` for usage.

```bash
cd src/prediction
sbatch train.sbatch tcga2chile                                   # train TCGA, test Chile
sbatch --export=ALL,TAG=3class_s0 train.sbatch tcga2chile --label-scheme 3class --seed 0
sbatch --export=ALL,TAG=binary_s0 train.sbatch all --seed 0      # all TCGA+Chile, for new cohorts
```

New external cohort: see `src/data/README.md` (convert -> set up encoder once ->
embed -> `predict.py` -> collaborator runs `evaluate_predictions.py`).

## Notes

- Tiling is by MPP, not objective power: tiles always cover 128 µm. All Chile and
  most TCGA slides are scanned at ~40x (0.25 µm/px) and downsampled.
- Embeddings must come from the same encoder setup as training;
  `src/data/check_encoder.py` verifies a new machine reproduces stored embeddings.
- Ventana DP 200/600 `.bif` files use LEFT tile joints that released OpenSlide
  cannot open; `convert_bif_to_svs.py` falls back to a tifffile reader and refuses
  files whose tile joints overlap.
- Paths in the `.sbatch` files point at the project on Sherlock
  (`/scratch/users/kchen24/EBV_prediction`) and the group conda envs.
