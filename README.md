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
| External collaborator cohort (blinded) | 69 slides (VENTANA DP 600 .bif, 20x) | collaborator-held | collaborator-held |

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
   - `--aggregation additive` (Additive MIL): the slide logit is the
     attention-weighted sum of per-tile logits, giving exact signed tile
     contributions for heatmaps (`src/qupath/export_heatmaps.py`).
   - `--graph-layers L`: GraphSAGE-style layers that mix each tile with its 8 grid
     neighbours before attention (Patch-GCN-like spatial context; 0 = off).
3. **Evaluate / interpret**: PR curves and confusion matrices (`src/plotting/`),
   MSI-confound and multi-seed comparisons (`src/analysis/`), per-tile attention
   (`attention.h5`) and a blinded QuPath review package (`src/qupath/`).
4. **New cohorts** (`src/data/`): tissue-segmentation tuning per cohort,
   embedding (Ventana `.bif` read directly by `src/bif_reader.py`), and blinded
   prediction (`src/prediction/predict.py`, `predict.sbatch`).

## Layout

```
environment.yml              conda env (uni2h); torch built for CUDA 12.1 on Sherlock
metadata/                    cohort_labels.csv (labels + QC), ebv_status.csv, STAD_merged_table.csv
src/
  run_pipeline.py            segment + tile + UNI2-h embed (step 1)
  SlideReaderClass.py        fixed-µm/px tile reader (OpenSlide; .bif fallback)
  bif_reader.py              Ventana DP 200/600 .bif reader (tile overlaps ignored)
  TissueSegmentation.py      tissue mask + QC overlay
  encoders/UNI.py            UNI2-h wrapper (needs MahmoodLab `uni`, gated HF weights)
  prediction/                abmil.py, cohort.py (labels, splits, bags), train.py, predict.py,
                             export_attention.py, train.sbatch (tcga2chile|chile2tcga|mixed|all)
  plotting/                  PR curves, confusion matrices, lab-meeting slide figures
  analysis/                  msi_confound.py, compare_runs.py, cohort_table.py
  data/                      setup_encoder.sbatch, check_encoder.py, embed_cohort.sbatch,
                             tune_segmentation.py, convert_bif_to_svs.py/.sbatch, download_cptac.sbatch
  qupath/                    QuPath 0.7: blinded attention review + additive-MIL heatmap export
  metadata_prep/             scripts that built metadata/ (as originally run; old paths)
```
