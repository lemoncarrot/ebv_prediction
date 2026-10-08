# Adding an external cohort (e.g. robertoherrera, Ventana .bif)

All steps are Slurm jobs; run from `src/data/` unless noted. `P=/scratch/users/kchen24/EBV_prediction`.

| step | command | output |
|---|---|---|
| 1. convert .bif -> .svs (one slide first, then the rest) | `sbatch --array=0 convert_bif_to_svs.sbatch`<br>`sbatch --array=1-68%10 convert_bif_to_svs.sbatch` | `data/robertoherrera_svs/<slide>.svs` + `<slide>.json` |
| 1b. conversion + magnification report | `$GROUP_HOME/Virtual_envs/histology_conversion/bin/python convert_bif_to_svs.py --summarize --out-dir $P/data/robertoherrera_svs` (in a job) | `conversion_report.csv` |
| 2. encoder setup (once per machine) | `sbatch setup_encoder.sbatch` | log must end in `MATCH` |
| 3. embed (UNI2-h, 256 px @ 0.5 um/px) | `sbatch embed_cohort.sbatch $P/data/robertoherrera_svs $P/data/robertoherrera_embeddings` | `<slide>.h5` + `magnification_report.csv` |
| 4. model on all TCGA+Chile (binary) | `cd ../prediction && sbatch --export=ALL,TAG=binary_s0 train.sbatch all --seed 0` | `results/all_binary_s0_<job>/final.pt` |
| 5. blinded predictions | `python predict.py --run-dir <run> --embeddings-dir $P/data/robertoherrera_embeddings --out-dir $P/results/robertoherrera_eval --cohort robertoherrera` (in a GPU or CPU job) | `predictions.csv`, `model.json`, `attention.h5` |
| 6. collaborator scores it | `python src/analysis/evaluate_predictions.py --predictions predictions.csv --labels labels.csv` | metrics with 95% CIs |

Notes
- Conversion re-encodes pixels as JPEG (quality 90); `pixel_mae` in the report
  quantifies the change. Original scan resolution and magnification are kept in
  the SVS header and recorded per slide.
- Whatever the scan magnification, tiles are read at 0.5 um/px (20x-equivalent),
  as for TCGA and Chile.
- `download_cptac.sbatch` is unrelated: CPTAC-STAD slides from TCIA.
