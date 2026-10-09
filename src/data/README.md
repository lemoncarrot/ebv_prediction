# Adding an external cohort (e.g. robertoherrera, Ventana .bif)

All steps are Slurm jobs; run from `src/data/` unless noted. `P=/scratch/users/kchen24/EBV_prediction`.

| step | command | output |
|---|---|---|
| 1. encoder setup (once per machine) | `sbatch setup_encoder.sbatch` | log must end in `MATCH` |
| 2. embed (UNI2-h, 256 px @ 0.5 um/px), reading .bif directly | `sbatch embed_cohort.sbatch $P/data/robertoherrera $P/data/robertoherrera_embeddings` | `<slide>.h5` + `magnification_report.csv` |
| 3. model on all TCGA+Chile (binary) | `cd ../prediction && sbatch --export=ALL,TAG=binary_s0 train.sbatch all --seed 0` | `results/all_binary_s0_<job>/final.pt` |
| 4. blinded predictions | `python predict.py --run-dir <run> --embeddings-dir $P/data/robertoherrera_embeddings --out-dir $P/results/robertoherrera_eval --cohort robertoherrera` (in a job) | `predictions.csv`, `model.json`, `attention.h5` |

Ventana DP 200/600 .bif
- OpenSlide cannot open these (LEFT tile joints). `src/bif_reader.py` reads them
  instead, laying scanner frames on their grid and **ignoring the joint
  overlaps**: on the robertoherrera DP 600 slides ~25% of side-by-side joints
  overlap by exactly 24 px (11 um), so those seams keep a duplicated strip and
  rows drift by up to ~1%. Embeddings record `reader` and `bif_overlaps_ignored`;
  treat predictions from them as preliminary.
- `convert_bif_to_svs.py` (BIF -> Aperio SVS) refuses slides with overlapping
  joints until proper stitching is implemented; it is not needed for embedding.

Notes
- Whatever the scan magnification, tiles are read at 0.5 um/px (20x-equivalent),
  as for TCGA and Chile. robertoherrera is scanned at 20x, 0.465 um/px.
- `download_cptac.sbatch` is unrelated: CPTAC-STAD slides from TCIA.
