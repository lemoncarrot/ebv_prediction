#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.err

# Segment, tile and UNI2-h-embed every slide in a folder with the standard
# pipeline (src/run_pipeline.py: 256 px tiles at 0.5 um/px, i.e. 20x-equivalent,
# whatever the scan magnification), then report the scan magnification per slide.
# Resumable: slides whose .h5 exists are skipped.
#
#   sbatch embed_cohort.sbatch <slides-dir> <embeddings-out-dir>
#   sbatch embed_cohort.sbatch $P/data/robertoherrera_svs $P/data/robertoherrera_embeddings
#
# Run setup_encoder.sbatch once first (installs `uni`, checks embeddings match).

set -euo pipefail
SLIDES=${1:?usage: sbatch embed_cohort.sbatch <slides-dir> <out-dir>}
OUT=${2:?usage: sbatch embed_cohort.sbatch <slides-dir> <out-dir>}
PROJECT=/scratch/users/kchen24/EBV_prediction
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$GROUP_HOME/Virtual_envs/uni2h"
python -c "import uni" || { echo "ERROR: uni not installed; run setup_encoder.sbatch" >&2; exit 1; }

cd "$PROJECT/src"
python -u run_pipeline.py --slides "$SLIDES" --out-dir "$OUT" --workers 6

# magnification actually seen by the pipeline, per slide
python - "$OUT" <<'PY'
import glob, sys, h5py, collections
rows = []
for p in sorted(glob.glob(f"{sys.argv[1]}/*.h5")):
    with h5py.File(p, "r") as h:
        a = h.attrs
        rows.append((p.split("/")[-1][:-3], float(a["mpp_base"]), str(a["mpp_source"]),
                     float(a["mpp_achieved"]), h["features"].shape[0]))
with open(f"{sys.argv[1]}/magnification_report.csv", "w") as f:
    f.write("slide_id,mpp_base,scan_magnification,mpp_source,mpp_achieved,n_tiles\n")
    for s, mpp, src, ach, n in rows:
        f.write(f"{s},{mpp:.4f},{10 / mpp:.1f},{src},{ach:.4f},{n}\n")
mags = collections.Counter(f"{10 / r[1]:.0f}x" for r in rows)
print(f"{len(rows)} slides embedded; scan magnification {dict(mags)}; "
      f"all tiled at {sorted({round(r[3], 3) for r in rows})} um/px; "
      f"tiles median {sorted(r[4] for r in rows)[len(rows) // 2]}")
PY
