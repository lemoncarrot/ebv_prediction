#!/bin/bash
#SBATCH --job-name=bif2svs
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%A_%a.err

# Convert one Ventana .bif to Aperio-style .svs per array task (convert_bif_to_svs.py).
# Task i converts the i-th .bif in sorted order; finished slides are skipped, so
# re-submitting only redoes missing/failed ones.
#
#   N=$(ls $SRC/*.bif | wc -l)
#   sbatch --array=0 convert_bif_to_svs.sbatch                  # test one slide first
#   sbatch --array=1-$((N-1))%10 convert_bif_to_svs.sbatch      # the rest, 10 at a time
#   then summarize (merges per-slide reports):
#   $PY convert_bif_to_svs.py --summarize --out-dir $DEST
#
# SRC/DEST can be overridden: sbatch --export=ALL,SRC=...,DEST=... convert_bif_to_svs.sbatch

set -euo pipefail

PROJECT=/scratch/users/kchen24/EBV_prediction
SRC=${SRC:-$PROJECT/data/robertoherrera}
DEST=${DEST:-$PROJECT/data/robertoherrera_svs}
# group env with openslide + tifffile + imagecodecs (nothing installed for this)
PY=$GROUP_HOME/Virtual_envs/histology_conversion/bin/python

mapfile -t SLIDES < <(ls "$SRC"/*.bif | sort)
SLIDE=${SLIDES[$SLURM_ARRAY_TASK_ID]:?no slide for task $SLURM_ARRAY_TASK_ID (${#SLIDES[@]} slides)}

echo "$(date '+%F %T') task $SLURM_ARRAY_TASK_ID/${#SLIDES[@]} on $(hostname): $SLIDE"
"$PY" -u "$PROJECT/src/data/convert_bif_to_svs.py" --input "$SLIDE" --out-dir "$DEST"
echo "$(date '+%F %T') done"
