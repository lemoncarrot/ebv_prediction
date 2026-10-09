#!/bin/bash
#SBATCH --job-name=ebv-predict
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.err

# Score an unlabelled cohort with a trained run's final model (predict.py).
#
#   sbatch predict.sbatch <run-dir> <embeddings-dir> <out-dir> <cohort-name>
#   sbatch --dependency=afterok:<embed-job> predict.sbatch \
#       $P/results/all_binary_s0_46941346 $P/data/robertoherrera_embeddings \
#       $P/results/robertoherrera_eval robertoherrera

set -euo pipefail
RUN=${1:?usage: predict.sbatch <run-dir> <embeddings-dir> <out-dir> <cohort>}
EMB=${2:?}; OUT=${3:?}; COHORT=${4:?}
PROJECT=/scratch/users/kchen24/EBV_prediction
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$GROUP_HOME/Virtual_envs/uni2h"
cd "$PROJECT/src/prediction"
python -u predict.py --run-dir "$RUN" --embeddings-dir "$EMB" --out-dir "$OUT" --cohort "$COHORT"
