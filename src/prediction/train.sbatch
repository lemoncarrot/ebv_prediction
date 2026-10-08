#!/bin/bash
#SBATCH --job-name=ebv-abmil
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.err

# 5-fold patient-grouped CV on the training slides, then a final model trained
# on all of them and scored once on the external set, then the plots.
#
#   sbatch train.sbatch tcga2chile      # train TCGA, external = Chile
#   sbatch train.sbatch chile2tcga      # train Chile, external = TCGA
#   sbatch train.sbatch mixed           # TCGA+Chile, 20% patient-grouped test split held out
#   sbatch train.sbatch all             # TCGA+Chile, all slides; final model for new cohorts (predict.py)
#   sbatch train.sbatch mixed --epochs 60 --lr 3e-4   # extra args pass through to train.py
#   sbatch --export=ALL,TAG=3class_s1 train.sbatch tcga2chile --label-scheme 3class --seed 1

set -e

SCHEME=${1:?usage: sbatch train.sbatch tcga2chile|chile2tcga|mixed|all [train.py args]}
shift
case "$SCHEME" in
    tcga2chile) SPLIT=(--external-cohort chile) ;;
    chile2tcga) SPLIT=(--external-cohort tcga) ;;
    mixed)      SPLIT=(--holdout-frac 0.2) ;;
    all)        SPLIT=(--no-external) ;;   # TCGA+Chile, no test set: score new cohorts with predict.py
    *) echo "unknown scheme: $SCHEME (tcga2chile|chile2tcga|mixed|all)" >&2; exit 1 ;;
esac

PROJECT=/scratch/users/kchen24/EBV_prediction
OUT_DIR=$PROJECT/results/${SCHEME}${TAG:+_$TAG}_${SLURM_JOB_ID}   # TAG: optional, via --export

# uni2h mirrors src/environment.yml, with torch built against cu121 for Sherlock.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$GROUP_HOME/Virtual_envs/uni2h"

cd "$PROJECT/src/prediction"

python -u train.py \
    --embeddings-dir "$PROJECT/data/tcga_embeddings" "$PROJECT/data/chile_embeddings" \
    --labels "$PROJECT/metadata/cohort_labels.csv" \
    --out-dir "$OUT_DIR" \
    "${SPLIT[@]}" \
    --folds 5 \
    --epochs 40 \
    --seed 0 \
    --workers 4 \
    --balance-loss \
    "$@"

[ "$SCHEME" = all ] && exit 0   # nothing external to plot
python -u ../plotting/auc_curves.py --run-dir "$OUT_DIR"
python -u ../plotting/confusion_matrix.py --run-dir "$OUT_DIR"
