#!/bin/bash
#SBATCH -J ebv-auc
#SBATCH -p normal
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -o logs/ebv-auc-%j.out
#SBATCH -e logs/ebv-auc-%j.err

# Usage (from the repo root):  sbatch src/plotting/plot_auc.sbatch [RUN_DIR]
RUN_DIR=${1:?usage: sbatch $0 RUN_DIR}
PYTHON=/home/groups/hanleeji/Virtual_envs/uni2h/bin/python

$PYTHON src/plotting/auc_curves.py --run-dir "$RUN_DIR"
