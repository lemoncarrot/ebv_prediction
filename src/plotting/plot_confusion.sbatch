#!/bin/bash
#SBATCH -J ebv-cm
#SBATCH -p normal
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -o logs/ebv-cm-%j.out
#SBATCH -e logs/ebv-cm-%j.err

# Usage (from the repo root):  sbatch src/plotting/plot_confusion.sbatch [RUN_DIR]
RUN_DIR=${1:?usage: sbatch $0 RUN_DIR}
PYTHON=/home/groups/hanleeji/Virtual_envs/uni2h/bin/python

$PYTHON src/plotting/confusion_matrix.py --run-dir "$RUN_DIR"
