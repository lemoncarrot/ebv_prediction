#!/bin/bash
#SBATCH --job-name=uni-setup
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.err

# One-time: make the uni2h env able to run run_pipeline.py on Sherlock, then
# prove it reproduces the existing (off-Sherlock) embeddings.
#   1. install MahmoodLab's `uni` package (pinned as uni==0.1.0 in
#      src/environment.yml, missing from the Sherlock env) with --no-deps, so
#      torch/timm/etc. versions are left exactly as they are;
#   2. download UNI2-h (gated on Hugging Face: needs the token in $HF_HOME and
#      approved access to MahmoodLab/UNI2-h);
#   3. check_encoder.py: re-embed tiles of one Chile slide at its stored
#      coordinates and compare with the stored embedding (must be ~identical).
#
#   sbatch setup_encoder.sbatch

set -euo pipefail
PROJECT=/scratch/users/kchen24/EBV_prediction
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$GROUP_HOME/Virtual_envs/uni2h"
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

if python -c "import uni" 2>/dev/null; then
    echo "uni already installed"
else
    pip install --no-deps "uni @ git+https://github.com/mahmoodlab/UNI.git"
fi
python -c "import uni, timm, torch; print('uni ok | timm', timm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

SLIDE_ID=140-A4_150818
python -u "$PROJECT/src/data/check_encoder.py" \
    --slide "$PROJECT/data/original_chile/$SLIDE_ID.svs" \
    --embedding "$PROJECT/data/chile_embeddings/$SLIDE_ID.h5"
