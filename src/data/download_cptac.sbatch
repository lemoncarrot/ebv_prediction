#!/bin/bash
#SBATCH --job-name=cptac-dl
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --output=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.out
#SBATCH --error=/scratch/users/kchen24/EBV_prediction/logs/%x-%j.err

# (normal partition: the service partition's nodes are often reserved)
# Download the CPTAC-STAD whole-slide images (TCIA, CC BY 4.0; 703 .svs, ~1.1 TB)
# with IBM ascli from TCIA's public Faspex link. Resumable: files already
# complete are skipped and partial files resume, so if the job hits its time
# limit or the connection drops, just submit it again.
#
#   sbatch download_cptac.sbatch
#
# Collection: https://www.cancerimagingarchive.net/collection/cptac-stad/
# Labels/clinical data come separately from LinkedOmics (metadata/cptac/).

set -euo pipefail

PROJECT=/scratch/users/kchen24/EBV_prediction
DEST=$PROJECT/data/original_cptac
LINK='https://faspex.cancerimagingarchive.net/?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEyNjkiLCJwYXNzY29kZSI6ImUyYmUzMzBiYmM1ZTU0NWI1MTdiMGE2MzM3MGQ5YzBkNjFmZGE5MjAiLCJwYWNrYWdlX2lkIjoiMTI2OSIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=='   # as published on the TCIA page

mkdir -p "$DEST"
# ascli >= 4.27.4 (public-link receive fixed), installed with: gem install --install-dir $GROUP_HOME/software/ascli aspera-cli
# ascp transfer engine from the Sherlock module, linked once as ~/.aspera/sdk/ascp
module load system aspera-cli/4.24.1 devel ruby/3.1.2
export GEM_HOME=$GROUP_HOME/software/ascli GEM_PATH=$GROUP_HOME/software/ascli
ascli() { "$GEM_HOME/bin/ascli" "$@"; }
[ -e ~/.aspera/sdk/ascp ] || { mkdir -p ~/.aspera/sdk; ln -s "$(command -v ascp)" ~/.aspera/sdk/ascp; }
echo "$(date '+%F %T') start on $(hostname): ascli $(ascli --version) -> $DEST"
echo "already present: $(find "$DEST" -name '*.svs' | wc -l) .svs, $(du -sh "$DEST" | cut -f1)"

# resume_policy sparse_csum: skip files whose size + sparse checksum match, resume the rest
ascli faspex5 packages receive \
    --url="$LINK" \
    --to-folder="$DEST" \
    --progress-bar=no \
    --ts=@json:'{"resume_policy":"sparse_csum"}'

echo "$(date '+%F %T') done: $(find "$DEST" -name '*.svs' | wc -l) .svs, $(du -sh "$DEST" | cut -f1)"
