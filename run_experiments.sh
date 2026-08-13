#!/usr/bin/env bash
# Five MIL experiments on the pen-QC'd cohort. Run from the repository root on
# the GPU server:
#
#   conda activate UNI
#   bash run_experiments.sh              # both GPUs, detached, survives logout
#   bash run_experiments.sh --dry-run    # resolve cohorts and exit, no training
#
# Every run reads metadata/cohort_labels.csv, which now carries qc_exclude=1 on
# the 35 pen-contaminated TCGA slides (pen_review r000-r034). 07_train_mil.py
# drops them automatically; add --ignore-qc-exclude to a run to override.
#
# Expected cohorts after exclusion:
#   tcga   315 slides / 292 patients   EBV+ 25   EBV- 290
#   chile  130 slides / 130 patients   EBV+ 22   EBV- 108
#
# Experiments are split across two GPUs and detached with setsid+nohup, so they
# survive the launching shell. Each GPU works through its queue sequentially.

set -uo pipefail

cd "$(dirname "$0")"

CHILE_FEATURES=02_uni2_embeddings
TCGA_FEATURES=02_uni2_embeddings_tcga
OUT=04_mil
LOGS=logs

# Shared settings, identical across runs so the five results stay comparable.
# Only cohort composition and the loss weighting differ between experiments.
COMMON=(--epochs 40 --patience 10 --batch-size 4 --max-patches 4096
        --lr 2e-4 --weight-decay 1e-4 --folds 5 --seed 0
        --workers 8 --amp --output-dir "$OUT" --overwrite)

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

run() {
    # run <gpu> <run-name> <args...>
    local gpu="$1" name="$2"; shift 2
    echo "[gpu $gpu] === $name START $(date '+%F %T') ==="
    CUDA_VISIBLE_DEVICES="$gpu" python src/07_train_mil.py \
        --run-name "$name" "${COMMON[@]}" ${DRY:+$DRY} "$@" \
        > "$LOGS/${name}.log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[gpu $gpu] === $name DONE  $(date '+%F %T') ==="
    else
        echo "[gpu $gpu] === $name FAILED (exit $rc) -- see $LOGS/${name}.log ==="
    fi
    return 0   # one failed experiment must not abort the rest of the queue
}

# GPU 0: the two cross-cohort transfer runs, then combined.
#
# tcga_to_chile vs tcga_to_chile_weighted is the requested weighting contrast.
# 07_train_mil.py defaults --pos-weight to neg/pos (auto), so the *unweighted*
# arm is the one needing an explicit flag: --pos-weight 1.0.
queue_gpu0() {
    run 0 tcga_to_chile \
        --features-dir "$TCGA_FEATURES" "$CHILE_FEATURES" \
        --cohort tcga --external-cohort chile \
        --pos-weight 1.0

    run 0 tcga_to_chile_weighted \
        --features-dir "$TCGA_FEATURES" "$CHILE_FEATURES" \
        --cohort tcga --external-cohort chile

    run 0 combined_5fold \
        --features-dir "$TCGA_FEATURES" "$CHILE_FEATURES" \
        --cohort tcga chile
}

# GPU 1: the two within-cohort cross-validations.
queue_gpu1() {
    run 1 tcga_5fold \
        --features-dir "$TCGA_FEATURES" \
        --cohort tcga

    run 1 chile_5fold \
        --features-dir "$CHILE_FEATURES" \
        --cohort chile
}

RUNS=(tcga_5fold chile_5fold tcga_to_chile tcga_to_chile_weighted combined_5fold)

# Re-invocation entry point for the detached queues (see the launch block below).
if [[ "${1:-}" == "--queue" ]]; then
    "queue_${2}"
    echo "[${2}] all experiments finished $(date '+%F %T')"
    exit 0
fi

if [[ "${1:-}" == "--summary" ]]; then
    echo "=== status ==="
    for r in "${RUNS[@]}"; do
        if [[ -f "$OUT/$r/metrics.json" ]]; then echo "  $r: done"
        elif pgrep -af "run-name $r" > /dev/null 2>&1; then echo "  $r: RUNNING"
        elif [[ -f "$LOGS/$r.log" ]]; then echo "  $r: incomplete -- tail $LOGS/$r.log"
        else echo "  $r: not started"; fi
    done
    echo
    python - "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path

def dig(m, *paths):
    for p in paths:
        cur = m
        for part in p.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    return None

rows = []
for name in sys.argv[1:]:
    path = Path("04_mil") / name / "metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    rows.append((name,
                 dig(m, "oof.slide.auroc", "val.slide.auroc"),
                 dig(m, "oof.slide.auprc", "val.slide.auprc"),
                 dig(m, "oof.patient.auroc", "val.patient.auroc"),
                 dig(m, "external.slide.auroc"),
                 dig(m, "external.slide.auprc")))

if not rows:
    sys.exit("no metrics.json found yet under 04_mil/")

hdr = (f"{'run':<26}{'slide AUROC':>12}{'slide AUPRC':>12}"
       f"{'pat AUROC':>11}{'ext AUROC':>11}{'ext AUPRC':>11}")
print("=== results ===")
print(hdr)
print("-" * len(hdr))
f = lambda v: f"{v:>11.3f}" if v is not None else f"{'-':>11}"
for name, a, p, pa, ea, ep in rows:
    print(f"{name:<26}{f(a)} {f(p)}{f(pa)}{f(ea)}{f(ep)}")
print("\nFull metrics: 04_mil/<run>/metrics.json")
PY
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
mkdir -p "$LOGS"

for d in "$CHILE_FEATURES" "$TCGA_FEATURES"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: missing features directory '$d'" >&2
        exit 1
    fi
done
if ! head -1 metadata/cohort_labels.csv | grep -q "qc_exclude"; then
    echo "ERROR: metadata/cohort_labels.csv has no qc_exclude column." >&2
    echo "       Run: python src/11_annotate_qc_exclusions.py" >&2
    exit 1
fi
if ! python -c "import torch, h5py, pandas, sklearn" 2>/dev/null; then
    echo "ERROR: torch/h5py/pandas/sklearn not importable. Did you 'conda activate UNI'?" >&2
    exit 1
fi
python - <<'PY' || exit 1
import sys, torch
n = torch.cuda.device_count()
print(f"visible GPUs: {n}")
if n < 1:
    sys.exit("ERROR: no CUDA device visible")
if n < 2:
    print("WARNING: fewer than 2 GPUs; both queues will share CUDA device 0.\n"
          "         Edit the queue_gpu1 CUDA_VISIBLE_DEVICES if that is not wanted.")
PY

if [[ -n "$DRY" ]]; then
    # Serial and in the foreground: a dry run takes seconds and the output is
    # the whole point, so it should not be buried in a detached log.
    echo "DRY RUN: resolving cohorts only, no training"
    queue_gpu0
    queue_gpu1
    echo
    echo "Cohort resolution above. Re-run without --dry-run to train."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch: each queue re-invokes this script, so no function/array serialisation
# is needed and the detached process reads the same definitions.
# ---------------------------------------------------------------------------
setsid nohup bash "$0" --queue gpu0 > "$LOGS/queue_gpu0.log" 2>&1 < /dev/null &
setsid nohup bash "$0" --queue gpu1 > "$LOGS/queue_gpu1.log" 2>&1 < /dev/null &

sleep 2
cat <<EOF

Launched 5 experiments, detached, across 2 GPUs.

  gpu 0:  tcga_to_chile -> tcga_to_chile_weighted -> combined_5fold
  gpu 1:  tcga_5fold    -> chile_5fold

Watch progress:
  tail -f $LOGS/queue_gpu0.log $LOGS/queue_gpu1.log
  tail -f $LOGS/tcga_5fold.log

Check what is still running:
  pgrep -af 'src/07_train_mil'

Collect results when finished:
  bash run_experiments.sh --summary

Results land in $OUT/<run-name>/ (metrics.json, oof_predictions.csv,
external_predictions.csv, folds.csv, checkpoints/).
EOF
