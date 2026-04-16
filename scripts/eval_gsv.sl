#!/bin/bash
#SBATCH --job-name=learnerpr-eval-gsv
#SBATCH --output=outputs/slurm/eval_gsv_%j.out
#SBATCH --error=outputs/slurm/eval_gsv_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ─── Evaluate trained student on GSV-Cities held-out cities ─────────
# CHECKPOINT:        path to best.pt or latest.pt
# HELD_OUT_CITIES:   space-separated city names (omit to auto-select via config)

set -euo pipefail

export LEARNERPR_REPO="${LEARNERPR_REPO:-/users/a/l/alshen/LearnerPR/LearnerPR}"
if [[ ! -f "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh" ]]; then
    echo "ERROR: ${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh not found." >&2
    echo "Set LEARNERPR_REPO to the directory that contains scripts/ and src/." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh"

CONFIG="${CONFIG:-configs/default.yaml}"
CHECKPOINT="${CHECKPOINT:-/users/a/l/alshen/LearnerPR/checkpoints/best.pt}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HELD_OUT_CITIES="${HELD_OUT_CITIES:-}"  # e.g. "Austin Bangkok BuenosAires"

mkdir -p outputs/slurm

echo "=== LearnerPR: Evaluate (GSV-Cities) ==="
echo "Checkpoint:        $CHECKPOINT"
echo "Held-out cities:   ${HELD_OUT_CITIES:-auto from config}"
echo "Batch size:        $BATCH_SIZE"
echo "GPU:               $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

EXTRA_ARGS=""
if [ -n "$HELD_OUT_CITIES" ]; then
    EXTRA_ARGS="--held_out_cities $HELD_OUT_CITIES"
fi

python src/eval_gsv.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS

echo "=== Evaluation complete ==="
