#!/bin/bash
#SBATCH --job-name=learnerpr-train
#SBATCH --output=outputs/slurm/train_%j.out
#SBATCH --error=outputs/slurm/train_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ─── Train DINOv2-S + GeM student with RKD ─────────────────────────
# Pass AUGMENT=1 to enable data augmentation.
# Pass RESUME=/path/to/checkpoint.pt to resume training.

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
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
AUGMENT="${AUGMENT:-0}"
RESUME="${RESUME:-}"

mkdir -p outputs/slurm

echo "=== LearnerPR: Train Student ==="
echo "Config:     $CONFIG"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LR"
echo "Augment:    $AUGMENT"
echo "Resume:     ${RESUME:-none}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

EXTRA_ARGS=""
if [ "$AUGMENT" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --augment"
fi
if [ -n "$RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
fi

"${PYTHON}" src/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $EXTRA_ARGS

echo "=== Training complete ==="
