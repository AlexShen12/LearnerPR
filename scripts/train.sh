#!/bin/bash
#SBATCH --job-name=learnerpr-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/train_%j.out
#SBATCH --error=outputs/slurm/train_%j.err

# ─── Train DINOv2-S + GeM student with RKD ─────────────────────────
# Pass AUGMENT=1 to enable data augmentation.
# Pass RESUME=/path/to/checkpoint.pt to resume training.

set -euo pipefail

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

source activate learnerpr 2>/dev/null || conda activate learnerpr

EXTRA_ARGS=""
if [ "$AUGMENT" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --augment"
fi
if [ -n "$RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
fi

python src/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $EXTRA_ARGS

echo "=== Training complete ==="
