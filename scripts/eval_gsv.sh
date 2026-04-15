#!/bin/bash
#SBATCH --job-name=learnerpr-eval-gsv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/eval_gsv_%j.out
#SBATCH --error=outputs/slurm/eval_gsv_%j.err

# ─── Evaluate trained student on GSV-Cities held-out cities ─────────
# CHECKPOINT:        path to best.pt or latest.pt
# HELD_OUT_CITIES:   space-separated city names (omit to auto-select via config)

set -euo pipefail

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

source activate learnerpr 2>/dev/null || conda activate learnerpr

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
