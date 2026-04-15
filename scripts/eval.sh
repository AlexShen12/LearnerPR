#!/bin/bash
#SBATCH --job-name=learnerpr-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/eval_%j.out
#SBATCH --error=outputs/slurm/eval_%j.err

# ─── Evaluate trained student on MSLS or GSV-Cities ─────────────────
# DATASET:           msls (default) or gsv_cities
# SPLIT:             val (default) or test  [MSLS only]
# CHECKPOINT:        path to best.pt or latest.pt
# HELD_OUT_CITIES:   space-separated GSV city names (omit = auto) [GSV only]

set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
CHECKPOINT="${CHECKPOINT:-/users/a/l/alshen/LearnerPR/checkpoints/best.pt}"
DATASET="${DATASET:-msls}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HELD_OUT_CITIES="${HELD_OUT_CITIES:-}"

mkdir -p outputs/slurm

echo "=== LearnerPR: Evaluate ($DATASET) ==="
echo "Checkpoint:  $CHECKPOINT"
echo "Batch size:  $BATCH_SIZE"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

source activate learnerpr 2>/dev/null || conda activate learnerpr

if [ "$DATASET" = "gsv_cities" ]; then
    EXTRA_ARGS=""
    if [ -n "$HELD_OUT_CITIES" ]; then
        EXTRA_ARGS="--held_out_cities $HELD_OUT_CITIES"
    fi
    python src/eval_gsv.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --batch_size "$BATCH_SIZE" \
        $EXTRA_ARGS
else
    python src/eval.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE"
fi

echo "=== Evaluation complete ==="
