#!/bin/bash
#SBATCH --job-name=learnerpr-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/eval_%j.out
#SBATCH --error=outputs/slurm/eval_%j.err

# ─── Evaluate trained student on MSLS val or test ───────────────────
# SPLIT: val (default) or test
# CHECKPOINT: path to best.pt or latest.pt

set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
CHECKPOINT="${CHECKPOINT:-/work/${USER}/learnerpr/checkpoints/best.pt}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p outputs/slurm

echo "=== LearnerPR: Evaluate ==="
echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

source activate learnerpr 2>/dev/null || conda activate learnerpr

python src/eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE"

echo "=== Evaluation complete ==="
