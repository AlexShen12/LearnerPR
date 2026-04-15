#!/bin/bash
#SBATCH --job-name=learnerpr-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/cache_%j.out
#SBATCH --error=outputs/slurm/cache_%j.err

# ─── Cache Qwen3-VL-8B teacher embeddings for MSLS ─────────────────
# Run ONCE before training.  Supports resumption — re-submit if it
# times out and it will skip already-cached images.

set -euo pipefail

MSLS_ROOT="${MSLS_ROOT:-/work/${USER}/datasets/msls}"
OUTPUT="${TEACHER_CACHE:-/work/${USER}/learnerpr/cache/teacher_embeddings.pt}"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE="${BATCH_SIZE:-16}"
SPLIT="${SPLIT:-train}"
SUBSET="${SUBSET:-both}"

mkdir -p "$(dirname "$OUTPUT")" outputs/slurm

echo "=== LearnerPR: Cache Teacher Embeddings ==="
echo "Model:      $MODEL"
echo "MSLS root:  $MSLS_ROOT"
echo "Output:     $OUTPUT"
echo "Split:      $SPLIT"
echo "Subset:     $SUBSET"
echo "Batch size: $BATCH_SIZE"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

source activate learnerpr 2>/dev/null || conda activate learnerpr

python src/cache_teacher_embeddings.py \
    --dataset msls \
    --msls_root "$MSLS_ROOT" \
    --output "$OUTPUT" \
    --model_name "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --split "$SPLIT" \
    --subset "$SUBSET"

echo "=== Caching complete ==="
