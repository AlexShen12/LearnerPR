#!/bin/bash
#SBATCH --job-name=learnerpr-cache-gsv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/cache_gsv_%j.out
#SBATCH --error=outputs/slurm/cache_gsv_%j.err

# ─── Cache Qwen3-VL-8B teacher embeddings for GSV-Cities ────────────
# Run ONCE before training with GSV-Cities.  Supports resumption —
# re-submit if it times out; already-cached paths are skipped.
#
# Set GSV_CITIES to a comma-separated list to cache a subset of cities,
# e.g. GSV_CITIES="London,Paris,Boston" sbatch scripts/cache_teacher_gsv.sh

set -euo pipefail

GSV_CITIES_ROOT="${GSV_CITIES_ROOT:-/work/${USER}/datasets/gsv-cities}"
OUTPUT="${TEACHER_CACHE_GSV:-/work/${USER}/learnerpr/cache/teacher_embeddings_gsv.pt}"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE="${BATCH_SIZE:-16}"
GSV_CITIES="${GSV_CITIES:-}"   # empty = all cities

mkdir -p "$(dirname "$OUTPUT")" outputs/slurm

echo "=== LearnerPR: Cache Teacher Embeddings (GSV-Cities) ==="
echo "Model:          $MODEL"
echo "GSV-Cities root: $GSV_CITIES_ROOT"
echo "Output:         $OUTPUT"
echo "Batch size:     $BATCH_SIZE"
echo "Cities:         ${GSV_CITIES:-all}"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

source activate learnerpr 2>/dev/null || conda activate learnerpr

EXTRA_ARGS=""
if [ -n "$GSV_CITIES" ]; then
    EXTRA_ARGS="--gsv_cities $GSV_CITIES"
fi

python src/cache_teacher_embeddings.py \
    --dataset gsv_cities \
    --gsv_cities_root "$GSV_CITIES_ROOT" \
    --output "$OUTPUT" \
    --model_name "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS

echo "=== Caching complete ==="
