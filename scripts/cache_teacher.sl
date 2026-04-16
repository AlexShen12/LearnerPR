#!/bin/bash
#SBATCH --job-name=learnerpr-cache
#SBATCH --output=outputs/slurm/cache_%j.out
#SBATCH --error=outputs/slurm/cache_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ─── Cache Qwen3-VL-8B teacher embeddings for MSLS ─────────────────
# Run ONCE before training.  Supports resumption — re-submit if it
# times out and it will skip already-cached images.

set -euo pipefail

export LEARNERPR_REPO="${LEARNERPR_REPO:-/users/a/l/alshen/LearnerPR/LearnerPR}"
if [[ ! -f "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh" ]]; then
    echo "ERROR: ${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh not found." >&2
    echo "Set LEARNERPR_REPO to the directory that contains scripts/ and src/." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh"

MSLS_ROOT="${MSLS_ROOT:-/users/a/l/alshen/LearnerPR/datasets/msls}"
OUTPUT="${TEACHER_CACHE:-/users/a/l/alshen/LearnerPR/cache/teacher_embeddings_msls.pt}"
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

python src/cache_teacher_embeddings.py \
    --dataset msls \
    --msls_root "$MSLS_ROOT" \
    --output "$OUTPUT" \
    --model_name "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --split "$SPLIT" \
    --subset "$SUBSET"

echo "=== Caching complete ==="
