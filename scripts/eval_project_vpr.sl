#!/bin/bash
#SBATCH --job-name=learnerpr-eval-vpr
#SBATCH --output=outputs/slurm/eval_vpr_%j.out
#SBATCH --error=outputs/slurm/eval_vpr_%j.err
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

# ─── Evaluate LearnerPR weights on project-vpr dataset_a / dataset_b ──
#
# Key env vars (all optional — defaults shown):
#
#   CHECKPOINT       path to LearnerPR .pt file  (default: checkpoints/best.pt)
#   PROJECT_VPR_ROOT path to the project-vpr dir (default: sibling of LEARNERPR_REPO)
#   DATASETS_ROOT    path containing dataset_a/   (default: PROJECT_VPR_ROOT/datasets)
#   DATASETS         space-separated list         (default: "dataset_a dataset_b")
#   BATCH_SIZE       inference batch size          (default: 128)
#   SAVE_PREDICTIONS dir to write CSV rankings     (default: off)
#
# Example — evaluate both datasets, save CSVs:
#   DATASETS="dataset_a dataset_b" \
#   SAVE_PREDICTIONS="/users/a/l/alshen/LearnerPR/predictions" \
#   sbatch scripts/eval_project_vpr.sl
#
# Example — evaluate a specific checkpoint:
#   CHECKPOINT="/users/a/l/alshen/LearnerPR/checkpoints/latest.pt" \
#   sbatch scripts/eval_project_vpr.sl

set -euo pipefail

export LEARNERPR_REPO="${LEARNERPR_REPO:-/users/a/l/alshen/LearnerPR/LearnerPR}"
if [[ ! -f "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh" ]]; then
    echo "ERROR: ${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh not found." >&2
    echo "Set LEARNERPR_REPO to the directory that contains scripts/ and src/." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${LEARNERPR_REPO}/scripts/slurm_longleaf_init.sh"

CHECKPOINT="${CHECKPOINT:-/users/a/l/alshen/LearnerPR/checkpoints/best.pt}"
PROJECT_VPR_ROOT="${PROJECT_VPR_ROOT:-/users/a/l/alshen/LearnerPR/../project-vpr}"
DATASETS_ROOT="${DATASETS_ROOT:-${PROJECT_VPR_ROOT}/datasets}"
DATASETS="${DATASETS:-dataset_a dataset_b}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-}"

mkdir -p outputs/slurm

echo "=== LearnerPR: Evaluate on project-vpr ==="
echo "Checkpoint:    $CHECKPOINT"
echo "project-vpr:   $PROJECT_VPR_ROOT"
echo "Datasets root: $DATASETS_ROOT"
echo "Datasets:      $DATASETS"
echo "Batch size:    $BATCH_SIZE"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

EXTRA_ARGS=""
if [ -n "$SAVE_PREDICTIONS" ]; then
    mkdir -p "$SAVE_PREDICTIONS"
    EXTRA_ARGS="$EXTRA_ARGS --save_predictions $SAVE_PREDICTIONS"
fi

# shellcheck disable=SC2086
"${PYTHON}" scripts/eval_project_vpr.py \
    --weights         "$CHECKPOINT" \
    --project_vpr_root "$PROJECT_VPR_ROOT" \
    --datasets_root   "$DATASETS_ROOT" \
    --datasets        $DATASETS \
    --batch_size      "$BATCH_SIZE" \
    --device          cuda \
    $EXTRA_ARGS

echo "=== Evaluation complete ==="
