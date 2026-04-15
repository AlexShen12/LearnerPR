#!/bin/bash
# ─── LearnerPR: Longleaf environment bootstrap ─────────────────────
# Run once on a login node to create the conda env.
#
#   bash setup.sh
#
# After setup, all SLURM scripts activate this env automatically.

set -euo pipefail

ENV_NAME="learnerpr"
PYTHON_VERSION="3.11"

echo "=== LearnerPR environment setup ==="

# Load modules available on Longleaf
module load cuda/12.2 2>/dev/null || true
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true

# Create conda env if it doesn't already exist
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
    echo "Conda env '${ENV_NAME}' already exists — skipping creation."
fi

source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

echo "Installing PyTorch (CUDA 12.1 wheels)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing flash-attn (needs CUDA toolkit at build time)..."
pip install flash-attn --no-build-isolation

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
