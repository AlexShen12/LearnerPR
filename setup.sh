#!/bin/bash
# ─── LearnerPR: Longleaf environment bootstrap ─────────────────────
set -euo pipefail

# 1. Define Paths
ENV_PATH="/work/users/a/l/alshen/conda_envs/learnerpr"
PYTHON_VERSION="3.11"
ENV_PYTHON="$ENV_PATH/bin/python"

# 2. Fix the "Invalid cross-device link" and Quota issues (CRITICAL)
# Put these at the top so ALL following commands use the /work space
mkdir -p /work/users/a/l/alshen/pip_cache /work/users/a/l/alshen/tmp
export PIP_CACHE_DIR="/work/users/a/l/alshen/pip_cache"
export TMPDIR="/work/users/a/l/alshen/tmp"

echo "=== LearnerPR environment setup ==="

# 3. Load Modules
module load cuda/12.2 2>/dev/null || true
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true

# 4. Create Conda Env (if missing)
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda env at: ${ENV_PATH}"
    mkdir -p "$(dirname "$ENV_PATH")"
    conda create -y -p "$ENV_PATH" python="$PYTHON_VERSION"
else
    echo "Conda env at '${ENV_PATH}' already exists — skipping creation."
fi

# 5. Activate & Force Environment Localness
source activate "$ENV_PATH" 2>/dev/null || conda activate "$ENV_PATH"
conda install -y -p "$ENV_PATH" pip

# 6. Set CUDA variables for flash-attn (ensure nvcc is found)
# $CUDA_INSTALL_PATH is set by 'module load cuda' on Longleaf
export CUDA_HOME=${CUDA_INSTALL_PATH:-/nas/longleaf/rhel9/apps/cuda/12.2}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Using Python: $(which python)"
echo "Using CUDA_HOME: $CUDA_HOME"

# 7. Install Packages
echo "Installing PyTorch..."
"$ENV_PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing build dependencies (psutil, ninja, packaging)..."
"$ENV_PYTHON" -m pip install psutil ninja packaging

echo "Installing flash-attn via direct wheel..."
# This specific URL matches Torch 2.5.1 + CUDA 12.1 + Python 3.11
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# Try the direct wheel first; if that fails, try a standard install
"$ENV_PYTHON" -m pip install "$WHEEL_URL" || "$ENV_PYTHON" -m pip install flash-attn --no-build-isolation

echo "Installing remaining dependencies..."
if [ -f "requirements.txt" ]; then
    "$ENV_PYTHON" -m pip install -r requirements.txt
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate $ENV_PATH"
