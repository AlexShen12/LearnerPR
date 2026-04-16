#!/bin/bash
# Sourced by LearnerPR *.sl batch scripts — not meant to be submitted alone.
#
# Parent *.sl jobs set LEARNERPR_REPO to the inner checkout (contains src/,
# configs/).  Slurm copies the batch script to /var/spool/.../slurm_script, so
# SLURM_SUBMIT_DIR alone may point at a parent folder if you sbatch from there.
#
# Longleaf (UNC): batch jobs do not load your login-shell conda init.  Use
# `module add anaconda` then `conda.sh` + `conda activate`.
#
# Exports for downstream commands in the same shell:
#   PYTHON  — absolute path to the conda env interpreter (use "${PYTHON}" in *.sl)
#
# Override defaults from sbatch environment if needed:
#   LEARNERPR_REPO=/path/to/inner/LearnerPR   (directory with src/ and scripts/)
#   ANACONDA_MODULE=anaconda/2023.09
#   CONDA_ENV_NAME=learnerpr
#   LEARNERPR_SKIP_TORCH_CHECK=1   (set before sourcing for jobs that do not need torch, e.g. GSV download)

# ── Repo working directory ───────────────────────────────────────────
if [[ -n "${LEARNERPR_REPO:-}" ]] && [[ -d "${LEARNERPR_REPO}" ]]; then
    cd "${LEARNERPR_REPO}" || exit 1
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}" || exit 1
else
    _init_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${_init_here}/.." || exit 1
fi

# ── Site modules (Longleaf uses `module add`) ───────────────────────
if command -v module >/dev/null 2>&1; then
    module purge
    module add "${ANACONDA_MODULE:-anaconda/2024.02}"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Load the Anaconda module (ANACONDA_MODULE) or use a login node to create the env." >&2
    exit 1
fi

# ── Conda environment ────────────────────────────────────────────────
# shellcheck disable=SC1091
_conda_base="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda info --base failed." >&2
    exit 1
}
source "${_conda_base}/etc/profile.d/conda.sh"

_env_name="${CONDA_ENV_NAME:-learnerpr}"
if ! conda activate "${_env_name}"; then
    echo "ERROR: conda activate ${_env_name} failed. Create the env (see setup.sh) or set CONDA_ENV_NAME." >&2
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: CONDA_PREFIX is empty after conda activate." >&2
    exit 1
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${CONDA_PREFIX}/bin/python3" ]]; then
    export PYTHON="${CONDA_PREFIX}/bin/python3"
else
    echo "ERROR: no python under ${CONDA_PREFIX}/bin" >&2
    exit 1
fi

echo "LearnerPR env: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-?}  PYTHON=${PYTHON}"
"${PYTHON}" -V

if [[ "${LEARNERPR_SKIP_TORCH_CHECK:-0}" != "1" ]]; then
    if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
        echo "ERROR: torch is not installed in env \"${_env_name}\" (${PYTHON})." >&2
        echo "  conda activate ${_env_name} && pip install -r requirements.txt" >&2
        exit 1
    fi
fi
