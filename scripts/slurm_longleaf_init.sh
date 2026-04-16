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
# Override defaults from sbatch environment if needed:
#   LEARNERPR_REPO=/path/to/inner/LearnerPR   (directory with src/ and scripts/)
#   ANACONDA_MODULE=anaconda/2023.09
#   CONDA_ENV_NAME=learnerpr

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

# ── Conda environment ────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-learnerpr}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
