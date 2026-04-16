#!/bin/bash
# Sourced by LearnerPR *.sl batch scripts — not meant to be submitted alone.
#
# Parent *.sl jobs must resolve this file via SLURM_SUBMIT_DIR (Slurm copies the
# batch script to /var/spool/.../slurm_script; BASH_SOURCE there is not the repo).
#
# Longleaf (UNC): batch jobs do not load your login-shell conda init.  Use
# `module add anaconda` then `conda.sh` + `conda activate`.
#
# Override defaults from sbatch environment if needed:
#   ANACONDA_MODULE=anaconda/2023.09
#   CONDA_ENV_NAME=learnerpr

# ── Repo working directory ───────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}" || exit 1
else
    # Local `bash scripts/*.sl`: this file lives in <repo>/scripts/
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
