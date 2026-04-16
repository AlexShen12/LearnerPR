#!/bin/bash
#SBATCH --job-name=gsv-download
#SBATCH --output=outputs/slurm/gsv_download_%j.out
#SBATCH --error=outputs/slurm/gsv_download_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --partition=general
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ─── Download and unpack GSV-Cities from Kaggle ─────────────────────
#
# Prerequisites:
#   1. A Kaggle account with API credentials at ~/.kaggle/kaggle.json
#      (chmod 600 ~/.kaggle/kaggle.json).
#   2. Accept the GSV-Cities dataset license at:
#      https://www.kaggle.com/datasets/amaralibey/gsv-cities
#   3. kaggle Python package installed (handled below if missing).
#
# Environment variables:
#   GSV_CITIES_ROOT  — where to unpack (default /users/a/l/alshen/LearnerPR/datasets/gsv-cities)
#   KAGGLE_CONFIG_DIR — directory containing kaggle.json (default ~/.kaggle)

set -euo pipefail

# Slurm runs a *copy* of this script under /var/spool/slurmd/.../slurm_script, so
# dirname(BASH_SOURCE) is not the repo.  Use SLURM_SUBMIT_DIR (cwd at sbatch time).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    _REPO="${SLURM_SUBMIT_DIR}"
else
    _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _REPO="$(cd "${_script_dir}/.." && pwd)"
fi
# shellcheck disable=SC1091
source "${_REPO}/scripts/slurm_longleaf_init.sh"

GSV_CITIES_ROOT="${GSV_CITIES_ROOT:-/users/a/l/alshen/LearnerPR/datasets/gsv-cities}"
KAGGLE_DATASET="amaralibey/gsv-cities"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/users/a/l/alshen/.kaggle}"

mkdir -p "$GSV_CITIES_ROOT" outputs/slurm

echo "=== LearnerPR: Download GSV-Cities ==="
echo "Destination:  $GSV_CITIES_ROOT"
echo "Kaggle creds: $KAGGLE_CONFIG_DIR/kaggle.json"
echo ""

# ── Ensure kaggle CLI is available ───────────────────────────────────
if ! python -c "import kaggle" &>/dev/null; then
    echo "Installing kaggle Python package..."
    pip install --quiet kaggle
fi

# ── Verify credentials ───────────────────────────────────────────────
if [ ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]; then
    echo "ERROR: ${KAGGLE_CONFIG_DIR}/kaggle.json not found."
    echo "  Create a Kaggle API token at https://www.kaggle.com/settings/account"
    echo "  and place it at ~/.kaggle/kaggle.json (chmod 600)."
    exit 1
fi
chmod 600 "${KAGGLE_CONFIG_DIR}/kaggle.json"

# ── Download ─────────────────────────────────────────────────────────
echo "Downloading $KAGGLE_DATASET ..."
KAGGLE_CONFIG_DIR="$KAGGLE_CONFIG_DIR" \
    kaggle datasets download \
        --dataset "$KAGGLE_DATASET" \
        --path "$GSV_CITIES_ROOT" \
        --unzip

# ── Verify layout ────────────────────────────────────────────────────
echo ""
echo "Verifying layout..."
IMAGES_DIR="$GSV_CITIES_ROOT/Images"
FRAMES_DIR="$GSV_CITIES_ROOT/Dataframes"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Expected $IMAGES_DIR after unzip — check the archive layout."
    exit 1
fi
if [ ! -d "$FRAMES_DIR" ]; then
    echo "WARNING: $FRAMES_DIR not found — place_id CSV metadata will be unavailable."
fi

N_CITIES=$(find "$IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_IMAGES=$(find "$IMAGES_DIR" -name "*.JPG" | wc -l)
echo "  Cities found:  $N_CITIES"
echo "  Images found:  $N_IMAGES"
echo "  Disk usage:    $(du -sh "$GSV_CITIES_ROOT" | cut -f1)"
echo ""
echo "=== Download complete ==="
echo "Set GSV_CITIES_ROOT=$GSV_CITIES_ROOT (or update configs/default.yaml)"
