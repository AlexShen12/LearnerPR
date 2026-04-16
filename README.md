# LearnerPR

VPR student: **DINOv2-Small + GeM**, trained with RKD from **Qwen3-VL-8B-Instruct** (see `PLAN.md`).

- **Course submission:** root [`model.py`](model.py) implements the COMP 560 `StudentModel` API (`device`, `encode`, `embedding_dim`) for the provided `evaluate.py`. Place trained weights as `submission_weights.pt` or set `LEARNERPR_WEIGHTS`.
- **Training / MSLS:** `src/train.py`, Longleaf scripts under `scripts/`.
- **Training / GSV-Cities:** see below.
- **COMP 560 alignment (metrics, datasets, integrity):** `PLAN.md` §11.

## Longleaf (UNC) Slurm

Batch jobs use `scripts/slurm_longleaf_init.sh` (sourced from every `*.sl` script): `module purge`, `module add anaconda/2024.02`, `source "$(conda info --base)/etc/profile.d/conda.sh"`, then `conda activate learnerpr`.

**Important:** Run `sbatch` from the inner repo directory that contains both `scripts/` and `src/` (e.g. `cd …/LearnerPR/LearnerPR` then `sbatch scripts/download_gsv_cities.sl`). Slurm sets `SLURM_SUBMIT_DIR` to that cwd; the batch script itself is copied to `/var/spool/slurmd/…`, so paths must not rely on the script file’s location.

Optional environment overrides: `ANACONDA_MODULE` (default `anaconda/2024.02`), `CONDA_ENV_NAME` (default `learnerpr`).

## GSV-Cities setup

GSV-Cities (~530k images, ~62k places across 40+ cities) can be added as a second training source alongside MSLS.

**Prerequisites:**

1. A [Kaggle account](https://www.kaggle.com) with an API token saved to `~/.kaggle/kaggle.json` (chmod 600). Accept the dataset license at <https://www.kaggle.com/datasets/amaralibey/gsv-cities>.
2. ~60 GB free on your work filesystem.

**Download (once):**

```bash
sbatch scripts/download_gsv_cities.sl
```

Expected layout after unzip:

```
gsv-cities/
    Images/<City>/*.JPG        # ~530k images
    Dataframes/<City>.csv      # place_id metadata per city
```

**Cache teacher embeddings for GSV-Cities (once):**

```bash
sbatch scripts/cache_teacher_gsv.sl
```

**Joint training (MSLS + GSV-Cities):**

Edit `configs/default.yaml`:
```yaml
training:
  train_datasets: [msls, gsv_cities]
```

Then run training as usual:
```bash
sbatch scripts/train.sl
```

**GSV-Cities eval (held-out cities):**

```bash
sbatch scripts/eval_gsv.sl
```