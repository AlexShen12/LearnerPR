# LearnerPR

VPR student: **DINOv2-Small + GeM**, trained with RKD from **Qwen3-VL-8B-Instruct** (see `PLAN.md`).

- **Course submission:** root [`model.py`](model.py) implements the COMP 560 `StudentModel` API (`device`, `encode`, `embedding_dim`) for the provided `evaluate.py`. Place trained weights as `submission_weights.pt` or set `LEARNERPR_WEIGHTS`.
- **Training / MSLS:** `src/train.py`, Longleaf scripts under `scripts/`.
- **Training / GSV-Cities:** see below.
- **COMP 560 alignment (metrics, datasets, integrity):** `PLAN.md` §11.

## Longleaf (UNC) Slurm

Batch jobs source `scripts/slurm_longleaf_init.sh`, which: loads the Anaconda module, runs `conda activate "${CONDA_ENV_NAME:-learnerpr}"`, exports **`PYTHON`** as `${CONDA_PREFIX}/bin/python`, and verifies `import torch` (skipped for the GSV download job via `LEARNERPR_SKIP_TORCH_CHECK=1`). All `*.sl` scripts invoke `"${PYTHON}" src/...` so Slurm never falls back to system `/usr/bin/python`.

**Repo path:** Each `*.sl` sets `LEARNERPR_REPO` (default `/users/a/l/alshen/LearnerPR/LearnerPR`) and sources `scripts/slurm_longleaf_init.sh` from there, then `cd`s into that directory so `python src/...` works even when Slurm copies the batch script to `/var/spool/...` or you `sbatch` from a parent folder. Override if your clone lives elsewhere: `LEARNERPR_REPO=/path/to/inner/LearnerPR sbatch scripts/train.sl`.

Optional environment overrides: `ANACONDA_MODULE` (default `anaconda/2024.02`), `CONDA_ENV_NAME` (default `learnerpr`). Install deps in that env (`pip install -r requirements.txt`) so the torch check passes.

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