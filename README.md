# LearnerPR

VPR student: **DINOv2-Small + GeM**, trained with RKD from **Qwen3-VL-8B-Instruct** (see `PLAN.md`).

- **Course submission:** root [`model.py`](model.py) implements the COMP 560 `StudentModel` API (`device`, `encode`, `embedding_dim`) for the provided `evaluate.py`. Place trained weights as `submission_weights.pt` or set `LEARNERPR_WEIGHTS`.
- **Training / MSLS:** `src/train.py`, Longleaf scripts under `scripts/`.
- **Training / GSV-Cities:** see below.
- **COMP 560 alignment (metrics, datasets, integrity):** `PLAN.md` §11.

## GSV-Cities setup

GSV-Cities (~530k images, ~62k places across 40+ cities) can be added as a second training source alongside MSLS.

**Prerequisites:**

1. A [Kaggle account](https://www.kaggle.com) with an API token saved to `~/.kaggle/kaggle.json` (chmod 600). Accept the dataset license at <https://www.kaggle.com/datasets/amaralibey/gsv-cities>.
2. ~60 GB free on your work filesystem.

**Download (once):**

```bash
GSV_CITIES_ROOT=/work/$USER/datasets/gsv-cities \
    bash scripts/download_gsv_cities.sh
# or: sbatch scripts/download_gsv_cities.sh
```

Expected layout after unzip:

```
gsv-cities/
    Images/<City>/*.JPG        # ~530k images
    Dataframes/<City>.csv      # place_id metadata per city
```

**Cache teacher embeddings for GSV-Cities (once):**

```bash
GSV_CITIES_ROOT=/work/$USER/datasets/gsv-cities \
    sbatch scripts/cache_teacher_gsv.sh
```

**Joint training (MSLS + GSV-Cities):**

Edit `configs/default.yaml`:
```yaml
training:
  train_datasets: [msls, gsv_cities]
```

Then run training as usual:
```bash
sbatch scripts/train.sh
```

**GSV-Cities eval (held-out cities):**

```bash
CHECKPOINT=/work/$USER/learnerpr/checkpoints/best.pt \
    sbatch scripts/eval_gsv.sh
```