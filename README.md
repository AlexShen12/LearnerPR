# LearnerPR

VPR student: **DINOv2-Small + GeM**, trained with RKD from **Qwen3-VL-8B-Instruct** (see `PLAN.md`).

- **Course submission:** root [`model.py`](model.py) implements the COMP 560 `StudentModel` API (`device`, `encode`, `embedding_dim`) for the provided `evaluate.py`. Place trained weights as `submission_weights.pt` or set `LEARNERPR_WEIGHTS`.
- **Training / MSLS:** `src/train.py`, Longleaf scripts under `scripts/`.
- **COMP 560 alignment (metrics, datasets, integrity):** `PLAN.md` §11.