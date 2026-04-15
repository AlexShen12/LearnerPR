# LearnerPR — Project Plan

## Overview

LearnerPR is a Visual Place Recognition (VPR) system that distills relational knowledge
from a large vision-language teacher (**Qwen3-VL-8B-Instruct**) into a lightweight
student backbone (**DINOv2-Small + GeM pooling**) using Relational Knowledge
Distillation (RKD). The student produces compact 384-d image descriptors suitable for
large-scale geo-retrieval on resource-constrained hardware.

---

## 1. Architecture

### 1.1 Student — DINOv2-Small + GeM + FC head

```
image → DINOv2-S (frozen or fine-tuned patch tokens)
      → GeM pool (learnable p)
      → FC 384→384 (projection head, used during training only)
      → L2-normalize
      → 384-d descriptor
```

- **Backbone:** `facebook/dinov2-small` (ViT-S/14, ~22M params, 384-d).
- **GeM pooling:** Replaces DINOv2's default CLS token with generalized-mean pooling
  over all patch tokens. The exponent `p` is a learnable scalar (init ~3.0) that lets the
  network up-weight high-activation regions (architectural features, lane markings) vs
  uniform background.
- **Projection head:** A single linear layer used only during RKD training so the
  backbone embedding space is not distorted; discarded at evaluation.
- **Fine-tuning strategy:** Freeze DINOv2 for the first N epochs (warm-up), then
  unfreeze the last K transformer blocks for joint training with GeM + head.

### 1.2 Teacher — Qwen3-VL-8B-Instruct (frozen)

```
image → Qwen3-VL vision encoder + multimodal merger
      → mean-pool image hidden states (language head unused)
      → L2-normalize
      → D-d descriptor  (D depends on model hidden size)
```

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` loaded in `bfloat16`, **inference-only**.
- **Embedding extraction protocol (fixed for entire project):**
  1. Resize shortest side to 448 px, center-crop to 448×448.
  2. Single `ImagePart` message, **no text prompt** (or minimal fixed prompt like
     `"Describe this place."` — chosen once and never changed).
  3. Run `model.forward()` → take the **last hidden state** at all **image token
     positions** (identified via `image_token_id` in the Qwen processor).
  4. Mean-pool those positions → single vector.
  5. L2-normalize.
- **Why image-only:** VPR needs fine-grained visual similarity, not caption semantics.
  Conditioning on text would shift the teacher's geometry toward language alignment,
  which is the wrong inductive bias for geo-retrieval.

### 1.3 Relational Knowledge Distillation (RKD)

For a batch of N images:

1. Compute teacher descriptors `T ∈ ℝ^{N×D_t}` (from cache or live).
2. Compute student descriptors `S ∈ ℝ^{N×D_s}`.
3. Build pairwise cosine similarity matrices:
   - `M_t = T · Tᵀ`  (N×N, teacher)
   - `M_s = S · Sᵀ`  (N×N, student)
4. **Distance RKD loss:**
   `L_rkd = MSE(M_s, M_t)` or `Huber(M_s, M_t)` (Huber is more robust to outlier
   pairs).
5. Optionally add a **triplet / contrastive loss** on the student alone for direct metric
   learning signal.

**Total loss:**

```
L = α · L_rkd  +  β · L_triplet
```

Default: `α = 1.0`, `β = 0.5`. Sweep if time permits.

---

## 2. Dataset — MSLS (Mapillary Street-Level Sequences)

- **Source:** Official MSLS download via the Mapillary research page.
- **Splits:** Use the **official MSLS train / val / test** city-based splits (no random
  shuffling). This tests generalization to **unseen cities**, the correct protocol.
- **Training images:** Database + query images from the train cities.
- **Evaluation images:** Official val and test queries + databases.

### 2.1 Data loading

Each sample is a single image identified by its MSLS key. During RKD training we
load batches of images that share a city (or are randomly sampled across cities) and
compute pairwise teacher/student matrices over the batch.

Place labels (GPS-derived place IDs) are used only for the optional triplet loss and
for evaluation, **not** for RKD itself (RKD is label-free on the teacher side).

---

## 3. Data Augmentation (optional, flag-controlled)

All augmentations are applied **only to the student input**, never to the teacher
(teacher embeddings are pre-cached from clean images).

| Augmentation | Purpose | Default |
|---|---|---|
| RandomResizedCrop(224, scale=(0.7, 1.0)) | Viewpoint invariance | ON |
| ColorJitter(0.4, 0.4, 0.4, 0.1) | Lighting / seasonal shift | ON |
| RandomGrayscale(p=0.1) | Channel-drop regularization | ON |
| GaussianBlur(kernel=23, σ=[0.1, 2.0]) | Sensor noise / motion blur | OFF |
| RandomHorizontalFlip(p=0.5) | Mirror invariance | OFF (risky for geo) |

Controlled by `--augment` flag in training script. When `--augment` is off, student
sees the same deterministic resize + center-crop as the teacher.

---

## 4. Training Pipeline

### Phase 0 — Cache teacher embeddings (one-time)

Run Qwen3-VL-8B-Instruct over the entire MSLS training set and store the resulting
L2-normalized vectors in a single `.pt` file keyed by image path. This avoids running
the 8B model during every training epoch.

**Script:** `scripts/cache_teacher.sh` → calls `src/cache_teacher_embeddings.py`

### Phase 1 — Train student with RKD

1. Load cached teacher embeddings.
2. For each batch, look up teacher vectors → build `M_t`.
3. Forward student on (possibly augmented) images → build `M_s`.
4. Compute `L = α·L_rkd + β·L_triplet`, backprop through student only.
5. Optimizer: AdamW, lr=1e-4 with cosine decay, weight decay=1e-4.
6. Epochs: 30 (early-stop on val R@5).

**Script:** `scripts/train.sh` → calls `src/train.py`

### Phase 2 — Evaluate

1. Encode MSLS val/test database + queries through the trained student (no aug, no
   projection head).
2. Compute cosine similarities, rank, report **R@1, R@5, R@10, R@20**.

**Script:** `scripts/eval.sh` → calls `src/eval.py`

---

## 5. Metrics

| Metric | Definition |
|---|---|
| **Recall@K** (K=1,5,10,20) | Fraction of queries whose true match is in the top-K retrieved database images by cosine similarity. |

Primary metric for model selection / early stopping: **R@5** on MSLS val.

---

## 6. Baselines

| ID | Description | Purpose |
|---|---|---|
| B1 | DINOv2-S + GeM, **no distillation** (triplet only) | Isolate RKD contribution |
| B2 | DINOv2-S + CLS token (no GeM), no distillation | Isolate GeM contribution |
| B3 | DINOv2-S + GeM + RKD **without** augmentation | Isolate augmentation contribution |

---

## 7. Directory Layout

```
LearnerPR/
├── PLAN.md                  ← this file
├── model.py                 ← COMP 560 `StudentModel` API for course `evaluate.py`
├── requirements.txt
├── setup.sh                 ← conda/pip env bootstrap for Longleaf
├── src/
│   ├── models/
│   │   ├── student.py       ← DINOv2-S + GeM + projection head
│   │   └── teacher.py       ← Qwen3-VL-8B embedding extractor
│   ├── losses/
│   │   └── rkd.py           ← RKD + optional triplet loss
│   ├── data/
│   │   ├── msls.py          ← MSLS dataset / dataloader
│   │   └── augmentations.py ← torchvision transform pipelines
│   ├── cache_teacher_embeddings.py
│   ├── train.py
│   └── eval.py
├── scripts/
│   ├── cache_teacher.sh     ← SLURM job: cache Qwen3-VL embeddings
│   ├── train.sh             ← SLURM job: train student
│   └── eval.sh              ← SLURM job: evaluate student
├── configs/
│   └── default.yaml         ← hyperparams, paths, flags
└── outputs/                  ← checkpoints, logs, cached embeddings (gitignored)
```

---

## 8. Longleaf Specifics

- **Partition:** `gpu` (A100 nodes).
- **Modules:** `cuda/12.x`, `anaconda` (or miniconda via setup.sh).
- **Storage:** MSLS images on `/work` or `/pine`; checkpoints + cache under
  `$SLURM_TMPDIR` during job, copied to `/work` on completion.
- **Walltime estimates:**
  - Cache teacher: ~2–4 h (MSLS train, single A100, batch 16).
  - Train student: ~1–2 h per epoch × 30 epochs (early-stop likely ~15).
  - Eval: ~10 min per split.

---

## 9. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Qwen3-VL-8B deps break on Longleaf | Fall back to `Qwen2-VL-7B-Instruct` (same pipeline, swap checkpoint string). |
| Teacher similarity matrix too flat / too sharp | Apply temperature scaling `τ` to `M_t` before loss; sweep `τ ∈ {0.05, 0.1, 0.5}`. |
| Student collapses (all embeddings identical) | Monitor embedding std per epoch; add triplet loss as regularizer. |
| A100 OOM on teacher forward | Reduce image resolution to 336 px or use `torch.compile` / `flash_attn`. |

---

## 10. Timeline (suggested)

| Week | Milestone |
|---|---|
| 1 | Env setup, MSLS download, teacher caching script working |
| 2 | Student model + RKD loss implemented, first training run |
| 3 | Augmentation ablation, baseline runs, hyperparameter sweep |
| 4 | Final eval, report writing |

---

## 11. COMP 560 course requirements (final project handout)

This section maps the **official project spec** to this repository so submission and
training stay aligned.

### 11.1 Required `StudentModel` API

The course harness expects a **single file** (`model.py`) with:

| Requirement | This repo |
|---|---|
| `__init__(self, device: str = "cuda")` | `model.py` → `StudentModel` |
| `encode(self, images)` → `(B, D)` L2-normalized, ImageNet-normalized `(B, 3, H, W)` input | Implemented; delegates to `src/models/student.py` (`DINOv2-S + GeM`, no projection at eval) |
| `embedding_dim` property | Returns **384** (DINOv2-Small channel width) |

**Weights for grading:** copy your best checkpoint to `submission_weights.pt` next to
`model.py`, or set `LEARNERPR_WEIGHTS=/path/to/best.pt` before running `evaluate.py`.
Training checkpoints from `src/train.py` (dict with `"model"` key) load correctly.

**Submit as a bundle:** `model.py` + `src/models/student.py` + `submission_weights.pt`
(or the full repo zip if staff allow). The course text only names `model.py` explicitly;
include dependencies your import chain needs.

### 11.2 Metrics

| Handout | LearnerPR |
|---|---|
| R@1, R@5, R@10, R@20 | `src/eval.py`, `configs/default.yaml` → `recall_ks` |
| mAP@K (CSV shows **mAP@20**) | `src/eval.py` now reports **mAP@1,5,10,20** (place-level AP@K, averaged over queries). Course `evaluate.py` remains the **authoritative** numbers for grading. |
| Efficiency (throughput, peak memory, embedding dim, total time) | Produced by the provided **`evaluate.py`**, not `src/eval.py`. Run their script on your `model.py` after training. |

### 11.3 Datasets A / B vs MSLS

| Handout | Implication |
|---|---|
| **Dataset A** — small, 1-to-1 by index, for fast iteration | Use for debugging with `--debug` on the course `evaluate.py`. |
| **Dataset B** — large street-view, **final evaluation** | Treat as **held-out**: do **not** use for training or hyperparameter selection. |
| **MSLS** (this plan) | Research / Longleaf pipeline; **not** a substitute for validating on the course release. Before submission, always run the official `evaluate.py` on **Dataset A + B** with your `model_path` pointing at this repo’s `model.py`. |

### 11.4 Academic integrity (verbatim constraint)

> You may NOT use any public test set for training or hyperparameter tuning.

**Action items:**

- Do **not** train or tune on **Dataset B** if it is the class test set (assume it is).
- Do **not** tune on **MSLS test** cities for model selection; use MSLS **train** for
  learning, **val** (or a separate held-out subset) for early stopping only, and treat
  MSLS **test** as off-limits for hyperparameters if you mirror the same rule as the course.
- **Teacher embedding cache:** build caches only from images you are allowed to use for
  training (e.g. MSLS train cities, **not** course test data).

### 11.5 Grading weights (from handout)

| Component | Weight |
|---|---|
| Performance (Recall on held-out test) | 40% |
| Efficiency | 30% |
| Report (max 4 pages) | 30% |

**Efficiency:** A **384-d** student is favorable vs the ResNet50 baseline (2048-d). Target
high throughput on `encode()` (batch inference, `torch.inference_mode`, no teacher at
test time).

### 11.6 Checklist before submission

1. [ ] `python evaluate.py --student_id … --model_path …/LearnerPR/model.py --datasets_root …` passes on Dataset A (full + debug).
2. [ ] Same on Dataset B **once**, when allowed (final run only if B is truly blind).
3. [ ] `submission_weights.pt` or `LEARNERPR_WEIGHTS` set; confirm non-trivial R@1 vs
      untrained DINOv2+GeM.
4. [ ] Report PDF ≤ 4 pages, cites pretrained sources (DINOv2, Qwen3-VL, etc.).
5. [ ] No test-set leakage in training or HPO (§11.4).
