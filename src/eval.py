"""Evaluate a trained LearnerPR student on MSLS val or test.

Usage:
    python src/eval.py \
        --config configs/default.yaml \
        --checkpoint /work/$USER/learnerpr/checkpoints/best.pt \
        --split val
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.student import StudentModel
from data.msls import MSLSDataset
from data.augmentations import get_eval_transform


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LearnerPR student.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def encode_dataset(model, loader, device):
    model.eval()
    all_embeds, all_ids, all_keys = [], [], []
    for batch in tqdm(loader, desc="Encoding"):
        imgs = batch["image"].to(device)
        embeds = model(imgs)
        all_embeds.append(embeds.cpu())
        all_ids.extend(batch["place_id"])
        all_keys.extend(batch["key"])
    return torch.cat(all_embeds), all_ids, all_keys


def recall_at_k(query_embeds, db_embeds, query_ids, db_ids, ks=(1, 5, 10, 20)):
    sims = query_embeds @ db_embeds.t()
    results = {}
    for k in ks:
        topk_indices = sims.topk(k, dim=1).indices
        correct = 0
        for i, qid in enumerate(query_ids):
            if qid == -1:
                continue
            retrieved = {db_ids[j] for j in topk_indices[i]}
            if qid in retrieved:
                correct += 1
        n_valid = sum(1 for qid in query_ids if qid != -1)
        results[f"recall_at_{k}"] = correct / n_valid if n_valid > 0 else 0.0
    return results


def _average_precision_at_k(sims_row: torch.Tensor, relevant: torch.Tensor, k: int) -> float:
    """
    AP truncated at rank K: sum over ranks r<=K of P(r)*rel(r), divided by total
    positives R in the full database (standard IR normalization). If R=0, returns 0.
    """
    r_total = int(relevant.sum().item())
    if r_total == 0:
        return 0.0
    _, order = sims_row.sort(descending=True)
    ap = 0.0
    hits = 0
    for rank, idx in enumerate(order.tolist()[:k], start=1):
        if relevant[idx]:
            hits += 1
            ap += hits / rank
    return ap / r_total


def map_at_k(
    query_embeds: torch.Tensor,
    db_embeds: torch.Tensor,
    query_ids: list,
    db_ids: list,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """
    Mean AP@K over queries for place-level relevance (all DB rows with matching
    `place_id` are positives). Aligns with course reporting of mAP@20 on Dataset B.

    Queries with `place_id == -1` are skipped.
    """
    sims = query_embeds @ db_embeds.t()
    db_ids_t = torch.tensor(db_ids, dtype=torch.long)
    results = {f"map_at_{k}": 0.0 for k in ks}
    counts = {k: 0 for k in ks}

    for i, qid in enumerate(query_ids):
        if qid == -1:
            continue
        relevant = db_ids_t == qid
        if not relevant.any():
            continue
        for k in ks:
            results[f"map_at_{k}"] += _average_precision_at_k(sims[i], relevant, k)
            counts[k] += 1

    for k in ks:
        if counts[k] > 0:
            results[f"map_at_{k}"] /= counts[k]
    return results


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    scfg = cfg["student"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model (no projection head at eval) ───────────────────────────
    model = StudentModel(
        backbone_name=scfg["backbone"],
        embed_dim=scfg["embed_dim"],
        gem_p_init=scfg["gem_p_init"],
        use_projection=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"]
    # Drop projection keys since we set use_projection=False for eval
    state = {k: v for k, v in state.items() if not k.startswith("projection")}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")

    # ── Data ─────────────────────────────────────────────────────────
    msls_root = os.path.expandvars(cfg["paths"]["msls_root"])
    transform = get_eval_transform(image_size=cfg["evaluation"]["image_size"])

    db_ds = MSLSDataset(msls_root, split=args.split, subset="database", transform=transform)
    q_ds = MSLSDataset(msls_root, split=args.split, subset="query", transform=transform)
    db_loader = DataLoader(db_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    q_loader = DataLoader(q_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # ── Encode & evaluate ────────────────────────────────────────────
    print(f"Encoding {len(db_ds)} database images...")
    db_embeds, db_ids, _ = encode_dataset(model, db_loader, device)
    print(f"Encoding {len(q_ds)} query images...")
    q_embeds, q_ids, _ = encode_dataset(model, q_loader, device)

    ks = tuple(cfg["evaluation"]["recall_ks"])
    metrics = recall_at_k(q_embeds, db_embeds, q_ids, db_ids, ks=ks)
    map_metrics = map_at_k(q_embeds, db_embeds, q_ids, db_ids, ks=ks)

    print(f"\n{'='*50}")
    print(f"Results on MSLS {args.split}")
    print(f"{'='*50}")
    for k in ks:
        print(f"  Recall@{k:>2d}: {metrics[f'recall_at_{k}']:.4f}")
    for k in ks:
        print(f"  mAP@{k:>2d}:    {map_metrics[f'map_at_{k}']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
