"""Evaluate a trained LearnerPR student on GSV-Cities held-out cities.

The val split is constructed by make_gsv_val_splits: for each place_id in the
held-out cities, the most recent image becomes a query and all prior images form the
database.  Queries and database are strictly non-overlapping, so no self-masking is
needed.

Usage:
    python src/eval_gsv.py \
        --config configs/default.yaml \
        --checkpoint /work/$USER/learnerpr/checkpoints/best.pt \
        --held_out_cities Austin Bangkok BuenosAires
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

try:
    from models.student import StudentModel
except ImportError:
    from models.student import DINOv2GeMStudent as StudentModel
from data.gsv_cities import auto_val_cities, make_gsv_val_splits
from data.augmentations import get_eval_transform


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LearnerPR student on GSV-Cities held-out cities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--held_out_cities",
        type=str,
        nargs="*",
        default=None,
        help="City folder names to use as the val set.  Omit to auto-select via "
             "gsv_val_fraction from the config.",
    )
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


def recall_at_k(q_embeds, db_embeds, q_ids, db_ids, ks=(1, 5, 10, 20)):
    sims = q_embeds @ db_embeds.t()
    results = {}
    for k in ks:
        topk_idx = sims.topk(k, dim=1).indices
        correct = 0
        for i, qid in enumerate(q_ids):
            if any(db_ids[j] == qid for j in topk_idx[i].tolist()):
                correct += 1
        results[f"recall_at_{k}"] = correct / len(q_ids) if q_ids else 0.0
    return results


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    scfg = cfg["student"]
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────
    model = StudentModel(
        backbone_name=scfg["backbone"],
        embed_dim=scfg["embed_dim"],
        gem_p_init=scfg["gem_p_init"],
        use_projection=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("projection")}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1}")

    # ── Val split ────────────────────────────────────────────────────
    gsv_root = os.path.expandvars(cfg["paths"]["gsv_cities_root"])

    if args.held_out_cities:
        val_cities = args.held_out_cities
    else:
        val_cities_cfg: list[str] = tcfg.get("gsv_val_cities", [])
        if val_cities_cfg:
            val_cities = val_cities_cfg
        else:
            _, val_cities = auto_val_cities(gsv_root, fraction=tcfg.get("gsv_val_fraction", 0.1))

    if not val_cities:
        raise ValueError(
            "No held-out cities found.  Pass --held_out_cities or set gsv_val_cities / "
            "gsv_val_fraction in the config."
        )

    print(f"Held-out cities ({len(val_cities)}): {', '.join(val_cities)}")

    transform = get_eval_transform(image_size=cfg["evaluation"]["image_size"])
    db_ds, q_ds = make_gsv_val_splits(gsv_root, val_cities, transform=transform)

    db_loader = DataLoader(db_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    q_loader = DataLoader(q_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    print(f"Database: {len(db_ds)} images   Queries: {len(q_ds)} images")

    # ── Encode ───────────────────────────────────────────────────────
    db_embeds, db_ids, _ = encode_dataset(model, db_loader, device)
    q_embeds, q_ids, _ = encode_dataset(model, q_loader, device)

    # ── Recall@K ─────────────────────────────────────────────────────
    ks = tuple(cfg["evaluation"]["recall_ks"])
    metrics = recall_at_k(q_embeds, db_embeds, q_ids, db_ids, ks=ks)

    print(f"\n{'='*50}")
    print(f"GSV-Cities eval — held-out: {', '.join(val_cities)}")
    print(f"{'='*50}")
    for k in ks:
        print(f"  Recall@{k:>2d}: {metrics[f'recall_at_{k}']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
