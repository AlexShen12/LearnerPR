"""Train the DINOv2-S + GeM student with RKD from cached teacher embeddings.

Usage:
    python src/train.py \
        --config configs/default.yaml \
        --augment \
        --epochs 30
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import yaml

try:
    from models.student import StudentModel
except ImportError:
    from models.student import DINOv2GeMStudent as StudentModel
from losses.rkd import CombinedLoss, mine_batch_hard
from data.msls import MSLSDataset
from data.gsv_cities import GSVCitiesDataset, auto_val_cities, make_gsv_val_splits
from data.cached_teacher import CachedTeacherDataset
from data.augmentations import get_train_transform, get_eval_transform
from data.pk_sampler import PKPlaceBatchSampler


def parse_args():
    p = argparse.ArgumentParser(description="Train LearnerPR student.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--init_from", type=str, default=None,
                   help="Load model weights from checkpoint and start a fresh schedule "
                        "(optimizer/epoch are NOT restored). Mutually exclusive with --resume.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def recall_at_k(query_embeds, db_embeds, query_place_ids, db_place_ids, ks=(1, 5, 10, 20)):
    """Compute Recall@K for a set of query and database embeddings."""
    sims = query_embeds @ db_embeds.t()
    n_db = db_embeds.shape[0]
    results = {}
    for k in ks:
        # Guard against databases smaller than k (e.g. tiny GSV val splits).
        effective_k = min(k, n_db)
        topk_indices = sims.topk(effective_k, dim=1).indices
        correct = 0
        for i, qid in enumerate(query_place_ids):
            retrieved_ids = [db_place_ids[j] for j in topk_indices[i]]
            if qid in retrieved_ids:
                correct += 1
        results[f"recall_at_{k}"] = correct / len(query_place_ids) if query_place_ids else 0.0
    return results


@torch.no_grad()
def evaluate(model, val_db_loader, val_query_loader, device, ks=(1, 5, 10, 20)):
    """Encode val database + queries, compute Recall@K."""
    model.eval()

    def encode_loader(loader):
        all_embeds, all_ids = [], []
        for batch in loader:
            imgs = batch["image"].to(device)
            embeds = model(imgs)
            all_embeds.append(embeds.cpu())
            all_ids.extend(batch["place_id"])
        return torch.cat(all_embeds), all_ids

    db_embeds, db_ids = encode_loader(val_db_loader)
    q_embeds, q_ids = encode_loader(val_query_loader)
    return recall_at_k(q_embeds, db_embeds, q_ids, db_ids, ks=ks)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    scfg = cfg["student"]

    epochs = args.epochs or tcfg["epochs"]
    batch_size = args.batch_size or tcfg["batch_size"]
    lr = args.lr or tcfg["lr"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Transforms ───────────────────────────────────────────────────
    aug_cfg = cfg.get("augmentation", {}) if args.augment else None
    train_transform = get_train_transform(
        augment=args.augment,
        image_size=cfg["evaluation"]["image_size"],
        cfg=aug_cfg,
    )
    eval_transform = get_eval_transform(image_size=cfg["evaluation"]["image_size"])

    # ── Datasets ─────────────────────────────────────────────────────
    ckpt_dir = Path(os.path.expandvars(cfg["paths"]["checkpoint_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_datasets_cfg: list[str] = tcfg.get("train_datasets", ["msls"])
    train_parts: list = []

    # MSLS training set
    if "msls" in train_datasets_cfg:
        msls_root = os.path.expandvars(cfg["paths"]["msls_root"])
        msls_cache = os.path.expandvars(cfg["paths"]["teacher_cache"])
        msls_train_ds = MSLSDataset(msls_root, split="train", subset="database",
                                    transform=train_transform)
        train_parts.append(CachedTeacherDataset(msls_train_ds, msls_cache))
        print(f"MSLS train: {len(msls_train_ds)} images")

    # GSV-Cities training set
    gsv_val_db_loader = gsv_val_q_loader = None
    if "gsv_cities" in train_datasets_cfg:
        gsv_root = os.path.expandvars(cfg["paths"]["gsv_cities_root"])
        gsv_cache = os.path.expandvars(cfg["paths"]["teacher_cache_gsv"])

        val_cities_cfg: list[str] = tcfg.get("gsv_val_cities", [])
        if val_cities_cfg:
            val_cities = val_cities_cfg
            train_cities, _ = auto_val_cities(gsv_root, fraction=0.0)
            # exclude val cities from train
            train_cities = [c for c in train_cities if c not in val_cities]
        else:
            train_cities, val_cities = auto_val_cities(
                gsv_root, fraction=tcfg.get("gsv_val_fraction", 0.1)
            )

        gsv_train_ds = GSVCitiesDataset(gsv_root, cities=train_cities,
                                        transform=train_transform)
        train_parts.append(CachedTeacherDataset(gsv_train_ds, gsv_cache))
        print(f"GSV-Cities train: {len(gsv_train_ds)} images  ({len(train_cities)} cities)")

        if tcfg.get("gsv_val_enabled", True) and val_cities:
            gsv_val_db_ds, gsv_val_q_ds = make_gsv_val_splits(
                gsv_root, val_cities, transform=eval_transform
            )
            gsv_val_db_loader = DataLoader(
                gsv_val_db_ds, batch_size=batch_size, num_workers=4, pin_memory=True
            )
            gsv_val_q_loader = DataLoader(
                gsv_val_q_ds, batch_size=batch_size, num_workers=4, pin_memory=True
            )
            print(f"GSV-Cities val:   {len(gsv_val_db_ds)} db / {len(gsv_val_q_ds)} queries"
                  f"  ({len(val_cities)} held-out cities)")

    if not train_parts:
        raise ValueError("train_datasets must contain at least one of: msls, gsv_cities")

    combined_train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

    # P×K triplet mode: only supported for GSV-only training.
    use_triplet = tcfg.get("beta_triplet", 0.0) > 0.0
    is_gsv_only = train_datasets_cfg == ["gsv_cities"]
    if use_triplet and not is_gsv_only:
        print(
            "WARNING: P×K triplet sampling requires GSV-only training. "
            "Mixed MSLS+GSV detected — triplet loss disabled for this run."
        )
        use_triplet = False

    if use_triplet:
        P = tcfg.get("triplet_p", 16)
        K = tcfg.get("triplet_k", 4)
        # place_ids parallel to combined_train_ds (CachedTeacherDataset wraps GSVCitiesDataset)
        place_ids = combined_train_ds.base.place_ids
        pk_sampler = PKPlaceBatchSampler(place_ids=place_ids, P=P, K=K)
        train_loader = DataLoader(
            combined_train_ds, batch_sampler=pk_sampler,
            num_workers=4, pin_memory=True,
        )
        print(f"P×K sampler: P={P}, K={K}, batch={P*K}, batches/epoch={len(pk_sampler)}")
    else:
        train_loader = DataLoader(
            combined_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )

    print(f"Total train samples: {len(combined_train_ds)}")

    # MSLS val — only built when MSLS is in train_datasets.
    val_db_loader = val_q_loader = None
    if "msls" in train_datasets_cfg:
        msls_root = os.path.expandvars(cfg["paths"]["msls_root"])
        val_db_ds = MSLSDataset(msls_root, split="val", subset="database", transform=eval_transform)
        val_q_ds = MSLSDataset(msls_root, split="val", subset="query", transform=eval_transform)
        val_db_loader = DataLoader(val_db_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
        val_q_loader = DataLoader(val_q_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────
    model = StudentModel(
        backbone_name=scfg["backbone"],
        embed_dim=scfg["embed_dim"],
        gem_p_init=scfg["gem_p_init"],
        use_projection=True,
    ).to(device)
    model.freeze_backbone()

    # ── Loss & Optimizer ─────────────────────────────────────────────
    criterion = CombinedLoss(
        alpha=tcfg["alpha_rkd"],
        beta=tcfg["beta_triplet"] if use_triplet else 0.0,
        rkd_temperature=tcfg["rkd_temperature"],
        triplet_margin=tcfg["triplet_margin"],
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=tcfg["weight_decay"],
    )

    # Base LRs tracked manually so warmup + cosine works correctly across
    # dynamically added param groups (backbone unfreezes mid-training).
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    base_lrs: list[float] = [lr]  # one entry per param group; grows at unfreeze

    def set_lrs(epoch: int) -> None:
        """Apply warmup-then-cosine LR to all current param groups."""
        W = warmup_epochs
        E = epochs
        warmup_mult = (epoch + 1) / max(W, 1) if epoch < W else 1.0
        if epoch < W:
            cos_mult = 1.0
        else:
            t = epoch - W
            T = max(E - W, 1)
            cos_mult = 0.5 * (1.0 + math.cos(math.pi * t / T))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * warmup_mult * cos_mult

    # ── Checkpoint init / resume ─────────────────────────────────────
    if args.resume and args.init_from:
        raise SystemExit("ERROR: --resume and --init_from are mutually exclusive. Use one at a time.")

    start_epoch = 0
    best_recall = 0.0
    patience_counter = 0

    if args.init_from and os.path.exists(args.init_from):
        # Model-weights-only init: fresh optimizer, schedule, and epoch counter.
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  init_from: {len(missing)} missing keys (e.g. {missing[0]})")
        if unexpected:
            print(f"  init_from: {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
        print(f"Initialized weights from {args.init_from} — fresh schedule starting at epoch 0.")

    elif args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # Restore tracked base_lrs if saved, otherwise fall back to lr.
        base_lrs = ckpt.get("base_lrs", base_lrs)
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", 0.0)
        print(f"Resumed from epoch {start_epoch}, best R@5 = {best_recall:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()

        # Unfreeze backbone after warm-up phases; add param group + track base LR.
        if epoch == scfg["freeze_backbone_epochs"]:
            print(f"Epoch {epoch}: unfreezing last {scfg['unfreeze_last_n_blocks']} blocks.")
            model.unfreeze_last_n_blocks(scfg["unfreeze_last_n_blocks"])
            backbone_lr = lr * scfg.get("backbone_lr_scale", 0.1)
            optimizer.add_param_group({
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": backbone_lr,
            })
            base_lrs.append(backbone_lr)

        # Apply LR schedule (warmup + cosine) at the start of each epoch.
        set_lrs(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = batch["image"].to(device)
            teacher_embeds = batch["teacher_embed"].to(device)

            student_embeds = model(imgs)

            if use_triplet:
                place_ids_batch = batch["place_id"].to(device)
                anchors, positives, negatives = mine_batch_hard(student_embeds, place_ids_batch)
                losses = criterion(student_embeds, teacher_embeds, anchors, positives, negatives)
            else:
                losses = criterion(student_embeds, teacher_embeds)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} — loss: {avg_loss:.4f}, lr: {current_lr:.2e}")

        # ── Validation ───────────────────────────────────────────────
        ks = tuple(cfg["evaluation"]["recall_ks"])
        early_stop_k = int(tcfg["early_stop_metric"].split("_")[-1])
        current_recall = 0.0
        metrics: dict = {}
        gsv_metrics: dict = {}

        if val_db_loader is not None and val_q_loader is not None:
            metrics = evaluate(model, val_db_loader, val_q_loader, device, ks=ks)
            metric_str = "  ".join(f"R@{k}: {metrics[f'recall_at_{k}']:.4f}" for k in ks)
            print(f"  MSLS val:    {metric_str}")
            current_recall = metrics[f"recall_at_{early_stop_k}"]

        if gsv_val_db_loader is not None and gsv_val_q_loader is not None:
            gsv_metrics = evaluate(model, gsv_val_db_loader, gsv_val_q_loader, device, ks=ks)
            gsv_str = "  ".join(f"R@{k}: {gsv_metrics[f'recall_at_{k}']:.4f}" for k in ks)
            print(f"  GSV val:     {gsv_str}")
            # Use GSV val as early-stop signal when MSLS val is not available.
            if val_db_loader is None:
                current_recall = gsv_metrics[f"recall_at_{early_stop_k}"]

        # ── Checkpointing ────────────────────────────────────────────
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "base_lrs": base_lrs,
            "metrics": {**metrics, **{f"gsv_{k}": v for k, v in gsv_metrics.items()}},
            "best_recall": max(best_recall, current_recall),
            "augment": args.augment,
        }
        torch.save(ckpt_data, ckpt_dir / "latest.pt")

        if current_recall > best_recall:
            best_recall = current_recall
            patience_counter = 0
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            print(f"  New best R@5 = {best_recall:.4f} — saved.")
        else:
            patience_counter += 1
            if patience_counter >= tcfg["early_stop_patience"]:
                print(f"  Early stopping after {tcfg['early_stop_patience']} epochs without improvement.")
                break

    print(f"Training complete. Best val R@5 = {best_recall:.4f}")


if __name__ == "__main__":
    main()
