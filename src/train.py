"""Train the DINOv2-S + GeM student with RKD from cached teacher embeddings.

Usage:
    python src/train.py \
        --config configs/default.yaml \
        --augment \
        --epochs 30
"""

import argparse
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
from losses.rkd import CombinedLoss
from data.msls import MSLSDataset
from data.gsv_cities import GSVCitiesDataset, auto_val_cities, make_gsv_val_splits
from data.cached_teacher import CachedTeacherDataset
from data.augmentations import get_train_transform, get_eval_transform


def parse_args():
    p = argparse.ArgumentParser(description="Train LearnerPR student.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def recall_at_k(query_embeds, db_embeds, query_place_ids, db_place_ids, ks=(1, 5, 10, 20)):
    """Compute Recall@K for a set of query and database embeddings."""
    sims = query_embeds @ db_embeds.t()
    results = {}
    for k in ks:
        topk_indices = sims.topk(k, dim=1).indices
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
        beta=tcfg["beta_triplet"],
        rkd_temperature=tcfg["rkd_temperature"],
        triplet_margin=tcfg["triplet_margin"],
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_recall = 0.0
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", 0.0)
        print(f"Resumed from epoch {start_epoch}, best R@5 = {best_recall:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()

        # Unfreeze backbone after warm-up
        if epoch == scfg["freeze_backbone_epochs"]:
            print(f"Epoch {epoch}: unfreezing last {scfg['unfreeze_last_n_blocks']} blocks.")
            model.unfreeze_last_n_blocks(scfg["unfreeze_last_n_blocks"])
            optimizer.add_param_group({
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": lr * 0.1,
            })

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = batch["image"].to(device)
            teacher_embeds = batch["teacher_embed"].to(device)

            student_embeds = model(imgs)
            losses = criterion(student_embeds, teacher_embeds)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} — loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")

        # ── Validation ───────────────────────────────────────────────
        ks = tuple(cfg["evaluation"]["recall_ks"])
        early_stop_k = int(tcfg["early_stop_metric"].split("_")[-1])
        current_recall = 0.0

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
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
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
