"""Cache Qwen3-VL-8B teacher embeddings for the entire MSLS training set.

Produces a single .pt file mapping image path → L2-normalised embedding.
Run once before training to avoid repeated 8B-parameter forward passes.

Usage:
    python src/cache_teacher_embeddings.py \
        --msls_root /work/$USER/datasets/msls \
        --output /work/$USER/learnerpr/cache/teacher_embeddings.pt \
        --batch_size 16
"""

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from models.teacher import load_teacher, extract_embeddings


def parse_args():
    p = argparse.ArgumentParser(description="Cache teacher embeddings for MSLS.")
    p.add_argument("--msls_root", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--prompt", type=str, default="Describe this place.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="database",
                   help="'database', 'query', or 'both'")
    return p.parse_args()


def collect_image_paths(msls_root: str, split: str, subset: str) -> list[str]:
    """Gather all .jpg paths from the MSLS layout."""
    root = Path(msls_root)
    split_dir = root / ("train_val" if split in ("train", "val") else "test")
    paths = []
    subsets = ["database", "query"] if subset == "both" else [subset]
    for city_dir in sorted(split_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for sub in subsets:
            img_dir = city_dir / sub / "images"
            if img_dir.exists():
                paths.extend(str(p) for p in sorted(img_dir.glob("*.jpg")))
    return paths


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading teacher: {args.model_name}")
    model, processor = load_teacher(args.model_name)

    paths = collect_image_paths(args.msls_root, args.split, args.subset)
    print(f"Found {len(paths)} images to cache.")

    # Resume support: load existing cache if present
    cache: dict[str, torch.Tensor] = {}
    if os.path.exists(args.output):
        cache = torch.load(args.output, map_location="cpu")
        print(f"Resuming — {len(cache)} embeddings already cached.")
        paths = [p for p in paths if p not in cache]
        print(f"{len(paths)} remaining.")

    for i in tqdm(range(0, len(paths), args.batch_size), desc="Caching"):
        batch_paths = paths[i : i + args.batch_size]
        embeds = extract_embeddings(model, processor, batch_paths, prompt=args.prompt)
        for path, emb in zip(batch_paths, embeds):
            cache[path] = emb.cpu()

        # Periodic checkpoint every 500 batches
        if (i // args.batch_size) % 500 == 0 and i > 0:
            torch.save(cache, args.output)

    torch.save(cache, args.output)
    print(f"Saved {len(cache)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
