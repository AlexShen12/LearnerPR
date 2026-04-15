"""Cache Qwen3-VL-8B teacher embeddings for MSLS or GSV-Cities.

Produces a single .pt file mapping absolute image path → L2-normalised embedding.
Run once per dataset before training.  Resume is supported: re-submitting the job
will skip paths already present in the output file.

Usage (MSLS):
    python src/cache_teacher_embeddings.py \
        --dataset msls \
        --msls_root /work/$USER/datasets/msls \
        --output /work/$USER/learnerpr/cache/teacher_embeddings_msls.pt \
        --batch_size 16

Usage (GSV-Cities):
    python src/cache_teacher_embeddings.py \
        --dataset gsv_cities \
        --gsv_cities_root /work/$USER/datasets/gsv-cities \
        --output /work/$USER/learnerpr/cache/teacher_embeddings_gsv.pt \
        --batch_size 16
"""

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from models.teacher import load_teacher, extract_embeddings


def parse_args():
    p = argparse.ArgumentParser(
        description="Cache teacher embeddings for MSLS or GSV-Cities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="msls",
        choices=["msls", "gsv_cities"],
        help="Which dataset to cache embeddings for.",
    )
    # MSLS args
    p.add_argument("--msls_root", type=str, default=None,
                   help="MSLS root directory (required when --dataset msls).")
    p.add_argument("--split", type=str, default="train",
                   help="MSLS split: train, val, or test.")
    p.add_argument("--subset", type=str, default="both",
                   help="MSLS subset: database, query, or both.")
    # GSV-Cities args
    p.add_argument("--gsv_cities_root", type=str, default=None,
                   help="GSV-Cities root directory (required when --dataset gsv_cities).")
    p.add_argument("--gsv_cities", type=str, default=None,
                   help="Comma-separated list of GSV city folders to cache; None = all.")
    # Shared
    p.add_argument("--output", type=str, required=True,
                   help="Path to output .pt cache file.")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--prompt", type=str, default="Describe this place.")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def collect_msls_paths(root: str, split: str, subset: str) -> list[str]:
    """Gather all .jpg paths from the MSLS directory layout."""
    split_dir = Path(root) / ("train_val" if split in ("train", "val") else "test")
    subsets = ["database", "query"] if subset == "both" else [subset]
    paths: list[str] = []
    for city_dir in sorted(split_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for sub in subsets:
            img_dir = city_dir / sub / "images"
            if img_dir.exists():
                paths.extend(str(p.resolve()) for p in sorted(img_dir.glob("*.jpg")))
    return paths


def collect_gsv_cities_paths(root: str, cities: list[str] | None = None) -> list[str]:
    """Gather all .JPG paths from the GSV-Cities Images/ directory."""
    images_dir = Path(root) / "Images"
    available = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    selected = cities if cities is not None else available
    paths: list[str] = []
    for city in selected:
        city_dir = images_dir / city
        if city_dir.exists():
            paths.extend(str(p.resolve()) for p in sorted(city_dir.glob("*.JPG")))
    return paths


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Collect paths ────────────────────────────────────────────────
    if args.dataset == "msls":
        if not args.msls_root:
            raise ValueError("--msls_root is required when --dataset msls")
        paths = collect_msls_paths(args.msls_root, args.split, args.subset)
        print(f"Dataset: MSLS  split={args.split}  subset={args.subset}")
    else:
        if not args.gsv_cities_root:
            raise ValueError("--gsv_cities_root is required when --dataset gsv_cities")
        cities = [c.strip() for c in args.gsv_cities.split(",")] if args.gsv_cities else None
        paths = collect_gsv_cities_paths(args.gsv_cities_root, cities)
        label = ", ".join(cities) if cities else "all cities"
        print(f"Dataset: GSV-Cities  cities={label}")

    print(f"Found {len(paths)} images to cache.")

    # ── Load teacher ─────────────────────────────────────────────────
    print(f"Loading teacher: {args.model_name}")
    model, processor = load_teacher(args.model_name)

    # ── Resume support ───────────────────────────────────────────────
    cache: dict[str, torch.Tensor] = {}
    if os.path.exists(args.output):
        cache = torch.load(args.output, map_location="cpu")
        print(f"Resuming — {len(cache)} embeddings already cached.")
        paths = [p for p in paths if p not in cache]
        print(f"{len(paths)} remaining.")

    # ── Embed ────────────────────────────────────────────────────────
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
