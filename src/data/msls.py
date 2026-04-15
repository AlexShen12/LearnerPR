"""MSLS dataset loader for LearnerPR.

Expects the standard MSLS directory layout:
    msls_root/
        train_val/
            <city>/
                query/images/
                database/images/
        ...

The official train/val/test splits are defined by CSV files shipped with MSLS.
"""

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MSLSDataset(Dataset):
    """Loads MSLS images for a given split with optional transform.

    For *training* with RKD we only need images (teacher embeddings come from
    the pre-computed cache).  Place IDs are carried along for the optional
    triplet loss and for evaluation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        subset: str = "database",
        transform=None,
        cities: list[str] | None = None,
    ):
        """
        Args:
            root: Path to the MSLS root (contains train_val/, test/).
            split: One of 'train', 'val', 'test'.
            subset: 'database' or 'query'.
            transform: torchvision transform applied to each image.
            cities: Restrict to specific cities; None = all in split.
        """
        self.root = Path(root)
        self.split = split
        self.subset = subset
        self.transform = transform

        split_dir = self.root / ("train_val" if split in ("train", "val") else "test")
        self.image_paths: list[str] = []
        self.place_ids: list[int] = []
        self.keys: list[str] = []

        available_cities = sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        ) if cities is None else cities

        for city in available_cities:
            img_dir = split_dir / city / subset / "images"
            if not img_dir.exists():
                continue
            postfix_csv = split_dir / city / subset / "postfix.csv"
            if postfix_csv.exists():
                df = pd.read_csv(postfix_csv)
                for _, row in df.iterrows():
                    fpath = img_dir / row["key"]
                    if fpath.exists():
                        self.image_paths.append(str(fpath))
                        self.place_ids.append(int(row.get("place_id", -1)))
                        self.keys.append(row["key"])
            else:
                for fpath in sorted(img_dir.glob("*.jpg")):
                    self.image_paths.append(str(fpath))
                    self.place_ids.append(-1)
                    self.keys.append(fpath.name)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "place_id": self.place_ids[idx],
            "key": self.keys[idx],
            "path": self.image_paths[idx],
        }


class TeacherCacheDataset(Dataset):
    """Wraps MSLSDataset and pairs each sample with its cached teacher embedding."""

    def __init__(self, msls_dataset: MSLSDataset, cache_path: str):
        self.msls = msls_dataset
        self.cache: dict[str, torch.Tensor] = torch.load(cache_path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.msls)

    def __getitem__(self, idx: int) -> dict:
        sample = self.msls[idx]
        teacher_embed = self.cache.get(sample["path"])
        if teacher_embed is None:
            teacher_embed = self.cache.get(sample["key"])
        if teacher_embed is None:
            raise KeyError(
                f"Teacher embedding not found for {sample['path']} / {sample['key']}. "
                "Re-run cache_teacher_embeddings.py."
            )
        sample["teacher_embed"] = teacher_embed
        return sample
