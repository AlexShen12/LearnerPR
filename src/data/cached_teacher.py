"""CachedTeacherDataset — wraps any base dataset with pre-computed teacher embeddings.

The base dataset must return dicts containing at least `path` and `key`.  Embeddings
are looked up by `path` first, then `key`, matching the convention used everywhere in
the codebase.
"""

import torch
from torch.utils.data import Dataset


class CachedTeacherDataset(Dataset):
    """Pairs each sample from *base_dataset* with its cached teacher embedding.

    Args:
        base_dataset: Any Dataset whose items include `path` and `key`.
        cache_path:   Path to a .pt file mapping path/key strings to L2-normalised
                      embedding tensors (produced by cache_teacher_embeddings.py).
    """

    def __init__(self, base_dataset: Dataset, cache_path: str):
        self.base = base_dataset
        self.cache: dict[str, torch.Tensor] = torch.load(
            cache_path, map_location="cpu", weights_only=True
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        embed = self.cache.get(sample["path"]) or self.cache.get(sample["key"])
        if embed is None:
            raise KeyError(
                f"Teacher embedding not found for {sample['path']} / {sample['key']}. "
                "Re-run cache_teacher_embeddings.py."
            )
        sample["teacher_embed"] = embed
        return sample
