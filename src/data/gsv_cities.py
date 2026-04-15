"""GSV-Cities dataset loader for LearnerPR.

Expects the standard GSV-Cities directory layout (Kaggle release):

    gsv_cities_root/
        Images/
            <City>/
                <CityPrefix>_<placeID>_<year>_<month>_<bearing>_<lat>_<lon>_<panoid>.JPG
        Dataframes/
            <City>.csv     # columns: place_id, year, month, northdeg, city_id, lat, lon, panoid

place_ids are globally unique across cities via a stable rank-based offset:
    global_place_id = city_rank * _PLACE_OFFSET + local_place_id

With ~62k total places and ~40 cities, _PLACE_OFFSET=100_000 avoids collisions.
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


_PLACE_OFFSET = 100_000

# Matches the place_id (zero-padded 7 digits) in filenames:
# e.g. PRS_0000003_2015_05_... or Boston_0006385_2018_09_...
_FNAME_PLACE_RE = re.compile(r"_(\d{7})_\d{4}_\d{2}_")


def _parse_place_id_from_name(fname: str) -> int | None:
    m = _FNAME_PLACE_RE.search(fname)
    return int(m.group(1)) if m else None


def _city_rank_map(images_dir: Path) -> dict[str, int]:
    """Return a stable alphabetical rank for every city folder under images_dir."""
    if not images_dir.exists():
        return {}
    return {d.name: rank for rank, d in enumerate(sorted(images_dir.iterdir()))}


class GSVCitiesDataset(Dataset):
    """All images from the selected GSV-Cities cities.

    Each sample satisfies the same dict contract as MSLSDataset:
        image, place_id (globally unique int), key (basename), path (absolute str).

    Args:
        root:    Path to gsv_cities_root (parent of Images/ and Dataframes/).
        cities:  Restrict to these city folders; None = all available.
        transform: torchvision transform applied to each image.
    """

    def __init__(
        self,
        root: str,
        cities: list[str] | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform

        images_dir = self.root / "Images"
        rank_map = _city_rank_map(images_dir)
        selected = cities if cities is not None else sorted(rank_map.keys())

        self.image_paths: list[str] = []
        self.place_ids: list[int] = []
        self.keys: list[str] = []

        for city in selected:
            city_dir = images_dir / city
            if not city_dir.exists():
                continue
            rank = rank_map[city]
            for fpath in sorted(city_dir.glob("*.JPG")):
                local_pid = _parse_place_id_from_name(fpath.name)
                if local_pid is None:
                    continue
                self.image_paths.append(str(fpath.resolve()))
                self.place_ids.append(rank * _PLACE_OFFSET + local_pid)
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


def _make_static(paths, place_ids, keys, transform) -> "_StaticDataset":
    return _StaticDataset(paths, place_ids, keys, transform)


class _StaticDataset(Dataset):
    """Pre-assembled path/place_id/key list — used for GSV val db/query splits."""

    def __init__(self, paths, place_ids, keys, transform=None):
        self.image_paths = paths
        self.place_ids = place_ids
        self.keys = keys
        self.transform = transform

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


def make_gsv_val_splits(
    root: str,
    held_out_cities: list[str],
    transform=None,
) -> tuple[_StaticDataset, _StaticDataset]:
    """Build database and query datasets from held-out GSV-Cities cities.

    For each place_id the most recent image (last when sorted alphabetically by
    filename, which encodes year/month) becomes the query; all prior images form the
    database.  Places with only one image are skipped to guarantee non-empty databases.

    This is deterministic and mirrors the "unseen geography" evaluation spirit of MSLS:
    the model has never seen these cities during training.

    Returns:
        (db_dataset, query_dataset) — both yield the standard sample dict.
    """
    root_path = Path(root)
    images_dir = root_path / "Images"
    rank_map = _city_rank_map(images_dir)

    db_paths: list[str] = []
    db_pids: list[int] = []
    db_keys: list[str] = []
    q_paths: list[str] = []
    q_pids: list[int] = []
    q_keys: list[str] = []

    for city in held_out_cities:
        city_dir = images_dir / city
        if not city_dir.exists():
            continue
        rank = rank_map.get(city, 0)

        # Group images by local place_id
        place_to_files: dict[int, list[Path]] = defaultdict(list)
        for fpath in sorted(city_dir.glob("*.JPG")):
            local_pid = _parse_place_id_from_name(fpath.name)
            if local_pid is None:
                continue
            place_to_files[local_pid].append(fpath)

        for local_pid, fpaths in place_to_files.items():
            if len(fpaths) < 2:
                continue
            global_pid = rank * _PLACE_OFFSET + local_pid
            # Last file (most recent capture) → query; rest → database
            *db_fpaths, q_fpath = fpaths
            for p in db_fpaths:
                db_paths.append(str(p.resolve()))
                db_pids.append(global_pid)
                db_keys.append(p.name)
            q_paths.append(str(q_fpath.resolve()))
            q_pids.append(global_pid)
            q_keys.append(q_fpath.name)

    return (
        _make_static(db_paths, db_pids, db_keys, transform),
        _make_static(q_paths, q_pids, q_keys, transform),
    )


def auto_val_cities(root: str, fraction: float = 0.1) -> tuple[list[str], list[str]]:
    """Split available cities into (train_cities, val_cities) by held-out fraction.

    Cities are sorted alphabetically for determinism.  The last `fraction` are held
    out for validation (tail of the sorted list, so they are geographically diverse
    given the alphabetical spread of city names).

    Returns:
        (train_cities, val_cities)
    """
    images_dir = Path(root) / "Images"
    if not images_dir.exists():
        return [], []
    all_cities = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    n_val = max(1, int(len(all_cities) * fraction))
    return all_cities[:-n_val], all_cities[-n_val:]
