"""P×K place-balanced batch sampler for VPR training.

Each batch contains exactly P distinct places with K images each (N = P*K total).
This guarantees at least one positive pair per anchor, which is required for
batch-hard triplet mining.

Usage:
    sampler = PKPlaceBatchSampler(place_ids=dataset.place_ids, P=16, K=4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
"""

import math
import random
from collections import defaultdict
from typing import Iterator


class PKPlaceBatchSampler:
    """Yields batches of P*K indices: P places, K images per place.

    Places with fewer than K images are silently excluded. The sampler
    reshuffles its eligible place pool each epoch, so batches vary across epochs.

    Args:
        place_ids: Sequence of integer place IDs, one per dataset index.
        P: Number of places per batch.
        K: Number of images per place per batch.
    """

    def __init__(self, place_ids: list[int], P: int, K: int) -> None:
        if P < 1 or K < 2:
            raise ValueError(f"P must be >= 1 and K must be >= 2, got P={P}, K={K}")

        self.P = P
        self.K = K

        # Group dataset indices by place_id.
        place_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, pid in enumerate(place_ids):
            place_to_indices[pid].append(idx)

        # Keep only places with enough images for at least one full K-sample.
        self._eligible: dict[int, list[int]] = {
            pid: idxs
            for pid, idxs in place_to_indices.items()
            if len(idxs) >= K
        }

        n_eligible = len(self._eligible)
        if n_eligible < P:
            raise ValueError(
                f"P×K sampler needs at least P={P} eligible places (>= {K} images each), "
                f"but only {n_eligible} places qualify."
            )

        # Stable epoch length: how many complete P×K batches fit the eligible pool.
        total_usable = sum(len(v) for v in self._eligible.values())
        self._num_batches = max(1, math.floor(total_usable / (P * K)))

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        place_ids = list(self._eligible.keys())

        for _ in range(self._num_batches):
            selected_places = random.sample(place_ids, self.P)
            batch: list[int] = []
            for pid in selected_places:
                batch.extend(random.sample(self._eligible[pid], self.K))
            yield batch
