"""
COMP 560 course submission interface for `evaluate.py`.

Expected usage (from the provided `comp560-project-vpr` bundle):

    python evaluate.py \\
        --student_id <id> \\
        --model_path /path/to/this/model.py \\
        --datasets_root ./datasets

The harness imports this file and instantiates `StudentModel(device=...)`.

Weights: place a checkpoint next to this file as `submission_weights.pt`, or set
the environment variable `LEARNERPR_WEIGHTS` to a `.pt` file produced by
`src/train.py` (full checkpoint dict with a `"model"` key is accepted).
If no weights are found, the network uses DINOv2-Small ImageNet pretraining only
(GeM + optional layers untrained from init) — fine for debugging, not for final scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.student import StudentModel as _DinoGeMModule  # noqa: E402


class StudentModel:
    """Course-required API: device, encode(), embedding_dim."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize your model.

        Args:
            device: Target device ("cuda" or "cpu")
        """
        self._device = torch.device(
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )

        core = _DinoGeMModule(
            backbone_name="facebook/dinov2-small",
            embed_dim=384,
            gem_p_init=3.0,
            use_projection=False,
        )

        weights_path = os.environ.get(
            "LEARNERPR_WEIGHTS", str(_ROOT / "submission_weights.pt")
        )
        if os.path.isfile(weights_path):
            payload = torch.load(weights_path, map_location="cpu")
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            state = {k: v for k, v in state.items() if not k.startswith("projection")}
            core.load_state_dict(state, strict=False)

        self._core = core.to(self._device).eval()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: (B, 3, H, W) tensor, normalized with ImageNet stats
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

        Returns:
            (B, D) tensor of L2-normalized embeddings
        """
        images = images.to(self._device, non_blocking=True)
        with torch.inference_mode():
            return self._core(images)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension D."""
        return 384
