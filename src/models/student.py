"""DINOv2-Small student with GeM pooling and optional projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GeM(nn.Module):
    """Generalized Mean Pooling.

    Pools spatial/token features by computing the generalized mean with a
    learnable exponent p.  Higher p → the pooling attends more to
    high-activation (salient) patches instead of averaging uniformly.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) patch tokens → (B, D) pooled descriptor."""
        x = x.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=1).pow(1.0 / self.p)


class DINOv2GeMStudent(nn.Module):
    """Trainable DINOv2-S + GeM encoder (internal backbone).

    For COMP 560 course submission use the root-level ``model.py`` ``StudentModel``
    wrapper (``encode()``, ``embedding_dim``) with ``evaluate.py``.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        embed_dim: int = 384,
        gem_p_init: float = 3.0,
        use_projection: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.gem = GeM(p=gem_p_init)
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    # ── Freezing helpers ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        blocks = self.backbone.encoder.layer
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) preprocessed images.
        Returns:
            (B, embed_dim) L2-normalised descriptors.
        """
        patch_tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        # DINOv2 prepends a CLS token — skip it for GeM over patches only
        patch_tokens = patch_tokens[:, 1:, :]

        desc = self.gem(patch_tokens)
        if self.use_projection:
            desc = self.projection(desc)
        return F.normalize(desc, p=2, dim=-1)
