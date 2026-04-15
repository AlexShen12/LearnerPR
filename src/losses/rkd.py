"""Relational Knowledge Distillation + optional triplet loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKDLoss(nn.Module):
    """Distance-wise Relational Knowledge Distillation.

    Penalises differences between the student's and teacher's pairwise
    cosine-similarity matrices, optionally with temperature scaling on
    the teacher side to soften/sharpen the target distribution.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, student_embeds: torch.Tensor, teacher_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_embeds: (N, D_s) L2-normalised student descriptors.
            teacher_embeds: (N, D_t) L2-normalised teacher descriptors.
        Returns:
            Scalar RKD loss.
        """
        sim_s = student_embeds @ student_embeds.t()
        sim_t = (teacher_embeds @ teacher_embeds.t()) / self.temperature

        # Huber loss is more robust than MSE to outlier pairs
        return F.smooth_l1_loss(sim_s, sim_t)


class TripletLoss(nn.Module):
    """Standard triplet margin loss on student embeddings."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)


class CombinedLoss(nn.Module):
    """Weighted sum of RKD and triplet losses."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        rkd_temperature: float = 0.1,
        triplet_margin: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.rkd = RKDLoss(temperature=rkd_temperature)
        self.triplet = TripletLoss(margin=triplet_margin)

    def forward(
        self,
        student_embeds: torch.Tensor,
        teacher_embeds: torch.Tensor,
        anchors: torch.Tensor | None = None,
        positives: torch.Tensor | None = None,
        negatives: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_rkd = self.rkd(student_embeds, teacher_embeds)
        total = self.alpha * loss_rkd

        loss_triplet = torch.tensor(0.0, device=student_embeds.device)
        if anchors is not None and positives is not None and negatives is not None:
            loss_triplet = self.triplet(anchors, positives, negatives)
            total = total + self.beta * loss_triplet

        return {"total": total, "rkd": loss_rkd, "triplet": loss_triplet}
