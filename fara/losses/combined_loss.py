"""Combined training loss for FARA.

Reference: Luo et al., "A Robust Region-Aware Framework for Audio Forgery
Localization", IEEE/ACM TASLP 2026, Section III-C, Equation 8.

L_train = L_spoof + 0.5 * L_boundary + 0.2 * L_CRL

L_spoof:    CrossEntropyLoss on spoof_logits vs frame_labels
L_boundary: CrossEntropyLoss on boundary_logits vs boundary_labels
L_CRL:      Group Contrastive Loss on fused features
"""
from typing import Dict

import torch
import torch.nn as nn

from fara.losses.group_contrastive import GroupContrastiveLoss


class CombinedLoss(nn.Module):
    """Combined FARA training loss (Eq. 8).

    Computes weighted sum of spoof classification loss, boundary detection
    loss, and group contrastive representation loss.

    Args:
        spoof_weight: Weight for spoof classification loss. Default 1.0.
        boundary_weight: Weight for boundary detection loss. Default 0.5.
        crl_weight: Weight for group contrastive loss. Default 0.2.
        beta: Margin parameter for GCL negative pairs. Default 0.3.
        ignore_index: Label index to ignore in CE losses (padding). Default -1.
    """

    def __init__(
        self,
        spoof_weight: float = 1.0,
        boundary_weight: float = 0.5,
        crl_weight: float = 0.2,
        beta: float = 0.3,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.spoof_weight = spoof_weight
        self.boundary_weight = boundary_weight
        self.crl_weight = crl_weight

        self.ce_spoof = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_boundary = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.gcl = GroupContrastiveLoss(beta=beta, ignore_index=ignore_index)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        frame_labels: torch.Tensor,
        boundary_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined training loss.

        Args:
            model_output: Dict with keys:
                - 'spoof_logits': (B, T, 2) spoof classification logits.
                - 'boundary_logits': (B, T, 2) boundary detection logits.
                - 'fused_features': (B, T, D) fused frame features for GCL.
                - 'cluster_assignments': (B, T) hard cluster indices from CMoE.
            frame_labels: (B, T) ground truth frame labels.
                0=bonafide, 1=spoof, -1=padding.
            boundary_labels: (B, T) ground truth boundary labels.
                0=non-boundary, 1=boundary, -1=padding.

        Returns:
            Dict with keys: 'loss', 'loss_spoof', 'loss_boundary', 'loss_crl'.
            All values are scalar tensors.
        """
        spoof_logits = model_output["spoof_logits"]          # (B, T, 2)
        boundary_logits = model_output["boundary_logits"]    # (B, T, 2)
        fused_features = model_output["fused_features"]      # (B, T, D)
        cluster_assignments = model_output["cluster_assignments"]  # (B, T)

        B, T, C = spoof_logits.shape

        # Reshape for CrossEntropyLoss: (B*T, C) and (B*T,)
        loss_spoof = self.ce_spoof(
            spoof_logits.reshape(B * T, C),
            frame_labels.reshape(B * T),
        )

        loss_boundary = self.ce_boundary(
            boundary_logits.reshape(B * T, C),
            boundary_labels.reshape(B * T),
        )

        # Group contrastive loss (Eq. 5-7)
        loss_crl = self.gcl(fused_features, frame_labels, cluster_assignments)

        # Combined loss (Eq. 8)
        total = (
            self.spoof_weight * loss_spoof
            + self.boundary_weight * loss_boundary
            + self.crl_weight * loss_crl
        )

        return {
            "loss": total,
            "loss_spoof": loss_spoof,
            "loss_boundary": loss_boundary,
            "loss_crl": loss_crl,
        }
