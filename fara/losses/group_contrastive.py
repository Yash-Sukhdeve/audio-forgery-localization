"""Group Contrastive Loss (GCL) for FARA.

Reference: Luo et al., "A Robust Region-Aware Framework for Audio Forgery
Localization", IEEE/ACM TASLP 2026, Section III-C, Equations 5-7.

Eq. 5: Sim(f_i, f_j) = (f_i . f_j) / (||f_i|| * ||f_j||)
Eq. 6: Diff(f) = I*(1 - Sim(f, f+))^2 + (1-I)*(max(0, Sim(f, f-) - beta))^2
Eq. 7: L = (1/G) * sum_g (1/J_g) * sum_j Diff(f_{g,j})

Groups are defined by CMoE cluster assignments. Within each group, frames
sharing the same label are pulled together (positive pairs) while frames
with different labels are pushed apart with margin beta.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupContrastiveLoss(nn.Module):
    """Group Contrastive Representation Loss (Eq. 5-7).

    Args:
        beta: Edge/margin parameter for negative pairs (default 0.3).
            Negative pairs with cosine similarity below beta incur zero loss.
        max_samples: Maximum number of valid frames to use per batch.
            When N > max_samples, randomly sample to avoid O(N^2) blowup.
            Default 2048.
        ignore_index: Label value to ignore (padding). Default -1.
    """

    def __init__(
        self,
        beta: float = 0.3,
        max_samples: int = 2048,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.beta = beta
        self.max_samples = max_samples
        self.ignore_index = ignore_index

    def forward(
        self,
        features: torch.Tensor,
        frame_labels: torch.Tensor,
        cluster_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Compute group contrastive loss.

        Args:
            features: Frame-level features, shape (B, T, D).
            frame_labels: Ground truth per frame, shape (B, T).
                0=bonafide, 1=spoof, -1=padding (ignored).
            cluster_assignments: Hard cluster index per frame, shape (B, T).
                Integer values in [0, K-1] from CMoE routing.

        Returns:
            Scalar loss tensor. Returns 0 if no valid frames exist.
        """
        B, T, D = features.shape

        # Flatten across batch and time
        flat_features = features.reshape(-1, D)           # (B*T, D)
        flat_labels = frame_labels.reshape(-1)             # (B*T,)
        flat_clusters = cluster_assignments.reshape(-1)    # (B*T,)

        # Exclude padding
        valid_mask = flat_labels != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        valid_features = flat_features[valid_mask]    # (N, D)
        valid_labels = flat_labels[valid_mask]         # (N,)
        valid_clusters = flat_clusters[valid_mask]     # (N,)

        N = valid_features.shape[0]

        # Subsample if too many valid frames to keep O(N^2) tractable
        if N > self.max_samples:
            perm = torch.randperm(N, device=features.device)[:self.max_samples]
            valid_features = valid_features[perm]
            valid_labels = valid_labels[perm]
            valid_clusters = valid_clusters[perm]
            N = self.max_samples

        # Force float32 for normalize + matmul to prevent float16 overflow
        # (1024-dim dot products overflow float16 range)
        with torch.cuda.amp.autocast(enabled=False):
            valid_f32 = valid_features.float()
            normed = F.normalize(valid_f32, p=2, dim=1)  # (N, D)
            sim_matrix = torch.mm(normed, normed.t())  # (N, N)

        # Build masks: same group and same/different label
        same_group = valid_clusters.unsqueeze(0) == valid_clusters.unsqueeze(1)  # (N, N)
        same_label = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)      # (N, N)

        # Exclude self-pairs
        eye = torch.eye(N, dtype=torch.bool, device=features.device)
        same_group = same_group & ~eye

        # Positive pairs: same group AND same label (Eq. 6, I=1 term)
        pos_mask = same_group & same_label       # (N, N)
        # Negative pairs: same group AND different label (Eq. 6, I=0 term)
        neg_mask = same_group & ~same_label      # (N, N)

        # --- Compute loss per group (Eq. 7) ---
        unique_groups = valid_clusters.unique()
        G = unique_groups.numel()

        if G == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=features.device)

        for g in unique_groups:
            group_idx = (valid_clusters == g).nonzero(as_tuple=True)[0]
            J_g = group_idx.numel()

            if J_g < 2:
                continue

            # Extract sub-matrices for this group
            group_sim = sim_matrix[group_idx][:, group_idx]           # (J_g, J_g)
            group_pos = pos_mask[group_idx][:, group_idx]             # (J_g, J_g)
            group_neg = neg_mask[group_idx][:, group_idx]             # (J_g, J_g)

            # Positive pair loss: (1 - sim)^2 for same-label pairs
            pos_loss = torch.tensor(0.0, device=features.device)
            if group_pos.any():
                pos_sims = group_sim[group_pos]
                pos_loss = ((1.0 - pos_sims) ** 2).mean()

            # Negative pair loss: max(0, sim - beta)^2 for cross-label pairs
            neg_loss = torch.tensor(0.0, device=features.device)
            if group_neg.any():
                neg_sims = group_sim[group_neg]
                neg_loss = (F.relu(neg_sims - self.beta) ** 2).mean()

            # Average Diff over frames in group (Eq. 7 inner sum)
            group_loss = pos_loss + neg_loss
            total_loss = total_loss + group_loss / G

        return total_loss
