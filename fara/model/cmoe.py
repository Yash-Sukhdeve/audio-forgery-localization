"""Cluster-Based Mixture of Experts (CMoE) module.

Implements the CMoE framework from FARA (Luo et al., IEEE/ACM TASLP 2026,
Section III-B, Eq. 2-4).

The CMoE module:
  1. Maintains K cluster centroids updated via EMA (Eq. 2, alpha=0.1)
  2. Routes each frame to experts based on L2 distance to centroids (Eq. 3)
  3. Combines expert outputs via weighted summation (Eq. 4)

Uses PyTorch-native K-means (no FAISS dependency) for centroid computation.

Input:  (B, T, D) fused features
Output: (B, T, D) expert-weighted output, (B, T) cluster assignments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batch_kmeans(
    x: torch.Tensor, k: int, n_iters: int = 5
) -> torch.Tensor:
    """Simple K-means on a batch of vectors. Returns centroids.

    All computations forced to float32 to avoid AMP float16 overflow
    in torch.cdist (PyTorch issue #57109) and scatter_add_ precision loss.

    Args:
        x: (N, D) feature vectors.
        k: Number of clusters.
        n_iters: Number of Lloyd iterations.

    Returns:
        centroids: (K, D) cluster centers in float32.
    """
    # Force float32 — cdist overflows in float16 with d=1024
    x = x.float()
    n = x.shape[0]
    if n <= k:
        pad = torch.zeros(k - n, x.shape[1], device=x.device, dtype=torch.float32)
        return torch.cat([x, pad], dim=0)

    # Random initialization: pick k random points
    indices = torch.randperm(n, device=x.device)[:k]
    centroids = x[indices].clone()

    for _ in range(n_iters):
        # Assign each point to nearest centroid (float32 cdist)
        dists = torch.cdist(x.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        assignments = dists.argmin(dim=1)  # (N,)

        # Recompute centroids (explicit float32 counts)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=x.device, dtype=torch.float32)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand_as(x), x)
        counts.scatter_add_(0, assignments, torch.ones(n, device=x.device, dtype=torch.float32))

        # Avoid division by zero for empty clusters
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids

    return centroids


class CMoERouter(nn.Module):
    """Cluster-based routing layer for CMoE.

    Maintains K cluster centroids as a buffer, updated via exponential
    moving average (Eq. 2: F_c = (1-alpha)*F_c + alpha*F_bar_c).

    Routes frames to experts by computing softmax over negative L2
    distances to centroids (Eq. 3).

    Args:
        d_model: Feature dimension (default 1024).
        num_clusters: Number of clusters K (default 8).
        ema_alpha: EMA update rate for centroids (default 0.1).
    """

    def __init__(
        self, d_model: int = 1024, num_clusters: int = 8, ema_alpha: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.ema_alpha = ema_alpha

        # Centroids as buffer (not a parameter — updated via EMA, not gradient)
        self.register_buffer(
            "centroids", torch.randn(num_clusters, d_model) * 0.01
        )
        self.register_buffer("initialized", torch.tensor(False))

    def _update_centroids(self, x_flat: torch.Tensor) -> None:
        """Update centroids via EMA with K-means on current batch (Eq. 2).

        All operations forced to float32 to prevent AMP-induced NaN.
        Includes NaN sentinel to prevent permanent centroid corruption.
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            batch_centroids = _batch_kmeans(x_flat.float(), self.num_clusters)

            # NaN sentinel: if K-means produced NaN, skip this update
            if torch.isnan(batch_centroids).any():
                return

            if not self.initialized:
                self.centroids.copy_(batch_centroids)
                self.initialized.fill_(True)
            else:
                # EMA update: F_c = (1 - alpha) * F_c + alpha * F_bar_c
                self.centroids.mul_(1.0 - self.ema_alpha).add_(
                    batch_centroids, alpha=self.ema_alpha
                )

    def forward(self, x: torch.Tensor) -> tuple:
        """Route frames to experts based on centroid distances.

        Args:
            x: (B, T, D) fused features.

        Returns:
            weights: (B, T, K) soft expert assignment weights (Eq. 3).
            assignments: (B, T) hard cluster assignments (argmax).
        """
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)  # (B*T, D)

        # Update centroids during training
        if self.training:
            self._update_centroids(x_flat)

        # Compute L2 distance in float32 to avoid cdist float16 overflow
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x_flat.float()
            c_f32 = self.centroids.float()
            dists = torch.cdist(x_f32.unsqueeze(0), c_f32.unsqueeze(0))
            dists = dists.squeeze(0)  # (B*T, K)

            # Eq. 3: W_{i,j} = softmax(-sqrt(sum((x_i - c_j)^2)))
            weights = F.softmax(-dists, dim=-1)  # (B*T, K)
            assignments = dists.argmin(dim=-1)  # (B*T,)

        return weights.view(b, t, -1), assignments.view(b, t)


class CMoEExpert(nn.Module):
    """Single CMoE expert — feed-forward network.

    Architecture: Linear(d, 4d) → GELU → Linear(4d, d)
    Standard transformer FFN expansion ratio.

    Args:
        d_model: Feature dimension (default 1024).
        expansion: FFN expansion factor (default 4).
    """

    def __init__(self, d_model: int = 1024, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CMoE(nn.Module):
    """Cluster-Based Mixture of Experts module.

    Combines router (clustering-based) with K expert networks.
    Output: weighted sum of expert outputs (Eq. 4).

    Args:
        d_model: Feature dimension (default 1024).
        num_experts: Number of experts K (default 8).
        ema_alpha: EMA rate for centroid updates (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_experts: int = 8,
        ema_alpha: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router = CMoERouter(d_model, num_experts, ema_alpha)
        self.experts = nn.ModuleList(
            [CMoEExpert(d_model) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through CMoE.

        Args:
            x: (B, T, D) fused features.

        Returns:
            output: (B, T, D) weighted expert output (Eq. 4).
            assignments: (B, T) hard cluster assignments for GCL.
        """
        weights, assignments = self.router(x)  # (B,T,K), (B,T)

        # Compute all expert outputs and weight them
        # Eq. 4: O_i = sum_j W_{i,j} * E_j(x_i)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-2
        )  # (B, T, K, D)

        # Weighted combination: (B,T,K,1) * (B,T,K,D) → sum over K → (B,T,D)
        output = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)

        return output, assignments
