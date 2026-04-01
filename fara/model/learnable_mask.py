"""Learnable Mask — differentiable layer selection for WavLM-Large.

Implements the noisy top-k gating mechanism from FARA (Luo et al., IEEE/ACM
TASLP 2026, Section III-A), based on Shazeer et al. (2017, ICLR) "Outrageously
Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer".

Given all N=24 WavLM hidden states x ∈ R^{B×T×N×D}:
  1. Compute per-frame, per-layer importance via learned w_g: score = x · w_g
  2. Add Gaussian noise scaled by Softplus(x · w_noise) during training
  3. Suppress the k smallest scores to -inf (hard top-K sparsity)
  4. Apply softmax over the layer dimension to get mask weights
  5. Return weighted combination: output ∈ R^{B×T×D}

NOTE: The paper does not specify the value of k. We default to k=12
(suppress half of 24 layers) as a reasonable starting point. This is an
ENGINEERING ASSUMPTION to be validated on the dev set. Sweep: {8, 10, 12, 14}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMask(nn.Module):
    """Differentiable layer selection via noisy top-k gating.

    Args:
        d_model: Feature dimension per layer (1024 for WavLM-Large).
        n_layers: Number of transformer layers (24 for WavLM-Large).
        k: Number of layers to suppress per frame (default 12).
    """

    def __init__(self, d_model: int = 1024, n_layers: int = 24, k: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.k = k

        # Shared linear projections (no bias) — produce scalar score per (frame, layer)
        self.w_g = nn.Linear(d_model, 1, bias=False)      # importance gate
        self.w_noise = nn.Linear(d_model, 1, bias=False)   # noise scale

    def _compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gated scores with top-k suppression (no noise).

        This is exposed for testing the internal mask state.

        Args:
            x: (B, T, N, D) — all hidden states stacked on dim 2.

        Returns:
            scores: (B, T, N) — with k smallest set to -inf.
        """
        # Raw importance scores: (B, T, N, 1) -> (B, T, N)
        scores = self.w_g(x).squeeze(-1)

        # Top-k suppression: keep (N - k) largest, set rest to -inf
        keep = self.n_layers - self.k
        # topk returns (values, indices) for the largest `keep` entries
        _, top_indices = scores.topk(keep, dim=-1)  # (B, T, keep)

        # Build suppressed scores
        suppressed = torch.full_like(scores, float("-inf"))
        suppressed.scatter_(-1, top_indices, scores.gather(-1, top_indices))

        return suppressed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: select and weight layers.

        Args:
            x: (B, T, N, D) — stacked WavLM hidden states.

        Returns:
            out: (B, T, D) — weighted layer combination.
        """
        # Raw importance scores: (B, T, N)
        scores = self.w_g(x).squeeze(-1)

        # Add noise during training (Shazeer et al., 2017)
        if self.training:
            # Clamp softplus to prevent float16 exp() overflow (max ~4.0)
            noise_scale = F.softplus(self.w_noise(x).squeeze(-1)).clamp(max=4.0)
            noise = torch.randn_like(scores) * noise_scale
            scores = scores + noise

        # Top-k suppression: keep (N - k) largest, set rest to -inf
        keep = self.n_layers - self.k
        _, top_indices = scores.topk(keep, dim=-1)  # (B, T, keep)

        suppressed = torch.full_like(scores, float("-inf"))
        suppressed.scatter_(-1, top_indices, scores.gather(-1, top_indices))

        # Softmax weights over layers
        weights = torch.softmax(suppressed, dim=-1)  # (B, T, N)

        # Weighted combination: (B, T, N, 1) * (B, T, N, D) -> sum over N -> (B, T, D)
        out = (weights.unsqueeze(-1) * x).sum(dim=2)

        return out
