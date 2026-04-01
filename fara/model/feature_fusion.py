"""Feature Fusion — gating mechanism for WavLM and SincNet features.

Implements the feature fusion module from FARA (Luo et al., IEEE/ACM
TASLP 2026, Section III): "The Feature Fusion module uses a gating
mechanism to balance frequency-domain features and pre-trained outputs,
followed by a linear layer to generate robust fused representations."

Architecture:
  1. Project SincNet features (80-dim) to match WavLM dim (1024)
  2. Learned per-dimension sigmoid gate balances the two streams
  3. Final linear layer produces fused representation

Input:  wavlm_feat (B, T, 1024), sincnet_feat (B, T, 80)
Output: fused (B, T, 1024)
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Gated fusion of WavLM and SincNet feature streams.

    Args:
        d_model: WavLM feature dimension (default 1024).
        sincnet_dim: SincNet output dimension (default 80).
    """

    def __init__(self, d_model: int = 1024, sincnet_dim: int = 80):
        super().__init__()
        self.d_model = d_model

        # Project SincNet features to match WavLM dimension
        self.sincnet_proj = nn.Linear(sincnet_dim, d_model)

        # Learned gating parameter (per-dimension)
        # Initialized to 0 so sigmoid(0) = 0.5 (equal weighting at start)
        self.gate = nn.Parameter(torch.zeros(d_model))

        # Final fusion projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, wavlm_feat: torch.Tensor, sincnet_feat: torch.Tensor
    ) -> torch.Tensor:
        """Fuse WavLM and SincNet features via learned gating.

        Args:
            wavlm_feat: (B, T, d_model) from LearnableMask output.
            sincnet_feat: (B, T, sincnet_dim) from SincNet output.

        Returns:
            fused: (B, T, d_model) fused representation.
        """
        # Project SincNet to WavLM dimension
        sincnet_proj = self.sincnet_proj(sincnet_feat)  # (B, T, d_model)

        # Sigmoid gate: per-dimension balance between streams
        g = torch.sigmoid(self.gate)  # (d_model,)

        # Gated combination
        fused = g * wavlm_feat + (1.0 - g) * sincnet_proj  # (B, T, d_model)

        # Final projection
        fused = self.out_proj(fused)  # (B, T, d_model)

        return fused
