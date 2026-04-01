"""Boundary Enhancement, Classification Heads, and Attention Mask.

Implements the boundary-aware modules from FARA (Luo et al., IEEE/ACM
TASLP 2026, Section III):

- BoundaryEnhance: "a 1D convolution to map the CMoE output into a
  boundary-aware representation space"
- ClassifyHead: "an MLP followed by a linear layer" for both spoof
  and boundary classification
- AttentionMask: "Based on the Boundary Classify output, an Attention
  Mask is generated to guide the separation of attention across
  different regions"
"""

import torch
import torch.nn as nn


class BoundaryEnhance(nn.Module):
    """Boundary enhancement via 1D convolution.

    Maps CMoE output into a boundary-aware representation space using
    a 1D convolution that captures local temporal context around
    forgery boundaries.

    Args:
        d_model: Feature dimension (default 1024).
        kernel_size: Conv kernel size (default 3, captures ±1 frame context).
    """

    def __init__(self, d_model: int = 1024, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2  # Same-length output
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.act = nn.ReLU()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply boundary-aware convolution.

        Args:
            x: (B, T, D) CMoE output.

        Returns:
            out: (B, T, D) boundary-enhanced features.
        """
        # Conv1D expects (B, D, T)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        h = self.act(h)
        out = self.proj(h)  # (B, T, D)
        return out


class ClassifyHead(nn.Module):
    """2-layer MLP classification head.

    Used for both spoof classification and boundary classification.
    Paper: "sharing the same structure as the Boundary Classify module."

    Args:
        d_model: Input feature dimension (default 1024).
        num_classes: Number of output classes (default 2).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self, d_model: int = 1024, num_classes: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify each frame.

        Args:
            x: (B, T, D) input features.

        Returns:
            logits: (B, T, num_classes) classification logits.
        """
        return self.mlp(x)


class AttentionMask(nn.Module):
    """Attention mask from boundary predictions.

    Generates a soft attention mask from boundary classification logits
    to guide the spoof classifier to focus on regions around forgery
    boundaries.

    The boundary probability (class 1) is used as a multiplicative gate
    on the features before spoof classification.
    """

    def forward(
        self, features: torch.Tensor, boundary_logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply boundary-guided attention mask.

        Args:
            features: (B, T, D) features to be masked.
            boundary_logits: (B, T, 2) boundary classification logits.

        Returns:
            masked: (B, T, D) attention-modulated features.
        """
        # Boundary probability as attention weight
        # Use sigmoid on the boundary class logit for a smooth gate
        boundary_prob = torch.sigmoid(boundary_logits[:, :, 1])  # (B, T)

        # Additive bias: scale features but don't zero them out
        # mask = 1 + boundary_prob ensures all regions contribute
        mask = 1.0 + boundary_prob  # (B, T)

        return features * mask.unsqueeze(-1)  # (B, T, D)
