"""FARA — Full model assembly.

Assembles all FARA components into a single nn.Module following the
architecture in Fig. 1 of Luo et al. (IEEE/ACM TASLP 2026).

Pipeline:
  WavLM hidden states → LearnableMask → ┐
  Raw waveform → SincNet →              ├→ FeatureFusion → CMoE
                                              │
                              ├── BoundaryEnhance → BoundaryClassify
                              └── AttentionMask → Classify

Note: WavLM feature extraction is handled externally (frozen backbone).
This module expects pre-extracted hidden states as input.
"""

import torch
import torch.nn as nn

from fara.model.learnable_mask import LearnableMask
from fara.model.sincnet import SincNet
from fara.model.feature_fusion import FeatureFusion
from fara.model.cmoe import CMoE
from fara.model.boundary_enhance import (
    AttentionMask,
    BoundaryEnhance,
    ClassifyHead,
)


class FARA(nn.Module):
    """FARA: A Robust Region-Aware Framework for Audio Forgery Localization.

    Args:
        d_model: Feature dimension (default 1024, WavLM-Large).
        n_wavlm_layers: Number of WavLM transformer layers (default 24).
        mask_k: Number of layers to suppress in LearnableMask (default 12).
        sincnet_channels: SincNet filter bank size (default 80).
        sincnet_kernel: SincNet kernel size (default 251).
        sincnet_stride: SincNet stride (default 320 = 20ms at 16kHz).
        num_experts: Number of CMoE experts K (default 8).
        num_classes: Number of output classes (default 2).
        boundary_kernel: BoundaryEnhance conv kernel size (default 3).
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_wavlm_layers: int = 24,
        mask_k: int = 12,
        sincnet_channels: int = 80,
        sincnet_kernel: int = 251,
        sincnet_stride: int = 320,
        num_experts: int = 8,
        num_classes: int = 2,
        boundary_kernel: int = 3,
    ):
        super().__init__()

        # Layer selection from WavLM hidden states
        self.learnable_mask = LearnableMask(d_model, n_wavlm_layers, mask_k)

        # Frequency-domain feature extraction from raw waveform
        self.sincnet = SincNet(
            out_channels=sincnet_channels,
            kernel_size=sincnet_kernel,
            stride=sincnet_stride,
        )

        # Gated fusion of WavLM and SincNet features
        self.feature_fusion = FeatureFusion(d_model, sincnet_channels)

        # Cluster-Based Mixture of Experts
        self.cmoe = CMoE(d_model, num_experts)

        # Boundary enhancement branch
        self.boundary_enhance = BoundaryEnhance(d_model, boundary_kernel)
        self.boundary_classify = ClassifyHead(d_model, num_classes)

        # Spoof classification branch (with attention mask from boundary)
        self.attention_mask = AttentionMask()
        self.classify = ClassifyHead(d_model, num_classes)

    def forward(
        self,
        wavlm_hidden_states: torch.Tensor,
        waveform: torch.Tensor,
    ) -> dict:
        """Forward pass through the full FARA pipeline.

        Args:
            wavlm_hidden_states: (B, T, N, D) all WavLM transformer
                hidden states stacked on dim 2. N=24 layers, D=1024.
            waveform: (B, T_samples) raw audio at 16kHz.

        Returns:
            Dict with keys:
                spoof_logits: (B, T, num_classes) forgery localization logits.
                boundary_logits: (B, T, num_classes) boundary detection logits.
                cluster_assignments: (B, T) hard cluster IDs for GCL loss.
                fused_features: (B, T, D) for contrastive loss computation.
        """
        # 1. LearnableMask: select and weight WavLM layers
        wavlm_feat = self.learnable_mask(wavlm_hidden_states)  # (B, T, D)

        # 2. SincNet: extract frequency features from raw waveform
        sincnet_feat = self.sincnet(waveform)  # (B, T_frames, C)

        # Align frame counts (SincNet and WavLM may differ by ±1 frame)
        t_min = min(wavlm_feat.shape[1], sincnet_feat.shape[1])
        wavlm_feat = wavlm_feat[:, :t_min]
        sincnet_feat = sincnet_feat[:, :t_min]

        # 3. Feature Fusion: gate and combine streams
        fused = self.feature_fusion(wavlm_feat, sincnet_feat)  # (B, T, D)

        # 4. CMoE: cluster-based expert routing
        cmoe_out, assignments = self.cmoe(fused)  # (B, T, D), (B, T)

        # 5. Boundary branch
        boundary_feat = self.boundary_enhance(cmoe_out)  # (B, T, D)
        boundary_logits = self.boundary_classify(boundary_feat)  # (B, T, 2)

        # 6. Spoof branch with attention mask from boundary predictions
        masked_feat = self.attention_mask(cmoe_out, boundary_logits)
        spoof_logits = self.classify(masked_feat)  # (B, T, 2)

        return {
            "spoof_logits": spoof_logits,
            "boundary_logits": boundary_logits,
            "cluster_assignments": assignments,
            "fused_features": fused,
        }
