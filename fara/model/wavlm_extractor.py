"""WavLM-Large feature extractor — frozen backbone for FARA.

Loads WavLM-Large via s3prl and extracts all 24 transformer hidden states
per frame. The backbone is frozen (no gradients) during FARA training.

WavLM-Large architecture (Chen et al., IEEE JSTSP 2022):
  - 24 transformer layers, each producing 1024-dim frame-level features
  - Input: raw waveform at 16kHz
  - Frame rate: 20ms (stride=320 samples)
  - Output: list of 25 tensors (CNN output + 24 transformer layers)
  - We use layers 1-24 (transformer only, skip CNN layer 0)

Reference for FARA usage: Luo et al., IEEE/ACM TASLP 2026, Section III-A.
"""

import torch
import torch.nn as nn


class WavLMExtractor(nn.Module):
    """Frozen WavLM-Large feature extractor.

    Extracts all 24 transformer hidden states and stacks them for the
    LearnableMask module.

    Args:
        checkpoint: Path to WavLM-Large .pt checkpoint.
        n_layers: Number of transformer layers to extract (default 24).
        freeze: Whether to freeze backbone weights (default True).
    """

    def __init__(
        self,
        checkpoint: str,
        n_layers: int = 24,
        freeze: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Load via s3prl hub
        import s3prl.hub as hub
        self.backbone = hub.wavlm_local(ckpt=checkpoint, fairseq=True)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract all transformer hidden states from raw waveform.

        Args:
            waveform: (B, T_samples) raw audio at 16kHz.

        Returns:
            hidden_states: (B, T_frames, N, D) where N=24 transformer
                layers and D=1024. T_frames ≈ T_samples // 320.
        """
        # s3prl returns dict with "hidden_states": list of (B, T, D)
        # Index 0 = CNN output, indices 1-24 = transformer layers
        output = self.backbone(waveform)
        all_states = output["hidden_states"]

        # Take transformer layers only (skip CNN layer 0)
        transformer_states = all_states[1 : self.n_layers + 1]

        # Stack: list of (B, T, D) → (B, T, N, D)
        stacked = torch.stack(transformer_states, dim=2)

        return stacked
