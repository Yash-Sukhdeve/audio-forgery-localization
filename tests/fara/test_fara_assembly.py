"""Tests for full FARA model assembly.

Reference: Luo et al., IEEE/ACM TASLP 2026, Fig. 1.
"""
import torch

from fara.model.fara import FARA


class TestFARA:
    """Tests for the full FARA model forward pass."""

    def setup_method(self):
        # Use small dimensions for fast testing
        self.model = FARA(
            d_model=64,
            n_wavlm_layers=4,
            mask_k=2,
            sincnet_channels=16,
            sincnet_kernel=31,
            sincnet_stride=320,
            num_experts=2,
            num_classes=2,
            boundary_kernel=3,
        )

    def test_forward_pass_shapes(self):
        """Verify all output tensor shapes from a full forward pass."""
        B, T_frames, N, D = 2, 50, 4, 64
        T_samples = T_frames * 320  # stride=320

        wavlm_states = torch.randn(B, T_frames, N, D)
        waveform = torch.randn(B, T_samples)

        output = self.model(wavlm_states, waveform)

        # Output keys
        assert "spoof_logits" in output
        assert "boundary_logits" in output
        assert "cluster_assignments" in output
        assert "fused_features" in output

        # Shape checks (T may be slightly different due to SincNet alignment)
        T_out = output["spoof_logits"].shape[1]
        assert output["spoof_logits"].shape == (B, T_out, 2)
        assert output["boundary_logits"].shape == (B, T_out, 2)
        assert output["cluster_assignments"].shape == (B, T_out)
        assert output["fused_features"].shape == (B, T_out, 64)

    def test_gradient_flows_end_to_end(self):
        """Verify gradients flow from loss back to all model parameters."""
        B, T_frames, N, D = 1, 20, 4, 64
        T_samples = T_frames * 320

        wavlm_states = torch.randn(B, T_frames, N, D)
        waveform = torch.randn(B, T_samples)

        output = self.model(wavlm_states, waveform)
        loss = output["spoof_logits"].sum() + output["boundary_logits"].sum()
        loss.backward()

        # Check that key parameters have gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self):
        """Model in eval mode should produce deterministic output."""
        self.model.eval()
        B, T_frames, N, D = 1, 10, 4, 64
        T_samples = T_frames * 320

        wavlm_states = torch.randn(B, T_frames, N, D)
        waveform = torch.randn(B, T_samples)

        out1 = self.model(wavlm_states, waveform)
        out2 = self.model(wavlm_states, waveform)

        assert torch.allclose(
            out1["spoof_logits"], out2["spoof_logits"], atol=1e-5
        )
