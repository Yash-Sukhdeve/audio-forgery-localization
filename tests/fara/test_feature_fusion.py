"""Tests for FeatureFusion module.

Reference: Luo et al., IEEE/ACM TASLP 2026, Section III.
"""
import torch

from fara.model.feature_fusion import FeatureFusion


class TestFeatureFusion:
    def setup_method(self):
        self.d_model = 64
        self.sincnet_dim = 16
        self.module = FeatureFusion(self.d_model, self.sincnet_dim)

    def test_output_shape(self):
        wavlm = torch.randn(2, 50, self.d_model)
        sincnet = torch.randn(2, 50, self.sincnet_dim)
        out = self.module(wavlm, sincnet)
        assert out.shape == (2, 50, self.d_model)

    def test_gate_values_bounded(self):
        """Sigmoid gate values must be in [0, 1]."""
        gate = torch.sigmoid(self.module.gate)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_gate_initial_value(self):
        """Gate initialized to 0 → sigmoid(0) = 0.5 (equal weighting)."""
        gate = torch.sigmoid(self.module.gate)
        assert torch.allclose(gate, torch.full_like(gate, 0.5))

    def test_gradient_flows_both_inputs(self):
        """Gradients must flow through both WavLM and SincNet branches."""
        wavlm = torch.randn(2, 10, self.d_model, requires_grad=True)
        sincnet = torch.randn(2, 10, self.sincnet_dim, requires_grad=True)
        out = self.module(wavlm, sincnet)
        out.sum().backward()
        assert wavlm.grad is not None
        assert sincnet.grad is not None
        assert not torch.all(wavlm.grad == 0)
        assert not torch.all(sincnet.grad == 0)
