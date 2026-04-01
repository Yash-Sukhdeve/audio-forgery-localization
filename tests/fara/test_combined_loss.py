"""Tests for Combined FARA Training Loss.

Reference: Luo et al., IEEE/ACM TASLP 2026, Section III-C, Eq. 8.
L_train = L_spoof + 0.5 * L_boundary + 0.2 * L_CRL
"""
import pytest
import torch

from fara.losses.combined_loss import CombinedLoss


def _make_model_output(B: int = 2, T: int = 20, D: int = 64, K: int = 4):
    """Create a dummy model output dict for testing.

    Args:
        B: Batch size.
        T: Number of frames.
        D: Feature dimension.
        K: Number of clusters.

    Returns:
        Tuple of (model_output dict, frame_labels, boundary_labels).
    """
    model_output = {
        "spoof_logits": torch.randn(B, T, 2),
        "boundary_logits": torch.randn(B, T, 2),
        "fused_features": torch.randn(B, T, D),
        "cluster_assignments": torch.randint(0, K, (B, T)),
    }
    frame_labels = torch.randint(0, 2, (B, T))
    boundary_labels = torch.randint(0, 2, (B, T))
    return model_output, frame_labels, boundary_labels


class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def setup_method(self):
        self.loss_fn = CombinedLoss(
            spoof_weight=1.0,
            boundary_weight=0.5,
            crl_weight=0.2,
            beta=0.3,
        )

    def test_returns_all_loss_components(self):
        """Output dict must contain loss, loss_spoof, loss_boundary, loss_crl."""
        model_output, frame_labels, boundary_labels = _make_model_output()
        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        assert "loss" in result
        assert "loss_spoof" in result
        assert "loss_boundary" in result
        assert "loss_crl" in result

    def test_all_losses_are_scalars(self):
        """All loss values must be scalar tensors."""
        model_output, frame_labels, boundary_labels = _make_model_output()
        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        for key, value in result.items():
            assert value.dim() == 0, f"{key} is not a scalar"

    def test_all_losses_non_negative(self):
        """All loss components must be non-negative."""
        model_output, frame_labels, boundary_labels = _make_model_output()
        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        for key, value in result.items():
            assert value.item() >= 0.0, f"{key} is negative: {value.item()}"

    def test_weighted_sum_correct(self):
        """Total loss must equal weighted sum of components (Eq. 8)."""
        model_output, frame_labels, boundary_labels = _make_model_output()
        result = self.loss_fn(model_output, frame_labels, boundary_labels)

        expected = (
            1.0 * result["loss_spoof"]
            + 0.5 * result["loss_boundary"]
            + 0.2 * result["loss_crl"]
        )
        assert result["loss"].item() == pytest.approx(
            expected.item(), abs=1e-5
        )

    def test_custom_weights(self):
        """Custom weight parameters should affect the total loss."""
        loss_fn = CombinedLoss(
            spoof_weight=2.0,
            boundary_weight=1.0,
            crl_weight=0.5,
        )
        model_output, frame_labels, boundary_labels = _make_model_output()
        result = loss_fn(model_output, frame_labels, boundary_labels)

        expected = (
            2.0 * result["loss_spoof"]
            + 1.0 * result["loss_boundary"]
            + 0.5 * result["loss_crl"]
        )
        assert result["loss"].item() == pytest.approx(
            expected.item(), abs=1e-5
        )

    def test_ignore_index_padding(self):
        """Padded frames (label=-1) must not contribute to CE losses."""
        B, T = 2, 20
        model_output, _, _ = _make_model_output(B=B, T=T)

        # All frames are padding
        frame_labels = torch.full((B, T), -1, dtype=torch.long)
        boundary_labels = torch.full((B, T), -1, dtype=torch.long)

        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        # CE with all ignore_index returns nan (no valid samples).
        # This is expected PyTorch behavior — in practice, batches always
        # have at least some valid frames.
        import math
        assert math.isnan(result["loss_spoof"].item()) or result["loss_spoof"].item() == pytest.approx(0.0, abs=1e-6)
        assert math.isnan(result["loss_boundary"].item()) or result["loss_boundary"].item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_padding(self):
        """Loss should only reflect non-padded frames."""
        B, T = 1, 10
        model_output, _, _ = _make_model_output(B=B, T=T)

        frame_labels = torch.zeros(B, T, dtype=torch.long)
        frame_labels[0, 5:] = -1  # Half padded
        boundary_labels = frame_labels.clone()

        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        # Should produce valid (finite) loss from non-padded frames
        assert torch.isfinite(result["loss"])

    def test_gradient_flows_through_all_components(self):
        """Gradients must flow to logits and features."""
        model_output, frame_labels, boundary_labels = _make_model_output()

        # Require grad on all inputs
        model_output["spoof_logits"].requires_grad_(True)
        model_output["boundary_logits"].requires_grad_(True)
        model_output["fused_features"].requires_grad_(True)

        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        result["loss"].backward()

        assert model_output["spoof_logits"].grad is not None
        assert model_output["boundary_logits"].grad is not None
        assert model_output["fused_features"].grad is not None

    def test_batch_size_one(self):
        """Loss should work with batch size 1."""
        model_output, frame_labels, boundary_labels = _make_model_output(B=1)
        result = self.loss_fn(model_output, frame_labels, boundary_labels)
        assert torch.isfinite(result["loss"])

    def test_deterministic_output(self):
        """Same input should produce same loss (no randomness in forward)."""
        torch.manual_seed(42)
        model_output, frame_labels, boundary_labels = _make_model_output()

        result1 = self.loss_fn(model_output, frame_labels, boundary_labels)
        result2 = self.loss_fn(model_output, frame_labels, boundary_labels)

        assert result1["loss"].item() == pytest.approx(
            result2["loss"].item(), abs=1e-6
        )
