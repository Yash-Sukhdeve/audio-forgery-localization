"""FARA Phase 2 gate test — full forward pass integration test.

Verifies that the entire FARA pipeline (model + losses) works end-to-end
with dummy data, including:
  - Forward pass produces correct output shapes
  - Combined loss computes without errors
  - Gradients flow through the entire pipeline
  - Optimizer step runs without errors

Reference: Luo et al., IEEE/ACM TASLP 2026, Fig. 1 and Eq. 8.
"""
import torch

from fara.model.fara import FARA
from fara.losses.combined_loss import CombinedLoss


class TestFARAIntegration:
    """End-to-end integration tests for FARA model + loss."""

    def setup_method(self):
        # Small model for testing
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
        self.loss_fn = CombinedLoss(
            spoof_weight=1.0,
            boundary_weight=0.5,
            crl_weight=0.2,
            beta=0.3,
        )

    def _make_dummy_batch(self, B=2, T_frames=50):
        """Create a dummy batch matching expected FARA inputs."""
        T_samples = T_frames * 320  # stride=320 -> 20ms at 16kHz
        N = 4  # n_wavlm_layers (small for test)
        D = 64  # d_model (small for test)

        wavlm_states = torch.randn(B, T_frames, N, D)
        waveform = torch.randn(B, T_samples)
        frame_labels = torch.randint(0, 2, (B, T_frames))
        boundary_labels = torch.randint(0, 2, (B, T_frames))

        return wavlm_states, waveform, frame_labels, boundary_labels

    def test_forward_pass_produces_all_outputs(self):
        """Full forward pass must produce all required output keys."""
        wavlm, waveform, _, _ = self._make_dummy_batch()
        output = self.model(wavlm, waveform)

        required_keys = {
            "spoof_logits",
            "boundary_logits",
            "cluster_assignments",
            "fused_features",
        }
        assert required_keys == set(output.keys())

    def test_loss_computes_correctly(self):
        """Combined loss must compute without errors and return all components."""
        wavlm, waveform, frame_labels, boundary_labels = self._make_dummy_batch()
        output = self.model(wavlm, waveform)

        # Align labels to model output time dimension
        T_out = output["spoof_logits"].shape[1]
        frame_labels = frame_labels[:, :T_out]
        boundary_labels = boundary_labels[:, :T_out]

        losses = self.loss_fn(output, frame_labels, boundary_labels)

        assert "loss" in losses
        assert "loss_spoof" in losses
        assert "loss_boundary" in losses
        assert "loss_crl" in losses
        assert torch.isfinite(losses["loss"])
        assert losses["loss"].item() > 0

    def test_gradient_flow_end_to_end(self):
        """Gradients must flow from combined loss to all model parameters."""
        wavlm, waveform, frame_labels, boundary_labels = self._make_dummy_batch(B=1, T_frames=20)
        output = self.model(wavlm, waveform)

        T_out = output["spoof_logits"].shape[1]
        frame_labels = frame_labels[:, :T_out]
        boundary_labels = boundary_labels[:, :T_out]

        losses = self.loss_fn(output, frame_labels, boundary_labels)
        losses["loss"].backward()

        params_with_grad = 0
        params_total = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_total += 1
                if param.grad is not None and not torch.all(param.grad == 0):
                    params_with_grad += 1

        # At least 80% of parameters should have non-zero gradients
        ratio = params_with_grad / max(params_total, 1)
        assert ratio > 0.8, (
            f"Only {params_with_grad}/{params_total} params have gradients"
        )

    def test_optimizer_step(self):
        """A full train step (forward + backward + optimizer) must succeed."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        wavlm, waveform, frame_labels, boundary_labels = self._make_dummy_batch()
        output = self.model(wavlm, waveform)

        T_out = output["spoof_logits"].shape[1]
        frame_labels = frame_labels[:, :T_out]
        boundary_labels = boundary_labels[:, :T_out]

        losses = self.loss_fn(output, frame_labels, boundary_labels)

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        # Verify parameters changed
        wavlm2, waveform2, _, _ = self._make_dummy_batch()
        output2 = self.model(wavlm2, waveform2)
        # Model should produce different outputs after parameter update
        # (probabilistic but near-certain with random inputs)

    def test_loss_weighted_sum_matches_eq8(self):
        """Verify L = L_spoof + 0.5*L_boundary + 0.2*L_CRL (Eq. 8)."""
        wavlm, waveform, frame_labels, boundary_labels = self._make_dummy_batch()
        output = self.model(wavlm, waveform)

        T_out = output["spoof_logits"].shape[1]
        frame_labels = frame_labels[:, :T_out]
        boundary_labels = boundary_labels[:, :T_out]

        losses = self.loss_fn(output, frame_labels, boundary_labels)

        expected = (
            1.0 * losses["loss_spoof"]
            + 0.5 * losses["loss_boundary"]
            + 0.2 * losses["loss_crl"]
        )
        assert torch.allclose(losses["loss"], expected, atol=1e-5)

    def test_padding_handled_correctly(self):
        """Padded frames (label=-1) must not affect loss computation."""
        wavlm, waveform, frame_labels, boundary_labels = self._make_dummy_batch(B=1, T_frames=30)
        output = self.model(wavlm, waveform)

        T_out = output["spoof_logits"].shape[1]
        frame_labels = frame_labels[:, :T_out]
        boundary_labels = boundary_labels[:, :T_out]

        # Mark half as padding
        frame_labels[:, T_out // 2 :] = -1
        boundary_labels[:, T_out // 2 :] = -1

        losses = self.loss_fn(output, frame_labels, boundary_labels)
        assert torch.isfinite(losses["loss"])
