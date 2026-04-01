"""Tests for CMoE (Cluster-Based Mixture of Experts) module.

Reference: Luo et al., IEEE/ACM TASLP 2026, Section III-B, Eq. 2-4.
"""
import torch

from fara.model.cmoe import CMoE, CMoERouter, CMoEExpert


class TestCMoERouter:
    def setup_method(self):
        self.d_model = 64
        self.k = 4
        self.router = CMoERouter(self.d_model, self.k)

    def test_weights_shape(self):
        x = torch.randn(2, 50, self.d_model)
        weights, assignments = self.router(x)
        assert weights.shape == (2, 50, self.k)
        assert assignments.shape == (2, 50)

    def test_weights_sum_to_one(self):
        """Expert weights from softmax must sum to 1 per frame."""
        x = torch.randn(2, 50, self.d_model)
        weights, _ = self.router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_assignments_valid_range(self):
        """Cluster assignments must be in [0, K-1]."""
        x = torch.randn(2, 50, self.d_model)
        _, assignments = self.router(x)
        assert (assignments >= 0).all()
        assert (assignments < self.k).all()

    def test_centroid_ema_update(self):
        """Centroids should change during training (EMA update)."""
        self.router.train()
        x = torch.randn(4, 20, self.d_model)
        old_centroids = self.router.centroids.clone()
        self.router(x)
        # After first forward, centroids should be initialized/updated
        assert not torch.equal(old_centroids, self.router.centroids)

    def test_no_update_in_eval(self):
        """Centroids must not change during eval."""
        self.router.train()
        x = torch.randn(2, 20, self.d_model)
        self.router(x)  # Initialize centroids
        self.router.eval()
        old = self.router.centroids.clone()
        self.router(x)
        assert torch.equal(old, self.router.centroids)


class TestCMoEExpert:
    def test_output_shape(self):
        expert = CMoEExpert(d_model=64)
        x = torch.randn(2, 50, 64)
        out = expert(x)
        assert out.shape == (2, 50, 64)


class TestCMoE:
    def setup_method(self):
        self.d_model = 64
        self.k = 4
        self.cmoe = CMoE(self.d_model, self.k)

    def test_output_shape(self):
        x = torch.randn(2, 50, self.d_model)
        out, assignments = self.cmoe(x)
        assert out.shape == (2, 50, self.d_model)
        assert assignments.shape == (2, 50)

    def test_gradient_flows(self):
        x = torch.randn(2, 10, self.d_model, requires_grad=True)
        out, _ = self.cmoe(x)
        out.sum().backward()
        assert x.grad is not None
