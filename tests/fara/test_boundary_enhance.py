"""Tests for BoundaryEnhance, ClassifyHead, and AttentionMask.

Reference: Luo et al., IEEE/ACM TASLP 2026, Section III.
"""
import torch

from fara.model.boundary_enhance import (
    AttentionMask,
    BoundaryEnhance,
    ClassifyHead,
)


class TestBoundaryEnhance:
    def test_output_shape(self):
        module = BoundaryEnhance(d_model=64, kernel_size=3)
        x = torch.randn(2, 50, 64)
        out = module(x)
        assert out.shape == (2, 50, 64)

    def test_gradient_flows(self):
        module = BoundaryEnhance(d_model=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None


class TestClassifyHead:
    def test_output_shape(self):
        head = ClassifyHead(d_model=64, num_classes=2)
        x = torch.randn(2, 50, 64)
        out = head(x)
        assert out.shape == (2, 50, 2)

    def test_different_num_classes(self):
        head = ClassifyHead(d_model=64, num_classes=3)
        x = torch.randn(2, 50, 64)
        out = head(x)
        assert out.shape == (2, 50, 3)


class TestAttentionMask:
    def test_output_shape(self):
        mask = AttentionMask()
        features = torch.randn(2, 50, 64)
        boundary_logits = torch.randn(2, 50, 2)
        out = mask(features, boundary_logits)
        assert out.shape == (2, 50, 64)

    def test_mask_amplifies_features(self):
        """Attention mask uses 1+sigmoid, so output magnitude >= input."""
        mask = AttentionMask()
        features = torch.ones(1, 5, 8)
        boundary_logits = torch.zeros(1, 5, 2)
        out = mask(features, boundary_logits)
        # sigmoid(0) = 0.5, so mask = 1.5, output = 1.5
        assert torch.allclose(out, torch.full_like(out, 1.5), atol=1e-5)
