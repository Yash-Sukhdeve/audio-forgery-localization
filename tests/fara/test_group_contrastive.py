"""Tests for Group Contrastive Loss (GCL).

Reference: Luo et al., IEEE/ACM TASLP 2026, Section III-C, Eq. 5-7.
"""
import pytest
import torch

from fara.losses.group_contrastive import GroupContrastiveLoss


class TestGroupContrastiveLoss:
    """Tests for GroupContrastiveLoss."""

    def setup_method(self):
        self.loss_fn = GroupContrastiveLoss(beta=0.3, max_samples=2048)

    def test_output_is_scalar(self):
        """Loss output must be a scalar tensor."""
        features = torch.randn(2, 10, 64)
        labels = torch.randint(0, 2, (2, 10))
        clusters = torch.randint(0, 2, (2, 10))
        loss = self.loss_fn(features, labels, clusters)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self):
        """GCL is a sum of squared terms, must be >= 0."""
        features = torch.randn(2, 20, 64)
        labels = torch.randint(0, 2, (2, 20))
        clusters = torch.randint(0, 3, (2, 20))
        loss = self.loss_fn(features, labels, clusters)
        assert loss.item() >= 0.0

    def test_identical_same_label_features_zero_pos_loss(self):
        """When all same-label frames have identical features,
        positive-pair loss should be zero (cosine sim = 1.0)."""
        B, T, D = 1, 10, 32
        features = torch.zeros(B, T, D)
        # All frames in same cluster, same label, identical features
        features[:, :, 0] = 1.0  # unit vector along dim 0
        labels = torch.zeros(B, T, dtype=torch.long)  # all bonafide
        clusters = torch.zeros(B, T, dtype=torch.long)  # single group
        loss = self.loss_fn(features, labels, clusters)
        # Only positive pairs exist, all with sim=1 -> loss=0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_beta_margin_negative_pairs(self):
        """Negative pairs with cosine similarity below beta contribute
        zero loss due to the ReLU margin (Eq. 6, I=0 term)."""
        B, T, D = 1, 4, 32
        features = torch.zeros(B, T, D)
        # Two clusters, each with one bonafide and one spoof frame
        # Make features orthogonal so cosine sim = 0 < beta=0.3
        features[0, 0, 0] = 1.0  # cluster 0, label 0
        features[0, 1, 1] = 1.0  # cluster 0, label 1
        features[0, 2, 2] = 1.0  # cluster 1, label 0
        features[0, 3, 3] = 1.0  # cluster 1, label 1

        labels = torch.tensor([[0, 1, 0, 1]])
        clusters = torch.tensor([[0, 0, 1, 1]])

        loss = self.loss_fn(features, labels, clusters)
        # sim between pairs within each group = 0.0 < beta=0.3
        # So negative loss = max(0, 0 - 0.3)^2 = 0
        # No positive pairs within groups (each label appears once)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_high_similarity_negative_pairs_nonzero_loss(self):
        """Negative pairs with similarity > beta must produce positive loss."""
        B, T, D = 1, 4, 16
        features = torch.zeros(B, T, D)
        # Same cluster, different labels, nearly identical features -> sim ~ 1
        features[0, 0] = torch.randn(D)
        features[0, 1] = features[0, 0].clone()  # near-identical
        labels = torch.tensor([[0, 1, 0, 1]])
        clusters = torch.tensor([[0, 0, 0, 0]])
        features[0, 2] = features[0, 0].clone()
        features[0, 3] = features[0, 0].clone()

        loss = self.loss_fn(features, labels, clusters)
        # sim ~ 1.0 > beta=0.3, so negative loss > 0
        # Also positive loss ~ 0 since features identical within labels
        assert loss.item() > 0.0

    def test_ignore_index_respected(self):
        """Frames with label=-1 (padding) must not contribute to loss."""
        B, T, D = 1, 10, 32
        features = torch.randn(B, T, D)
        labels = torch.full((B, T), -1, dtype=torch.long)  # all padding
        clusters = torch.zeros(B, T, dtype=torch.long)
        loss = self.loss_fn(features, labels, clusters)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_padding(self):
        """Only valid frames contribute; padded frames are excluded."""
        B, T, D = 1, 10, 32
        features = torch.randn(B, T, D)
        labels = torch.zeros(B, T, dtype=torch.long)
        labels[0, 5:] = -1  # last 5 frames are padding
        clusters = torch.zeros(B, T, dtype=torch.long)

        loss_all = self.loss_fn(
            features[:, :5], labels[:, :5], clusters[:, :5]
        )
        loss_padded = self.loss_fn(features, labels, clusters)
        # Should produce the same loss since padded frames are ignored
        assert loss_all.item() == pytest.approx(loss_padded.item(), abs=1e-5)

    def test_gradient_flows(self):
        """Verify gradient flows back through features."""
        features = torch.randn(2, 8, 32, requires_grad=True)
        labels = torch.randint(0, 2, (2, 8))
        clusters = torch.randint(0, 2, (2, 8))
        loss = self.loss_fn(features, labels, clusters)
        loss.backward()
        assert features.grad is not None
        assert not torch.all(features.grad == 0)

    def test_multiple_groups(self):
        """Loss aggregation across multiple cluster groups."""
        B, T, D = 2, 16, 64
        features = torch.randn(B, T, D)
        labels = torch.randint(0, 2, (B, T))
        # 4 distinct groups
        clusters = torch.randint(0, 4, (B, T))
        loss = self.loss_fn(features, labels, clusters)
        assert loss.item() >= 0.0

    def test_single_frame_group_no_error(self):
        """Groups with a single frame should not cause errors."""
        B, T, D = 1, 3, 16
        features = torch.randn(B, T, D)
        labels = torch.tensor([[0, 1, 0]])
        # Each frame in its own group
        clusters = torch.tensor([[0, 1, 2]])
        loss = self.loss_fn(features, labels, clusters)
        # Single-frame groups have no pairs, contribute zero
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_subsampling_large_input(self):
        """When N > max_samples, subsampling should still produce valid loss."""
        loss_fn = GroupContrastiveLoss(beta=0.3, max_samples=32)
        B, T, D = 4, 100, 16  # 400 frames > 32
        features = torch.randn(B, T, D)
        labels = torch.randint(0, 2, (B, T))
        clusters = torch.randint(0, 3, (B, T))
        loss = loss_fn(features, labels, clusters)
        assert loss.item() >= 0.0
