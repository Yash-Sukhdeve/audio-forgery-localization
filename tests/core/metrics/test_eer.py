import pytest
import numpy as np
from core.metrics.eer import compute_eer


class TestComputeEER:
    def test_perfect_separation(self):
        genuine_scores = np.array([0.9, 0.8, 0.95, 0.85])
        spoof_scores = np.array([0.1, 0.2, 0.05, 0.15])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.concatenate([genuine_scores, spoof_scores])
        eer, threshold = compute_eer(scores, labels)
        assert eer < 0.05

    def test_random_scores(self):
        np.random.seed(42)
        scores = np.random.rand(10000)
        labels = np.array([1] * 5000 + [0] * 5000)
        eer, _ = compute_eer(scores, labels)
        assert 0.4 < eer < 0.6

    def test_returns_threshold(self):
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1])
        eer, threshold = compute_eer(scores, labels)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            compute_eer(np.array([]), np.array([]))
