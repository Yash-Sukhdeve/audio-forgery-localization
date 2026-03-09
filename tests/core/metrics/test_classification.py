import pytest
import numpy as np
from core.metrics.classification import compute_frame_metrics


class TestComputeFrameMetrics:
    def test_perfect_predictions(self):
        preds = np.array([1, 1, 0, 0, 1])
        labels = np.array([1, 1, 0, 0, 1])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong(self):
        preds = np.array([0, 0, 1, 1])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_partial_match(self):
        preds = np.array([1, 1, 1, 0])
        labels = np.array([1, 1, 0, 1])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-6)
        assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-6)

    def test_returns_all_keys(self):
        preds = np.array([1, 0])
        labels = np.array([1, 0])
        metrics = compute_frame_metrics(preds, labels)
        assert set(metrics.keys()) == {"precision", "recall", "f1"}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_frame_metrics(np.array([]), np.array([]))
