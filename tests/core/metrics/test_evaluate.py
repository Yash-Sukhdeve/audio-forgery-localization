import pytest
import numpy as np
from core.metrics.evaluate import evaluate_localization


class TestEvaluateLocalization:
    def test_returns_all_metrics(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        result = evaluate_localization(scores, labels)
        assert "eer" in result
        assert "threshold" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_perfect_separation(self):
        scores = np.array([0.9, 0.85, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        result = evaluate_localization(scores, labels)
        assert result["eer"] < 0.05
        assert result["f1"] > 0.95

    def test_multi_utterance(self):
        all_scores = []
        all_labels = []
        for _ in range(5):
            n = np.random.randint(50, 200)
            scores = np.random.rand(n)
            labels = (scores > 0.5).astype(int)
            all_scores.append(scores)
            all_labels.append(labels)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        result = evaluate_localization(scores, labels)
        assert 0 <= result["eer"] <= 1
        assert 0 <= result["f1"] <= 1
