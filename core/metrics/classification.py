"""Frame-level classification metrics for forgery localization.

Provides Precision, Recall, and F1-score computed over binary
frame-level predictions vs ground-truth labels.
"""
from typing import Dict

import numpy as np


def compute_frame_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute frame-level Precision, Recall, and F1.

    Args:
        predictions: 1D binary array of predicted labels. Shape: (N,).
        labels: 1D binary array of ground truth. Shape: (N,). 1=spoof, 0=bonafide.

    Returns:
        Dict with keys: 'precision', 'recall', 'f1'.

    Raises:
        ValueError: If inputs are empty.
    """
    if len(predictions) == 0 or len(labels) == 0:
        raise ValueError("Predictions and labels must be non-empty.")

    predictions = np.asarray(predictions, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
