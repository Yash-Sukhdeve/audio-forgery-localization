"""Frame-level classification metrics.

Used by all methods: Precision, Recall, F1 at frame level.
Reference: FARA paper (Luo et al., IEEE TASLP 2026) Section IV-B.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_frame_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute frame-level Precision, Recall, and F1.

    Args:
        predictions: 1D binary array of predicted labels. Shape: (N,).
        labels: 1D binary array of ground truth labels. Shape: (N,).
        pos_label: Label considered as positive (spoof). Default: 1.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.

    Raises:
        ValueError: If inputs are empty or mismatched.
    """
    if len(predictions) == 0 or len(labels) == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions ({len(predictions)}) and labels ({len(labels)}) "
            "must have same length."
        )

    return {
        "precision": float(precision_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
        "recall": float(recall_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
        "f1": float(f1_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
    }
