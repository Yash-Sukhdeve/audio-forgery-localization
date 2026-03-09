"""Unified evaluation entry point for all methods.

Combines EER and classification metrics into a single call.
All methods (FARA, BAM, CFPRF, PSDS) use this for consistent evaluation.
"""
from typing import Dict

import numpy as np

from core.metrics.eer import compute_eer
from core.metrics.classification import compute_frame_metrics


def evaluate_localization(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate frame-level forgery localization.

    Computes EER from continuous scores, then binarizes predictions
    at the EER threshold to compute Precision, Recall, F1.

    Args:
        scores: 1D array of per-frame spoof scores. Shape: (N,).
        labels: 1D binary array of ground truth. Shape: (N,). 1=spoof, 0=bonafide.

    Returns:
        Dict with keys: 'eer', 'threshold', 'precision', 'recall', 'f1'.
    """
    eer, threshold = compute_eer(scores, labels)
    binary_preds = (scores >= threshold).astype(int)
    frame_metrics = compute_frame_metrics(binary_preds, labels)

    return {
        "eer": eer,
        "threshold": threshold,
        **frame_metrics,
    }
