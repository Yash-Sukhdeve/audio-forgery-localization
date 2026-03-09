"""Equal Error Rate (EER) computation for frame-level localization.

EER is the operating point where False Acceptance Rate equals
False Rejection Rate on the ROC curve.

Used by all methods for evaluation.
Reference: FARA paper (Luo et al., IEEE TASLP 2026) Section IV-B.
"""
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate and the corresponding threshold.

    Uses sklearn's roc_curve for O(n log n) efficiency, suitable for
    frame-level evaluation with millions of frames.

    Args:
        scores: 1D array of continuous detection scores. Shape: (N,).
        labels: 1D binary array of ground truth. Shape: (N,).
                1=spoof/positive, 0=bonafide/negative.

    Returns:
        Tuple of (eer, threshold) where eer is in [0, 1] and threshold
        is the score at which FAR approximately equals FRR.

    Raises:
        ValueError: If inputs are empty.
    """
    if len(scores) == 0 or len(labels) == 0:
        raise ValueError("Scores and labels must be non-empty.")

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find crossover: where FPR and FNR are closest
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))

    # Interpolate EER between the two nearest points
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    # Threshold at crossover (clip to valid range)
    threshold = float(np.clip(thresholds[idx], 0.0, 1.0))

    return eer, threshold
