"""Equal Error Rate (EER) computation for frame-level localization.

EER is the operating point where False Acceptance Rate equals
False Rejection Rate on the DET curve.
"""
from typing import Tuple

import numpy as np


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate and the corresponding threshold.

    Args:
        scores: 1D array of continuous detection scores. Shape: (N,).
        labels: 1D binary array of ground truth. Shape: (N,). 1=spoof/positive, 0=bonafide/negative.

    Returns:
        Tuple of (eer, threshold) where eer is the equal error rate value
        and threshold is the score at which FAR == FRR.

    Raises:
        ValueError: If inputs are empty or contain only one class.
    """
    if len(scores) == 0 or len(labels) == 0:
        raise ValueError("Scores and labels must be non-empty.")

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("Both positive and negative samples are required.")

    # Sort thresholds from all unique scores
    thresholds = np.sort(np.unique(scores))

    far_list = []
    frr_list = []

    for t in thresholds:
        # False Acceptance Rate: fraction of negatives scored >= threshold
        far = np.mean(negative_scores >= t)
        # False Rejection Rate: fraction of positives scored < threshold
        frr = np.mean(positive_scores < t)
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Find the crossover point where FAR and FRR are closest
    diff = far_arr - frr_arr
    idx = np.argmin(np.abs(diff))

    # Interpolate for more accurate EER
    eer = float((far_arr[idx] + frr_arr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return eer, threshold
