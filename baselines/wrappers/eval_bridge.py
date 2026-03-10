"""Evaluation bridge: captures baseline outputs for unified evaluation.

Converts frame-level predictions from any baseline into the format
expected by core/metrics/evaluate.py. Used by all baseline wrappers.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from core.metrics.evaluate import evaluate_localization


def collect_predictions_from_npy(
    pred_dir: str,
    label_dir: str,
    split: str,
    resolution: float = 0.02,
    invert_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect frame-level predictions and labels from npy files.

    Args:
        pred_dir: Directory containing per-utterance prediction .npy files.
        label_dir: Directory containing segment label .npy files.
        split: Dataset split (train/dev/eval).
        resolution: Label resolution in seconds.
        invert_labels: If True, invert raw labels (1->0, 0->1) to match
                       our convention (0=bonafide, 1=spoof). PartialSpoof
                       raw npy uses inverted convention.

    Returns:
        Tuple of (all_scores, all_labels) as 1D numpy arrays.
    """
    label_path = Path(label_dir) / f"{split}_seglab_{resolution:.2f}.npy"
    seg_labels = np.load(str(label_path), allow_pickle=True).item()

    all_scores = []
    all_labels = []

    pred_dir = Path(pred_dir)
    for utt_id, raw_labels in seg_labels.items():
        pred_file = pred_dir / f"{utt_id}.npy"
        if not pred_file.exists():
            continue

        scores = np.load(str(pred_file))
        int_labels = np.array([int(x) for x in raw_labels], dtype=np.int64)
        if invert_labels:
            int_labels = 1 - int_labels

        min_len = min(len(scores), len(int_labels))
        all_scores.append(scores[:min_len])
        all_labels.append(int_labels[:min_len])

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_baseline(
    pred_dir: str,
    label_dir: str,
    split: str = "eval",
    resolution: float = 0.02,
) -> Dict[str, float]:
    """Run unified evaluation on baseline predictions.

    Args:
        pred_dir: Directory with per-utterance prediction .npy files.
        label_dir: Directory with segment label .npy files.
        split: Dataset split.
        resolution: Label resolution.

    Returns:
        Dict with eer, threshold, precision, recall, f1.
    """
    scores, labels = collect_predictions_from_npy(
        pred_dir, label_dir, split, resolution
    )
    return evaluate_localization(scores, labels)
