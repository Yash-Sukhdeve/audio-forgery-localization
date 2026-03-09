"""Collation utilities for batching variable-length audio samples.

Used by all dataset loaders via DataLoader(collate_fn=pad_collate).

Reference: Standard practice in speech processing pipelines; see
    S. Watanabe et al., "ESPnet: End-to-End Speech Processing Toolkit,"
    Proc. Interspeech, 2018 (for padding/collation conventions).
"""
from typing import Any, Dict, List

import torch


def pad_collate(
    batch: List[Dict[str, Any]],
    label_pad_value: int = -1,
) -> Dict[str, Any]:
    """Collate variable-length samples with zero-padding for waveforms
    and ignore-index padding for labels.

    Args:
        batch: List of sample dicts from BaseAudioDataset.__getitem__.
        label_pad_value: Value used to pad frame_labels (default -1,
                         compatible with CrossEntropyLoss ignore_index).

    Returns:
        Dict with:
            'waveforms': Tensor[B, max_samples], zero-padded
            'frame_labels': Tensor[B, max_frames], padded with label_pad_value
            'lengths': Tensor[B], original waveform lengths
            'frame_lengths': Tensor[B], original frame counts
            'utt_ids': List[str]
    """
    waveforms = [s["waveform"] for s in batch]
    frame_labels = [s["frame_labels"] for s in batch]
    utt_ids = [s["utt_id"] for s in batch]

    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = lengths.max().item()

    padded_waveforms = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, : w.shape[0]] = w

    frame_lengths = torch.tensor(
        [fl.shape[0] for fl in frame_labels], dtype=torch.long
    )
    max_frames = frame_lengths.max().item()

    padded_labels = torch.full(
        (len(batch), max_frames), label_pad_value, dtype=torch.long
    )
    for i, fl in enumerate(frame_labels):
        padded_labels[i, : fl.shape[0]] = fl

    return {
        "waveforms": padded_waveforms,
        "frame_labels": padded_labels,
        "lengths": lengths,
        "frame_lengths": frame_lengths,
        "utt_ids": utt_ids,
    }
