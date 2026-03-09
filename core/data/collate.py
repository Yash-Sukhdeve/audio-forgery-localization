"""Collation utility for variable-length audio batches.

Pads waveforms and frame labels to the maximum length in each batch,
producing tensors suitable for batched model input.

Reference: Standard practice in speech processing pipelines; see
    S. Watanabe et al., "ESPnet: End-to-End Speech Processing Toolkit,"
    Proc. Interspeech, 2018 (for padding/collation conventions).
"""
from typing import Any, Dict, List

import torch


def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length samples into a padded batch.

    Each sample dict must contain:
        - "waveform": Tensor of shape (num_samples,)
        - "frame_labels": Tensor of shape (num_frames,)
        - "utt_id": str
        - "num_frames": int

    Returns:
        Dict with:
            - "waveforms": Tensor (B, max_samples), zero-padded
            - "frame_labels": Tensor (B, max_frames), zero-padded
            - "utt_ids": List[str]
            - "num_frames": Tensor (B,), original frame counts
            - "num_samples": Tensor (B,), original sample counts
    """
    waveforms = [s["waveform"] for s in batch]
    frame_labels = [s["frame_labels"] for s in batch]
    utt_ids = [s["utt_id"] for s in batch]
    num_frames = torch.tensor([s["num_frames"] for s in batch], dtype=torch.long)
    num_samples = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)

    max_samples = int(num_samples.max().item())
    max_frames = int(num_frames.max().item())

    padded_waveforms = torch.zeros(len(batch), max_samples)
    padded_labels = torch.zeros(len(batch), max_frames, dtype=torch.long)

    for i, (wav, lab) in enumerate(zip(waveforms, frame_labels)):
        padded_waveforms[i, : wav.shape[0]] = wav
        padded_labels[i, : lab.shape[0]] = lab

    return {
        "waveforms": padded_waveforms,
        "frame_labels": padded_labels,
        "utt_ids": utt_ids,
        "num_frames": num_frames,
        "num_samples": num_samples,
    }
