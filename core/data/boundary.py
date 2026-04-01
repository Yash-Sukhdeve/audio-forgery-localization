"""Boundary label generation from frame-level spoof labels.

Generates binary boundary labels indicating frames at the transition
between bonafide and spoof regions. Used as auxiliary supervision in
FARA (Luo et al., IEEE/ACM TASLP 2026, Section III).

Convention: boundary=1 at frames where label changes between adjacent frames.
"""

import torch


def generate_boundary_labels(
    frame_labels: torch.Tensor, pad_value: int = -1
) -> torch.Tensor:
    """Generate boundary labels from frame-level spoof labels.

    A frame is a boundary frame if its label differs from the previous
    frame's label. Padding frames retain pad_value.

    Args:
        frame_labels: (B, T) frame labels. 0=bonafide, 1=spoof, -1=padding.
        pad_value: Value used for padding (default -1).

    Returns:
        boundary_labels: (B, T) binary boundary labels.
            0=non-boundary, 1=boundary, pad_value=padding.
    """
    B, T = frame_labels.shape
    boundary = torch.full_like(frame_labels, pad_value)

    # Valid frames mask
    valid = frame_labels != pad_value

    # First valid frame is non-boundary
    boundary[valid] = 0

    if T > 1:
        # Detect transitions: label[t] != label[t-1]
        shifted = frame_labels[:, :-1]
        current = frame_labels[:, 1:]
        both_valid = valid[:, :-1] & valid[:, 1:]
        transitions = (shifted != current) & both_valid

        # Mark both sides of each transition as boundary
        boundary[:, 1:][transitions] = 1
        boundary[:, :-1][transitions] = 1

    return boundary
