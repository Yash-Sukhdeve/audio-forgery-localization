"""Audio I/O utilities. Shared by all dataset loaders and models."""
import math
from pathlib import Path

import torch
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and return 1D tensor at target sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, etc.)
        target_sr: Target sample rate in Hz.

    Returns:
        1D float32 tensor of shape (num_samples,).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def get_num_frames(num_samples: int, sr: int = 16000, frame_duration_ms: int = 20) -> int:
    """Compute number of frames for a given audio length.

    Args:
        num_samples: Number of audio samples.
        sr: Sample rate in Hz.
        frame_duration_ms: Frame duration in milliseconds.

    Returns:
        Number of frames (ceiling division).
    """
    samples_per_frame = sr * frame_duration_ms // 1000
    return math.ceil(num_samples / samples_per_frame)
