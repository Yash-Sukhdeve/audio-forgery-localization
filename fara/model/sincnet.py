"""SincNet — learnable sinc-function bandpass filter bank.

Implements the SincNet convolutional layer from Ravanelli & Bengio (2018,
IEEE SLT) "Speaker Recognition from Raw Waveform with SincNet", adapted
for FARA (Luo et al., IEEE/ACM TASLP 2026) with stride=320 to align
with WavLM's 20ms frame rate.

Key idea: Instead of learning all CNN filter weights freely, parameterize
each filter by only two learnable cutoff frequencies (low_hz, high_hz).
The filter shape is the analytically defined sinc function, yielding
interpretable bandpass filters with far fewer parameters.

Architecture:
  1. Input: raw waveform (B, T_samples) at 16kHz
  2. Pad input by kernel_size//2 on each side
  3. Apply 1D convolution with sinc-parameterized bandpass filters
  4. stride=320 produces T_samples//320 frames (matching WavLM)
  5. Output: (B, T_frames, 80) — transposed for downstream fusion
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincNet(nn.Module):
    """Learnable sinc-function bandpass filter bank.

    Only the cutoff frequencies are learned; the sinc filter shape is
    analytically defined. Filters are initialized on the mel scale
    (Ravanelli & Bengio 2018).

    Args:
        out_channels: Number of bandpass filters (default 80).
        kernel_size: Filter length in samples (default 251, ~15.7ms at 16kHz).
        stride: Convolution stride (default 320 = 20ms at 16kHz).
        sample_rate: Audio sample rate in Hz (default 16000).
        min_low_hz: Minimum low cutoff frequency in Hz (default 50).
        min_band_hz: Minimum bandwidth in Hz (default 50).
    """

    def __init__(
        self,
        out_channels: int = 80,
        kernel_size: int = 251,
        stride: int = 320,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric filters")

        # Initialize filter cutoff frequencies on the mel scale
        # Mel scale: m = 2595 * log10(1 + f/700)
        high_hz = sample_rate / 2.0
        mel_low = 2595.0 * math.log10(1.0 + min_low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)

        # out_channels + 1 points on mel scale -> out_channels bands
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Learnable parameters: initial low cutoff and bandwidth
        # low_hz = min_low_hz + abs(low_hz_values)
        # high_hz = low_hz + min_band_hz + abs(band_hz_values)
        self.low_hz_values = nn.Parameter(hz_points[:-1] - min_low_hz)
        self.band_hz_values = nn.Parameter(hz_points[1:] - hz_points[:-1] - min_band_hz)

        # Hamming window (fixed, not learnable)
        n = torch.arange(0.0, kernel_size).view(1, -1)
        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (kernel_size - 1))
        self.register_buffer("window", window)

        # Time axis for sinc computation: symmetric around 0
        # Only need the right half since sinc is symmetric; but we compute full
        # Avoid division by zero at center
        half = kernel_size // 2
        t = torch.arange(-(half), half + 1, dtype=torch.float32).view(1, -1)
        self.register_buffer("t", t / sample_rate)

    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized sinc: sin(pi*x) / (pi*x), with sinc(0)=1."""
        # Add small epsilon to avoid division by zero
        mask = x == 0
        x_safe = torch.where(mask, torch.ones_like(x), x)
        result = torch.sin(math.pi * x_safe) / (math.pi * x_safe)
        result = torch.where(mask, torch.ones_like(result), result)
        return result

    def _build_filters(self) -> torch.Tensor:
        """Construct bandpass filters from learnable cutoff frequencies.

        Returns:
            filters: (out_channels, 1, kernel_size) — conv1d weight tensor.
        """
        # Compute actual cutoff frequencies (ensure valid range)
        low_hz = self.min_low_hz + torch.abs(self.low_hz_values)  # (C,)
        high_hz = low_hz + self.min_band_hz + torch.abs(self.band_hz_values)  # (C,)

        # Clamp high_hz to Nyquist
        high_hz = torch.clamp(high_hz, max=self.sample_rate / 2.0)

        # Compute lowpass filters via sinc
        # lowpass(f) = 2f * sinc(2f * t)
        # bandpass = lowpass(f_high) - lowpass(f_low)
        low = low_hz.unsqueeze(1)   # (C, 1)
        high = high_hz.unsqueeze(1)  # (C, 1)

        # t is (1, K), so 2*f*t is (C, K)
        low_pass_low = 2.0 * low * self._sinc(2.0 * low * self.t)
        low_pass_high = 2.0 * high * self._sinc(2.0 * high * self.t)

        # Bandpass = difference of lowpass filters
        bandpass = low_pass_high - low_pass_low  # (C, K)

        # Apply Hamming window
        bandpass = bandpass * self.window  # (C, K)

        # Normalize by L1 norm (per filter)
        bandpass = bandpass / (bandpass.abs().sum(dim=1, keepdim=True) + 1e-8)

        # Reshape for conv1d: (out_channels, 1, kernel_size)
        return bandpass.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinc bandpass filter bank to raw waveform.

        Args:
            x: (B, T_samples) — raw audio waveform at sample_rate Hz.

        Returns:
            out: (B, T_frames, out_channels) where T_frames = T_samples // stride.
        """
        # Add channel dim for conv1d: (B, 1, T_samples)
        x = x.unsqueeze(1)

        # Pad to preserve frame count: padding = kernel_size // 2
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad))

        # Build filters on-the-fly from learnable parameters
        filters = self._build_filters()  # (C, 1, K)

        # Apply convolution
        out = F.conv1d(x, filters, stride=self.stride)  # (B, C, T_frames)

        # Transpose to (B, T_frames, C) for consistency with rest of model
        out = out.transpose(1, 2)

        return out
