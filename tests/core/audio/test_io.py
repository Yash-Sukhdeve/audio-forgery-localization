import pytest
import torch
import numpy as np
import tempfile
import soundfile as sf
from core.audio.io import load_audio, get_num_frames


class TestLoadAudio:
    def _make_wav(self, sr=16000, duration=1.0):
        """Create a temporary WAV file for testing."""
        samples = np.random.randn(int(sr * duration)).astype(np.float32)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, samples, sr)
        return path, samples

    def test_load_returns_tensor(self):
        path, _ = self._make_wav()
        waveform = load_audio(path)
        assert isinstance(waveform, torch.Tensor)

    def test_load_correct_shape(self):
        path, samples = self._make_wav(sr=16000, duration=2.0)
        waveform = load_audio(path, target_sr=16000)
        assert waveform.dim() == 1
        assert waveform.shape[0] == 32000

    def test_load_resamples(self):
        samples = np.random.randn(24000).astype(np.float32)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, samples, 24000)
        waveform = load_audio(path, target_sr=16000)
        assert waveform.shape[0] == 16000

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/file.wav")


class TestGetNumFrames:
    def test_basic_frame_count(self):
        assert get_num_frames(16000, sr=16000, frame_duration_ms=20) == 50

    def test_partial_frame_ceil(self):
        assert get_num_frames(16100, sr=16000, frame_duration_ms=20) == 51

    def test_exact_frames(self):
        assert get_num_frames(32000, sr=16000, frame_duration_ms=20) == 100
