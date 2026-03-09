import pytest
import torch
from core.data.base_dataset import BaseAudioDataset


class DummyDataset(BaseAudioDataset):
    """Concrete implementation for testing."""

    def __init__(self):
        self._items = [
            {
                "utt_id": "utt_001",
                "audio_path": "/tmp/fake.wav",
                "duration": 2.0,
                "label": "spoof",
            },
            {
                "utt_id": "utt_002",
                "audio_path": "/tmp/fake2.wav",
                "duration": 1.0,
                "label": "bonafide",
            },
        ]
        super().__init__()

    def _load_metadata(self):
        return self._items

    def _load_frame_labels(self, utt_id: str, num_frames: int):
        if utt_id == "utt_001":
            return torch.ones(num_frames, dtype=torch.long)
        return torch.zeros(num_frames, dtype=torch.long)

    def _get_audio_path(self, item: dict) -> str:
        return item["audio_path"]


class TestBaseDataset:
    def test_len(self):
        ds = DummyDataset()
        assert len(ds) == 2

    def test_abstract_methods_enforced(self):
        with pytest.raises(TypeError):
            BaseAudioDataset()

    def test_metadata_accessible(self):
        ds = DummyDataset()
        assert ds.metadata[0]["utt_id"] == "utt_001"
