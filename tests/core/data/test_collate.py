import pytest
import torch
from core.data.collate import pad_collate


class TestPadCollate:
    def test_pads_waveforms(self):
        batch = [
            {"waveform": torch.randn(16000), "frame_labels": torch.zeros(50),
             "utt_id": "a", "num_frames": 50},
            {"waveform": torch.randn(32000), "frame_labels": torch.zeros(100),
             "utt_id": "b", "num_frames": 100},
        ]
        collated = pad_collate(batch)
        assert collated["waveforms"].shape == (2, 32000)
        assert collated["frame_labels"].shape == (2, 100)

    def test_padding_mask(self):
        batch = [
            {"waveform": torch.randn(16000), "frame_labels": torch.zeros(50),
             "utt_id": "a", "num_frames": 50},
            {"waveform": torch.randn(32000), "frame_labels": torch.ones(100),
             "utt_id": "b", "num_frames": 100},
        ]
        collated = pad_collate(batch)
        assert collated["lengths"][0] == 16000
        assert collated["lengths"][1] == 32000
        assert collated["frame_lengths"][0] == 50
        assert collated["frame_lengths"][1] == 100

    def test_preserves_utt_ids(self):
        batch = [
            {"waveform": torch.randn(16000), "frame_labels": torch.zeros(50),
             "utt_id": "a", "num_frames": 50},
        ]
        collated = pad_collate(batch)
        assert collated["utt_ids"] == ["a"]

    def test_label_padding_uses_ignore_index(self):
        batch = [
            {"waveform": torch.randn(16000), "frame_labels": torch.ones(50),
             "utt_id": "a", "num_frames": 50},
            {"waveform": torch.randn(32000), "frame_labels": torch.ones(100),
             "utt_id": "b", "num_frames": 100},
        ]
        collated = pad_collate(batch)
        assert (collated["frame_labels"][0, 50:] == -1).all()
