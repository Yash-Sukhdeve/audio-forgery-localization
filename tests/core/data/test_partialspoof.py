import pytest
import numpy as np
import torch
from pathlib import Path

from core.data.partialspoof import PartialSpoofDataset

PS_ROOT = "/media/lab2208/ssd/datasets/PartialSpoof/database"


@pytest.mark.skipif(
    not Path(PS_ROOT).exists(), reason="PartialSpoof dataset not available"
)
class TestPartialSpoofDataset:
    def test_train_split_length(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        assert len(ds) == 25380

    def test_dev_split_length(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="dev")
        assert len(ds) == 24844

    def test_eval_split_length(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="eval")
        assert len(ds) == 71237

    def test_getitem_returns_expected_keys(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        sample = ds[0]
        assert "waveform" in sample
        assert "frame_labels" in sample
        assert "utt_id" in sample
        assert "num_frames" in sample

    def test_waveform_is_1d_tensor(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        sample = ds[0]
        assert isinstance(sample["waveform"], torch.Tensor)
        assert sample["waveform"].dim() == 1

    def test_frame_labels_are_binary(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        sample = ds[0]
        labels = sample["frame_labels"]
        assert set(labels.unique().tolist()).issubset({0, 1})

    def test_frame_count_matches_labels(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        sample = ds[0]
        assert sample["frame_labels"].shape[0] == sample["num_frames"]

    def test_spoof_sample_has_mixed_labels(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        for i, item in enumerate(ds.metadata):
            if item["utt_id"] == "CON_T_0000000":
                sample = ds[i]
                unique = sample["frame_labels"].unique().tolist()
                assert 0 in unique and 1 in unique
                break

    def test_bonafide_sample_has_all_zeros(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "bonafide":
                sample = ds[i]
                assert sample["frame_labels"].sum().item() == 0
                break
