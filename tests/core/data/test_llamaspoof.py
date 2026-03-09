import pytest
import torch
from pathlib import Path

from core.data.llamaspoof import LlamaPartialSpoofDataset

LLAMA_ROOT = "/media/lab2208/ssd/datasets/LlamaPartialSpoof"


@pytest.mark.skipif(
    not Path(LLAMA_ROOT).exists(), reason="LlamaPartialSpoof not available"
)
class TestLlamaPartialSpoofDataset:
    def test_subset_a_length(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        assert len(ds) == 76228

    def test_subset_b_length(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="b")
        assert len(ds) == 64388

    def test_both_subsets_length(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="both")
        assert len(ds) == 76228 + 64388

    def test_getitem_returns_expected_keys(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        sample = ds[0]
        assert "waveform" in sample
        assert "frame_labels" in sample
        assert "utt_id" in sample
        assert "num_frames" in sample

    def test_waveform_is_1d_tensor(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        sample = ds[0]
        assert isinstance(sample["waveform"], torch.Tensor)
        assert sample["waveform"].dim() == 1

    def test_frame_labels_are_binary(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        sample = ds[0]
        assert set(sample["frame_labels"].unique().tolist()).issubset({0, 1})

    def test_partial_spoof_has_mixed_labels(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "spoof" and len(item["segments"]) > 1:
                sample = ds[i]
                unique = sample["frame_labels"].unique().tolist()
                if 0 in unique and 1 in unique:
                    return
        pytest.fail("No partial spoof sample with mixed labels found")

    def test_bonafide_has_all_zeros(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "bonafide":
                sample = ds[i]
                assert sample["frame_labels"].sum().item() == 0
                break
