"""Integration test: load real data, run through evaluation pipeline."""
import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from core.data.partialspoof import PartialSpoofDataset
from core.data.llamaspoof import LlamaPartialSpoofDataset
from core.data.collate import pad_collate
from core.metrics.evaluate import evaluate_localization

PS_ROOT = "/media/lab2208/ssd/datasets/PartialSpoof/database"
LLAMA_ROOT = "/media/lab2208/ssd/datasets/LlamaPartialSpoof"


@pytest.mark.skipif(not Path(PS_ROOT).exists(), reason="PartialSpoof not available")
class TestPartialSpoofIntegration:
    def test_dataloader_iterates(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="dev")
        dl = DataLoader(ds, batch_size=4, collate_fn=pad_collate, num_workers=0)
        batch = next(iter(dl))
        assert batch["waveforms"].shape[0] == 4
        assert batch["frame_labels"].shape[0] == 4
        assert len(batch["utt_ids"]) == 4

    def test_evaluation_with_random_scores(self):
        ds = PartialSpoofDataset(root=PS_ROOT, split="dev")
        all_labels = []
        for i in range(100):
            sample = ds[i]
            all_labels.append(sample["frame_labels"].numpy())
        labels = np.concatenate(all_labels)
        scores = np.random.rand(len(labels))
        result = evaluate_localization(scores, labels)
        assert "eer" in result
        assert "f1" in result
        assert 0 <= result["eer"] <= 1


@pytest.mark.skipif(not Path(LLAMA_ROOT).exists(), reason="LlamaPS not available")
class TestLlamaPSIntegration:
    def test_dataloader_iterates(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        dl = DataLoader(ds, batch_size=4, collate_fn=pad_collate, num_workers=0)
        batch = next(iter(dl))
        assert batch["waveforms"].shape[0] == 4
        assert batch["frame_labels"].shape[0] == 4
