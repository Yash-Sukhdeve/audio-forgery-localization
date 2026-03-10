import pytest
import numpy as np
from pathlib import Path

PS_ROOT = Path("/media/lab2208/ssd/datasets/PartialSpoof/database")
BAM_DATA = Path("/media/lab2208/ssd/Explainablility/Localization/baselines/repos/BAM/data")


@pytest.mark.skipif(not BAM_DATA.exists(), reason="BAM data not prepared")
class TestBAMDataPrep:
    def test_audio_symlinks_exist(self):
        for split in ["train", "dev", "eval"]:
            path = BAM_DATA / "raw" / split
            assert path.exists(), f"Missing symlink: {path}"
            assert path.is_symlink() or path.is_dir()

    def test_segment_labels_exist(self):
        for split in ["train", "dev", "eval"]:
            assert (BAM_DATA / f"{split}_seglab_0.02.npy").exists()

    def test_boundary_labels_exist(self):
        boundary_dir = BAM_DATA / "boundary"
        assert boundary_dir.exists()
        npy_files = list(boundary_dir.glob("*_boundary.npy"))
        # train=25380, dev=24844, eval=71237 utterances total in labels
        assert len(npy_files) > 100000

    def test_boundary_label_values(self):
        boundary_dir = BAM_DATA / "boundary"
        npy_files = list(boundary_dir.glob("CON_*_boundary.npy"))
        # Check a spoof utterance has some boundaries
        found_boundary = False
        for f in npy_files[:100]:
            labels = np.load(str(f))
            assert set(np.unique(labels)).issubset({0, 1})
            if 1 in labels:
                found_boundary = True
        assert found_boundary, "No boundary labels found in spoof utterances"
