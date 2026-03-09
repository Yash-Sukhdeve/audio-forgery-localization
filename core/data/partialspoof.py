"""PartialSpoof (ASVspoof2019 PartialSpoof) dataset loader.

Dataset: L. Zhang, X. Wang, E. Cooper, N. Evans, J. Yamagishi,
"The PartialSpoof database and countermeasures for the detection of short
fake speech segments embedded in an utterance,"
IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, 2023.

Structure:
    database/
    ├── {split}/con_wav/*.wav
    ├── protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt
    └── segment_labels/{split}_seglab_0.02.npy

Label format in .npy: dict[utt_id -> array of '0'/'1' strings]
    '1' = bonafide, '0' = spoof (in raw npy)
    Output convention: 0 = bonafide, 1 = spoof (inverted for consistency)
    Resolution = 20ms
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from core.data.base_dataset import BaseAudioDataset


class PartialSpoofDataset(BaseAudioDataset):
    """PartialSpoof dataset loader.

    Args:
        root: Path to database/ directory.
        split: One of 'train', 'dev', 'eval'.
        target_sr: Target sample rate (default 16000).
        frame_duration_ms: Frame duration in ms (default 20, matching paper).
    """

    SPLITS = {"train", "dev", "eval"}
    PROTOCOL_DIR = "protocols/PartialSpoof_LA_cm_protocols"
    LABEL_RESOLUTION = 0.02  # 20ms

    def __init__(
        self,
        root: str,
        split: str,
        target_sr: int = 16000,
        frame_duration_ms: int = 20,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"Split must be one of {self.SPLITS}, got '{split}'")

        self.root = Path(root)
        self.split = split
        self._audio_dir = self.root / split / "con_wav"

        # Load segment labels (20ms resolution)
        label_path = self.root / "segment_labels" / f"{split}_seglab_{self.LABEL_RESOLUTION:.2f}.npy"
        self._seg_labels: Dict[str, np.ndarray] = np.load(
            str(label_path), allow_pickle=True
        ).item()

        super().__init__(target_sr=target_sr, frame_duration_ms=frame_duration_ms)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        protocol_file = (
            self.root
            / self.PROTOCOL_DIR
            / f"PartialSpoof.LA.cm.{self.split}.trl.txt"
        )
        items = []
        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                spk_id = parts[0]
                utt_id = parts[1]
                utt_label = parts[4]  # 'spoof' or 'bonafide'
                items.append({
                    "utt_id": utt_id,
                    "spk_id": spk_id,
                    "label": utt_label,
                })
        return items

    def _get_audio_path(self, item: Dict[str, Any]) -> str:
        return str(self._audio_dir / f"{item['utt_id']}.wav")

    def _load_frame_labels(self, utt_id: str, num_frames: int) -> torch.Tensor:
        if utt_id in self._seg_labels:
            raw = self._seg_labels[utt_id]
            # Raw npy: '1' = bonafide, '0' = spoof
            # Invert to output convention: 0 = bonafide, 1 = spoof
            labels = np.array([1 - int(x) for x in raw], dtype=np.int64)
        else:
            labels = np.zeros(num_frames, dtype=np.int64)

        # Align to actual audio length
        if len(labels) >= num_frames:
            labels = labels[:num_frames]
        else:
            pad = np.zeros(num_frames - len(labels), dtype=np.int64)
            labels = np.concatenate([labels, pad])

        return torch.from_numpy(labels)
