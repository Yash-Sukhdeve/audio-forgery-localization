"""Abstract base dataset class for all audio forgery localization datasets.

All dataset loaders (PartialSpoof, LlamaPS, HQ-MPSD, PartialEdit) extend
this class to provide a unified interface.

Common output format per sample:
    {
        "waveform": Tensor[num_samples],        # Raw audio at 16kHz
        "frame_labels": Tensor[num_frames],      # 0=bonafide, 1=spoof per frame
        "utt_id": str,                           # Utterance identifier
        "num_frames": int,                       # Number of frames
    }
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from core.audio.io import get_num_frames, load_audio


class BaseAudioDataset(ABC, Dataset):
    """Abstract base class for audio forgery localization datasets.

    Subclasses must implement:
        - _load_metadata(): Returns list of item dicts.
        - _load_frame_labels(utt_id, num_frames): Returns frame-level labels.
        - _get_audio_path(item): Returns path to audio file.
    """

    def __init__(self, target_sr: int = 16000, frame_duration_ms: int = 20):
        self.target_sr = target_sr
        self.frame_duration_ms = frame_duration_ms
        self._metadata: List[Dict[str, Any]] = self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load dataset metadata. Each item must have at least 'utt_id'."""
        ...

    @abstractmethod
    def _load_frame_labels(self, utt_id: str, num_frames: int) -> torch.Tensor:
        """Load frame-level labels for a given utterance.

        Args:
            utt_id: Utterance identifier.
            num_frames: Expected number of frames.

        Returns:
            LongTensor of shape (num_frames,) with 0=bonafide, 1=spoof.
        """
        ...

    @abstractmethod
    def _get_audio_path(self, item: Dict[str, Any]) -> str:
        """Return the file path for an item's audio."""
        ...

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._metadata[idx]
        utt_id = item["utt_id"]
        audio_path = self._get_audio_path(item)

        waveform = load_audio(audio_path, target_sr=self.target_sr)
        num_frames = get_num_frames(
            waveform.shape[0],
            sr=self.target_sr,
            frame_duration_ms=self.frame_duration_ms,
        )
        frame_labels = self._load_frame_labels(utt_id, num_frames)

        return {
            "waveform": waveform,
            "frame_labels": frame_labels,
            "utt_id": utt_id,
            "num_frames": num_frames,
        }
