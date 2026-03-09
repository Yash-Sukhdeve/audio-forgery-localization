"""LlamaPartialSpoof dataset loader.

Dataset: H.-T. Luong, H. Li, L. Zhang, K. A. Lee, E. S. Chng,
"LlamaPartialSpoof: An LLM-driven fake speech dataset simulating
disinformation generation," Proc. IEEE ICASSP, 2025.

Structure:
    LlamaPartialSpoof/
    ├── R01TTS.0.a/              # Audio files (subset a: crossfade)
    ├── R01TTS.0.b/              # Audio files (subset b: cut/paste, overlap/add)
    ├── label_R01TTS.0.a.txt     # Labels for subset a
    └── label_R01TTS.0.b.txt     # Labels for subset b

Label format per line:
    <id> <duration> <utt_label> <start1>-<end1>-<label1> <start2>-<end2>-<label2> ...
    Labels: "bonafide" or "spoof"

Note: This dataset has NO train/dev/eval split. The FARA paper uses it
exclusively for cross-dataset evaluation (model trained on PartialSpoof).
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from core.data.base_dataset import BaseAudioDataset


def _parse_segments(segment_strings: List[str]) -> List[Tuple[float, float, str]]:
    """Parse segment strings like '0.0000-1.2345-bonafide' into tuples."""
    segments = []
    for seg in segment_strings:
        parts = seg.split("-")
        start = float(parts[0])
        end = float(parts[1])
        label = parts[2]
        segments.append((start, end, label))
    return segments


def _segments_to_frame_labels(
    segments: List[Tuple[float, float, str]],
    num_frames: int,
    frame_duration_ms: int = 20,
) -> torch.Tensor:
    """Convert time-based segment annotations to frame-level binary labels."""
    frame_duration_s = frame_duration_ms / 1000.0
    labels = torch.zeros(num_frames, dtype=torch.long)

    for start, end, label in segments:
        if label == "spoof":
            start_frame = int(start / frame_duration_s)
            end_frame = min(int(np.ceil(end / frame_duration_s)), num_frames)
            labels[start_frame:end_frame] = 1

    return labels


class LlamaPartialSpoofDataset(BaseAudioDataset):
    """LlamaPartialSpoof dataset loader.

    Args:
        root: Path to LlamaPartialSpoof/ directory.
        subset: 'a' (crossfade), 'b' (cut/paste), or 'both'.
        target_sr: Target sample rate (default 16000).
        frame_duration_ms: Frame duration in ms (default 20).
    """

    SUBSETS = {"a", "b", "both"}

    def __init__(
        self,
        root: str,
        subset: str = "both",
        target_sr: int = 16000,
        frame_duration_ms: int = 20,
    ):
        if subset not in self.SUBSETS:
            raise ValueError(f"Subset must be one of {self.SUBSETS}, got '{subset}'")

        self.root = Path(root)
        self.subset = subset
        self._audio_dirs: Dict[str, Path] = {}
        self._label_files: List[str] = []

        if subset in ("a", "both"):
            self._audio_dirs["a"] = self.root / "R01TTS.0.a"
            self._label_files.append("label_R01TTS.0.a.txt")
        if subset in ("b", "both"):
            self._audio_dirs["b"] = self.root / "R01TTS.0.b"
            self._label_files.append("label_R01TTS.0.b.txt")

        super().__init__(target_sr=target_sr, frame_duration_ms=frame_duration_ms)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        items = []
        for label_file in self._label_files:
            sub_key = "a" if "0.a" in label_file else "b"
            audio_dir = self._audio_dirs[sub_key]

            with open(self.root / label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    duration = float(parts[1])
                    utt_label = parts[2]
                    segment_strings = parts[3:]
                    segments = _parse_segments(segment_strings)

                    items.append({
                        "utt_id": utt_id,
                        "duration": duration,
                        "label": utt_label,
                        "segments": segments,
                        "audio_dir": str(audio_dir),
                    })
        return items

    def _get_audio_path(self, item: Dict[str, Any]) -> str:
        return str(Path(item["audio_dir"]) / f"{item['utt_id']}.wav")

    def _load_frame_labels(self, utt_id: str, num_frames: int) -> torch.Tensor:
        item = None
        for m in self._metadata:
            if m["utt_id"] == utt_id:
                item = m
                break

        if item is None:
            raise KeyError(f"Utterance '{utt_id}' not found in metadata")

        return _segments_to_frame_labels(
            item["segments"], num_frames, self.frame_duration_ms
        )
