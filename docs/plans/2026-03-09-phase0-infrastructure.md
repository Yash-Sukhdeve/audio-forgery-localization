# Phase 0: Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the modular core library (data loading, metrics, training utilities) that all methods will share.

**Architecture:** A `core/` package with abstract base classes for datasets, unified metric computation, and reusable training utilities. Dataset loaders for PartialSpoof and LlamaPartialSpoof convert each dataset's native format into a common representation: `(waveform: Tensor[samples], labels: Tensor[frames], boundaries: Tensor[frames], metadata: dict)`.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchaudio, numpy, scikit-learn (for EER), PyYAML

---

### Task 1: Project Scaffolding

**Files:**
- Create: `core/__init__.py`
- Create: `core/data/__init__.py`
- Create: `core/metrics/__init__.py`
- Create: `core/training/__init__.py`
- Create: `core/audio/__init__.py`
- Create: `core/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/data/__init__.py`
- Create: `tests/core/metrics/__init__.py`
- Create: `tests/core/audio/__init__.py`
- Create: `requirements.txt`
- Create: `pyproject.toml`

**Step 1: Create directory structure**

```bash
cd /media/lab2208/ssd/Explainablility/Localization
mkdir -p core/{data,metrics,training,audio,utils}
mkdir -p tests/core/{data,metrics,audio}
touch core/__init__.py core/data/__init__.py core/metrics/__init__.py
touch core/training/__init__.py core/audio/__init__.py core/utils/__init__.py
touch tests/__init__.py tests/core/__init__.py
touch tests/core/data/__init__.py tests/core/metrics/__init__.py tests/core/audio/__init__.py
```

**Step 2: Create requirements.txt**

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
pyyaml>=6.0
```

**Step 3: Create pyproject.toml**

```toml
[project]
name = "audio-forgery-localization"
version = "0.1.0"
description = "Audio forgery localization: FARA reimplementation and baseline comparison"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 4: Commit**

```bash
git add -A
git commit -m "scaffold: create project structure with core modules and test directories"
```

---

### Task 2: Audio I/O Module

**Files:**
- Create: `core/audio/io.py`
- Create: `tests/core/audio/test_io.py`

**Step 1: Write the failing tests**

```python
# tests/core/audio/test_io.py
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
        """Audio at 24kHz should be resampled to 16kHz."""
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
        # 1 second at 16kHz with 20ms frames = 50 frames
        assert get_num_frames(16000, sr=16000, frame_duration_ms=20) == 50

    def test_partial_frame_ceil(self):
        # 16100 samples = 1.00625s = 50.3125 frames → 51
        assert get_num_frames(16100, sr=16000, frame_duration_ms=20) == 51

    def test_exact_frames(self):
        # 32000 samples = 2s = 100 frames
        assert get_num_frames(32000, sr=16000, frame_duration_ms=20) == 100
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/lab2208/ssd/Explainablility/Localization && python -m pytest tests/core/audio/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.audio.io'`

**Step 3: Write minimal implementation**

```python
# core/audio/io.py
"""Audio I/O utilities. Shared by all dataset loaders and models."""
import math
from pathlib import Path

import torch
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and return 1D tensor at target sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, etc.)
        target_sr: Target sample rate in Hz.

    Returns:
        1D float32 tensor of shape (num_samples,).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def get_num_frames(num_samples: int, sr: int = 16000,
                   frame_duration_ms: int = 20) -> int:
    """Compute number of frames for a given audio length.

    Args:
        num_samples: Number of audio samples.
        sr: Sample rate in Hz.
        frame_duration_ms: Frame duration in milliseconds.

    Returns:
        Number of frames (ceiling division).
    """
    samples_per_frame = sr * frame_duration_ms // 1000
    return math.ceil(num_samples / samples_per_frame)
```

**Step 4: Run tests to verify they pass**

Run: `cd /media/lab2208/ssd/Explainablility/Localization && python -m pytest tests/core/audio/test_io.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add core/audio/io.py tests/core/audio/test_io.py
git commit -m "feat(core): add audio I/O module with loading and frame counting"
```

---

### Task 3: Seed & Reproducibility Utilities

**Files:**
- Create: `core/utils/seed.py`
- Create: `tests/core/test_utils.py`

**Step 1: Write the failing test**

```python
# tests/core/test_utils.py
import torch
import numpy as np
from core.utils.seed import set_seed


class TestSetSeed:
    def test_torch_deterministic(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_numpy_deterministic(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        assert np.array_equal(a, b)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_utils.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/utils/seed.py
"""Reproducibility utilities."""
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/utils/seed.py tests/core/test_utils.py
git commit -m "feat(core): add seed utility for reproducibility"
```

---

### Task 4: Config Loader

**Files:**
- Create: `core/utils/config.py`
- Create: `tests/core/test_config.py`

**Step 1: Write the failing test**

```python
# tests/core/test_config.py
import tempfile
import yaml
import pytest
from core.utils.config import load_config


class TestLoadConfig:
    def test_load_yaml(self):
        cfg = {"batch_size": 8, "lr": 1e-5, "model": {"name": "fara"}}
        path = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        yaml.dump(cfg, path)
        path.close()
        loaded = load_config(path.name)
        assert loaded.batch_size == 8
        assert loaded.lr == 1e-5
        assert loaded.model.name == "fara"

    def test_nested_access(self):
        cfg = {"a": {"b": {"c": 42}}}
        path = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        yaml.dump(cfg, path)
        path.close()
        loaded = load_config(path.name)
        assert loaded.a.b.c == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent.yaml")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/utils/config.py
"""Configuration loading utilities."""
from pathlib import Path

import yaml


class DotDict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = DotDict(val)
            self[key] = val
        return val

    def __setattr__(self, key, val):
        self[key] = val


def load_config(path: str) -> DotDict:
    """Load a YAML config file and return as DotDict.

    Args:
        path: Path to YAML file.

    Returns:
        DotDict with nested attribute access.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return DotDict(data)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/utils/config.py tests/core/test_config.py
git commit -m "feat(core): add YAML config loader with dot-access"
```

---

### Task 5: Metrics — EER Computation

**Files:**
- Create: `core/metrics/eer.py`
- Create: `tests/core/metrics/test_eer.py`

**Step 1: Write the failing tests**

```python
# tests/core/metrics/test_eer.py
import pytest
import numpy as np
from core.metrics.eer import compute_eer


class TestComputeEER:
    def test_perfect_separation(self):
        """Perfectly separated scores should give EER ~0."""
        genuine_scores = np.array([0.9, 0.8, 0.95, 0.85])
        spoof_scores = np.array([0.1, 0.2, 0.05, 0.15])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.concatenate([genuine_scores, spoof_scores])
        eer, threshold = compute_eer(scores, labels)
        assert eer < 0.05

    def test_random_scores(self):
        """Random scores should give EER ~50%."""
        np.random.seed(42)
        scores = np.random.rand(10000)
        labels = np.array([1] * 5000 + [0] * 5000)
        eer, _ = compute_eer(scores, labels)
        assert 0.4 < eer < 0.6

    def test_returns_threshold(self):
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1])
        eer, threshold = compute_eer(scores, labels)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            compute_eer(np.array([]), np.array([]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/metrics/test_eer.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/metrics/eer.py
"""Equal Error Rate computation.

Used by all methods for frame-level forgery localization evaluation.
Reference: FARA paper Section IV-B uses EER as primary metric.
"""
from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) from scores and binary labels.

    The EER is the point where False Acceptance Rate equals
    False Rejection Rate on the ROC curve.

    Args:
        scores: 1D array of prediction scores (higher = more likely genuine/spoof
                depending on convention). Shape: (N,).
        labels: 1D array of binary ground truth labels (1=spoof, 0=bonafide).
                Shape: (N,).

    Returns:
        Tuple of (eer, threshold):
            - eer: Equal Error Rate as a fraction (0.0 to 1.0).
            - threshold: Score threshold at EER operating point.

    Raises:
        ValueError: If inputs are empty or have mismatched shapes.
    """
    if len(scores) == 0 or len(labels) == 0:
        raise ValueError("Scores and labels must not be empty.")
    if len(scores) != len(labels):
        raise ValueError(
            f"Scores ({len(scores)}) and labels ({len(labels)}) must have same length."
        )

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER where FPR == FNR via interpolation
    eer = float(brentq(lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x), 0.0, 1.0))
    # Find threshold closest to EER
    idx = np.nanargmin(np.abs(fpr - eer))
    threshold = float(thresholds[idx])

    return eer, threshold
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/metrics/test_eer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/metrics/eer.py tests/core/metrics/test_eer.py
git commit -m "feat(core): add EER computation module"
```

---

### Task 6: Metrics — Classification (Precision, Recall, F1)

**Files:**
- Create: `core/metrics/classification.py`
- Create: `tests/core/metrics/test_classification.py`

**Step 1: Write the failing tests**

```python
# tests/core/metrics/test_classification.py
import pytest
import numpy as np
from core.metrics.classification import compute_frame_metrics


class TestComputeFrameMetrics:
    def test_perfect_predictions(self):
        preds = np.array([1, 1, 0, 0, 1])
        labels = np.array([1, 1, 0, 0, 1])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong(self):
        preds = np.array([0, 0, 1, 1])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_partial_match(self):
        # 2 TP, 1 FP, 1 FN
        preds = np.array([1, 1, 1, 0])
        labels = np.array([1, 1, 0, 1])
        metrics = compute_frame_metrics(preds, labels)
        assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-6)
        assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-6)

    def test_returns_all_keys(self):
        preds = np.array([1, 0])
        labels = np.array([1, 0])
        metrics = compute_frame_metrics(preds, labels)
        assert set(metrics.keys()) == {"precision", "recall", "f1"}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_frame_metrics(np.array([]), np.array([]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/metrics/test_classification.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/metrics/classification.py
"""Frame-level classification metrics.

Used by all methods: Precision, Recall, F1 at frame level.
Reference: FARA paper Section IV-B.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_frame_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute frame-level Precision, Recall, and F1.

    Args:
        predictions: 1D binary array of predicted labels. Shape: (N,).
        labels: 1D binary array of ground truth labels. Shape: (N,).
        pos_label: Label considered as positive (spoof). Default: 1.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.

    Raises:
        ValueError: If inputs are empty or mismatched.
    """
    if len(predictions) == 0 or len(labels) == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions ({len(predictions)}) and labels ({len(labels)}) "
            "must have same length."
        )

    return {
        "precision": float(precision_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
        "recall": float(recall_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
        "f1": float(f1_score(
            labels, predictions, pos_label=pos_label, zero_division=0.0
        )),
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/metrics/test_classification.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/metrics/classification.py tests/core/metrics/test_classification.py
git commit -m "feat(core): add frame-level classification metrics"
```

---

### Task 7: Unified Evaluation Entry Point

**Files:**
- Create: `core/metrics/evaluate.py`
- Create: `tests/core/metrics/test_evaluate.py`

**Step 1: Write the failing tests**

```python
# tests/core/metrics/test_evaluate.py
import pytest
import numpy as np
from core.metrics.evaluate import evaluate_localization


class TestEvaluateLocalization:
    def test_returns_all_metrics(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        result = evaluate_localization(scores, labels)
        assert "eer" in result
        assert "threshold" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_perfect_separation(self):
        scores = np.array([0.9, 0.85, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        result = evaluate_localization(scores, labels)
        assert result["eer"] < 0.05
        assert result["f1"] > 0.95

    def test_multi_utterance(self):
        """Evaluate across multiple utterances (concatenated)."""
        all_scores = []
        all_labels = []
        for _ in range(5):
            n = np.random.randint(50, 200)
            scores = np.random.rand(n)
            labels = (scores > 0.5).astype(int)  # near-perfect
            all_scores.append(scores)
            all_labels.append(labels)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        result = evaluate_localization(scores, labels)
        assert 0 <= result["eer"] <= 1
        assert 0 <= result["f1"] <= 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/metrics/test_evaluate.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/metrics/evaluate.py
"""Unified evaluation entry point for all methods.

Combines EER and classification metrics into a single call.
All methods (FARA, BAM, CFPRF, PSDS) use this for consistent evaluation.
"""
from typing import Dict

import numpy as np

from core.metrics.eer import compute_eer
from core.metrics.classification import compute_frame_metrics


def evaluate_localization(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate frame-level forgery localization.

    Computes EER from continuous scores, then binarizes predictions
    at the EER threshold to compute Precision, Recall, F1.

    Args:
        scores: 1D array of per-frame spoof scores. Shape: (N,).
                Higher values indicate more likely spoof.
        labels: 1D binary array of ground truth. Shape: (N,).
                1 = spoof, 0 = bonafide.

    Returns:
        Dict with keys: 'eer', 'threshold', 'precision', 'recall', 'f1'.
    """
    eer, threshold = compute_eer(scores, labels)

    # Binarize predictions at EER threshold
    binary_preds = (scores >= threshold).astype(int)

    frame_metrics = compute_frame_metrics(binary_preds, labels)

    return {
        "eer": eer,
        "threshold": threshold,
        **frame_metrics,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/metrics/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/metrics/evaluate.py tests/core/metrics/test_evaluate.py
git commit -m "feat(core): add unified evaluation entry point"
```

---

### Task 8: Base Dataset Class

**Files:**
- Create: `core/data/base_dataset.py`
- Create: `tests/core/data/test_base_dataset.py`

**Step 1: Write the failing tests**

```python
# tests/core/data/test_base_dataset.py
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

    def _load_metadata(self):
        return self._items

    def _load_frame_labels(self, utt_id: str, num_frames: int):
        # All spoof for utt_001, all bonafide for utt_002
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
            BaseAudioDataset()  # Can't instantiate abstract class

    def test_metadata_accessible(self):
        ds = DummyDataset()
        assert ds.metadata[0]["utt_id"] == "utt_001"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/data/test_base_dataset.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/data/base_dataset.py
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

from core.audio.io import load_audio, get_num_frames


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
            waveform.shape[0], sr=self.target_sr,
            frame_duration_ms=self.frame_duration_ms,
        )
        frame_labels = self._load_frame_labels(utt_id, num_frames)

        return {
            "waveform": waveform,
            "frame_labels": frame_labels,
            "utt_id": utt_id,
            "num_frames": num_frames,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/data/test_base_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/data/base_dataset.py tests/core/data/test_base_dataset.py
git commit -m "feat(core): add abstract base dataset class"
```

---

### Task 9: PartialSpoof Dataset Loader

**Files:**
- Create: `core/data/partialspoof.py`
- Create: `tests/core/data/test_partialspoof.py`

**Dataset structure (verified):**
```
/media/lab2208/ssd/datasets/PartialSpoof/database/
├── train/con_wav/          # 25,380 WAV files (CON_T_*.wav, LA_T_*.wav)
├── dev/con_wav/            # 24,844 WAV files
├── eval/con_wav/           # 71,237 WAV files
├── protocols/PartialSpoof_LA_cm_protocols/
│   ├── PartialSpoof.LA.cm.train.trl.txt   # Format: <spk> <utt_id> - <type> <label>
│   ├── PartialSpoof.LA.cm.dev.trl.txt
│   └── PartialSpoof.LA.cm.eval.trl.txt
└── segment_labels/
    ├── train_seglab_0.02.npy   # Dict[utt_id → array of '0'/'1' strings], 20ms
    ├── dev_seglab_0.02.npy
    └── eval_seglab_0.02.npy
```

Label convention: '1' = spoof, '0' = bonafide (verified from data inspection).

**Step 1: Write the failing tests**

```python
# tests/core/data/test_partialspoof.py
import pytest
import numpy as np
import torch
from pathlib import Path

from core.data.partialspoof import PartialSpoofDataset

# Paths verified from dataset inspection
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
        """Find a CON_ sample (known spoof) and verify it has both 0 and 1."""
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        # CON_T_0000000 is known to be partial spoof
        for i, item in enumerate(ds.metadata):
            if item["utt_id"] == "CON_T_0000000":
                sample = ds[i]
                unique = sample["frame_labels"].unique().tolist()
                assert 0 in unique and 1 in unique
                break

    def test_bonafide_sample_has_all_zeros(self):
        """Find a bonafide sample and verify all labels are 0."""
        ds = PartialSpoofDataset(root=PS_ROOT, split="train")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "bonafide":
                sample = ds[i]
                assert sample["frame_labels"].sum().item() == 0
                break
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/data/test_partialspoof.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/data/partialspoof.py
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

Label format in .npy: dict[utt_id → array of '0'/'1' strings]
    '0' = bonafide, '1' = spoof, resolution = 20ms
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
        """Parse protocol file to get utterance list with labels."""
        protocol_file = (
            self.root
            / self.PROTOCOL_DIR
            / f"PartialSpoof.LA.cm.{self.split}.trl.txt"
        )
        items = []
        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                # Format: <spk_id> <utt_id> - <type> <label>
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
        """Load 20ms frame-level labels from precomputed .npy.

        If utterance is bonafide (not in seg_labels dict or all '0'),
        returns all zeros. Truncates or pads to match num_frames.
        """
        if utt_id in self._seg_labels:
            raw = self._seg_labels[utt_id]
            labels = np.array([int(x) for x in raw], dtype=np.int64)
        else:
            labels = np.zeros(num_frames, dtype=np.int64)

        # Align to actual audio length
        if len(labels) >= num_frames:
            labels = labels[:num_frames]
        else:
            # Pad with last label value (typically bonafide=0 at end)
            pad = np.zeros(num_frames - len(labels), dtype=np.int64)
            labels = np.concatenate([labels, pad])

        return torch.from_numpy(labels)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/data/test_partialspoof.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add core/data/partialspoof.py tests/core/data/test_partialspoof.py
git commit -m "feat(core): add PartialSpoof dataset loader"
```

---

### Task 10: LlamaPartialSpoof Dataset Loader

**Files:**
- Create: `core/data/llamaspoof.py`
- Create: `tests/core/data/test_llamaspoof.py`

**Dataset structure (verified):**
```
/media/lab2208/ssd/datasets/LlamaPartialSpoof/
├── R01TTS.0.a/               # 76,228 WAV files (bonafide + full + partial with crossfade)
├── R01TTS.0.b/               # 64,388 WAV files (partial with cut/paste or overlap/add)
├── label_R01TTS.0.a.txt      # 76,228 lines
├── label_R01TTS.0.b.txt      # 64,388 lines
├── metadata_crossfade.csv    # Crossfade function metadata
└── README.txt
```

Label format per line: `<id> <duration> <utt_label> <seg1_start>-<seg1_end>-<seg1_label> ...`
Labels: "bonafide" or "spoof". No train/dev/eval split provided — FARA paper uses it for cross-dataset eval only (no training on LlamaPS).

**Step 1: Write the failing tests**

```python
# tests/core/data/test_llamaspoof.py
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
        """Find a partial-spoof sample and verify mixed labels."""
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "spoof" and len(item["segments"]) > 1:
                sample = ds[i]
                unique = sample["frame_labels"].unique().tolist()
                if 0 in unique and 1 in unique:
                    return  # Found one
        pytest.fail("No partial spoof sample with mixed labels found")

    def test_bonafide_has_all_zeros(self):
        ds = LlamaPartialSpoofDataset(root=LLAMA_ROOT, subset="a")
        for i, item in enumerate(ds.metadata):
            if item["label"] == "bonafide":
                sample = ds[i]
                assert sample["frame_labels"].sum().item() == 0
                break
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/data/test_llamaspoof.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/data/llamaspoof.py
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
        # Format: start-end-label (label may be 'bonafide' or 'spoof')
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
    """Convert time-based segment annotations to frame-level binary labels.

    Args:
        segments: List of (start_sec, end_sec, label_str).
        num_frames: Total number of frames.
        frame_duration_ms: Frame duration in milliseconds.

    Returns:
        LongTensor of shape (num_frames,), 0=bonafide, 1=spoof.
    """
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
            # Determine which subset this file belongs to
            sub_key = "a" if "0.a" in label_file else "b"
            audio_dir = self._audio_dirs[sub_key]

            with open(self.root / label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    duration = float(parts[1])
                    utt_label = parts[2]  # 'bonafide' or 'spoof'
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
        """Convert segment annotations to frame-level labels."""
        # Find the item by utt_id
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/data/test_llamaspoof.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add core/data/llamaspoof.py tests/core/data/test_llamaspoof.py
git commit -m "feat(core): add LlamaPartialSpoof dataset loader"
```

---

### Task 11: Collation Utility

**Files:**
- Create: `core/data/collate.py`
- Create: `tests/core/data/test_collate.py`

**Step 1: Write the failing tests**

```python
# tests/core/data/test_collate.py
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
        # First sample padded: mask should be True for valid, False for padded
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
        # Padded region of first sample's labels should be -1 (ignore index)
        assert (collated["frame_labels"][0, 50:] == -1).all()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/core/data/test_collate.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# core/data/collate.py
"""Collation utilities for batching variable-length audio samples.

Used by all dataset loaders via DataLoader(collate_fn=pad_collate).
"""
from typing import Any, Dict, List

import torch


def pad_collate(
    batch: List[Dict[str, Any]],
    label_pad_value: int = -1,
) -> Dict[str, Any]:
    """Collate variable-length samples with zero-padding for waveforms
    and ignore-index padding for labels.

    Args:
        batch: List of sample dicts from BaseAudioDataset.__getitem__.
        label_pad_value: Value used to pad frame_labels (default -1,
                         compatible with CrossEntropyLoss ignore_index).

    Returns:
        Dict with:
            'waveforms': Tensor[B, max_samples], zero-padded
            'frame_labels': Tensor[B, max_frames], padded with label_pad_value
            'lengths': Tensor[B], original waveform lengths
            'frame_lengths': Tensor[B], original frame counts
            'utt_ids': List[str]
    """
    waveforms = [s["waveform"] for s in batch]
    frame_labels = [s["frame_labels"] for s in batch]
    utt_ids = [s["utt_id"] for s in batch]

    # Waveform lengths
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = lengths.max().item()

    # Pad waveforms with zeros
    padded_waveforms = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, : w.shape[0]] = w

    # Frame label lengths
    frame_lengths = torch.tensor(
        [fl.shape[0] for fl in frame_labels], dtype=torch.long
    )
    max_frames = frame_lengths.max().item()

    # Pad frame labels with ignore index
    padded_labels = torch.full(
        (len(batch), max_frames), label_pad_value, dtype=torch.long
    )
    for i, fl in enumerate(frame_labels):
        padded_labels[i, : fl.shape[0]] = fl

    return {
        "waveforms": padded_waveforms,
        "frame_labels": padded_labels,
        "lengths": lengths,
        "frame_lengths": frame_lengths,
        "utt_ids": utt_ids,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/core/data/test_collate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/data/collate.py tests/core/data/test_collate.py
git commit -m "feat(core): add pad collation for variable-length batching"
```

---

### Task 12: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
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
        """Smoke test: random scores through evaluation pipeline."""
        ds = PartialSpoofDataset(root=PS_ROOT, split="dev")
        # Collect first 100 samples' labels
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
```

**Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v --timeout=120`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for data pipeline and evaluation"
```

---

### Task 13: Phase 0 Gate — Verify All Tests Pass

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 28 tests total)

**Step 2: Verify module imports work cleanly**

```bash
python -c "
from core.audio.io import load_audio, get_num_frames
from core.data.partialspoof import PartialSpoofDataset
from core.data.llamaspoof import LlamaPartialSpoofDataset
from core.data.collate import pad_collate
from core.metrics.evaluate import evaluate_localization
from core.utils.seed import set_seed
from core.utils.config import load_config
print('All imports OK')
"
```
Expected: "All imports OK"

**Step 3: Final commit for Phase 0**

```bash
git add -A
git commit -m "phase0: complete core infrastructure - data, metrics, utils"
```

---

## Phase 0 Completion Gate Checklist

- [ ] All 13 tasks completed
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Integration tests pass with real data
- [ ] PartialSpoof loader: correct split sizes (25380/24844/71237)
- [ ] LlamaPartialSpoof loader: correct subset sizes (76228/64388)
- [ ] Metrics verified: EER, Precision, Recall, F1
- [ ] No duplicate functionality — all modules reusable
- [ ] Code committed to git

**After Phase 0 is complete, proceed to Phase 1 (BAM Baseline) plan.**
