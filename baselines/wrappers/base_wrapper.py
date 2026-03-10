"""Abstract base wrapper for all baseline methods."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class BaseWrapper(ABC):
    """Abstract wrapper for baseline methods.

    Wrappers invoke original repo scripts as subprocesses.
    They NEVER import or modify repo internals.
    """

    def __init__(self, repo_dir: str, experiment_name: str):
        self.repo_dir = Path(repo_dir)
        self.experiment_name = experiment_name

    @abstractmethod
    def train(self, **kwargs) -> Path:
        """Train the model. Returns path to best checkpoint."""
        ...

    @abstractmethod
    def evaluate(self, checkpoint: str, split: str, **kwargs) -> Dict[str, float]:
        """Evaluate on a given split. Returns metrics dict."""
        ...
