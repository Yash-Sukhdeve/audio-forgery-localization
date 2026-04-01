"""Training callbacks for checkpointing, early stopping, and logging."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Abstract base callback."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> bool:
        """Return True to trigger early stopping."""
        return False


class CheckpointCallback(Callback):
    """Save model checkpoints periodically and track best model.

    Args:
        save_dir: Directory to save checkpoints.
        save_every_n_epochs: Save every N epochs.
        monitor: Metric name to monitor for best model.
        mode: 'min' or 'max' — whether lower or higher is better.
    """

    def __init__(
        self,
        save_dir: str,
        save_every_n_epochs: int = 5,
        monitor: str = "eer",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value

    def on_epoch_end(
        self, trainer, epoch, train_metrics, val_metrics
    ) -> bool:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Periodic save
        if (epoch + 1) % self.save_every_n_epochs == 0:
            path = str(self.save_dir / f"epoch_{epoch}.pt")
            trainer._save_checkpoint(path, epoch, val_metrics)
            logger.info("Checkpoint saved: %s", path)

        # Best model save
        current = val_metrics.get(self.monitor)
        if current is not None and self._is_better(current):
            self.best_value = current
            best_path = str(self.save_dir / "best.pt")
            trainer._save_checkpoint(best_path, epoch, val_metrics)
            trainer.best_path = best_path
            logger.info(
                "New best %s: %.6f (epoch %d)", self.monitor, current, epoch
            )

        return False


class EarlyStoppingCallback(Callback):
    """Stop training when monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        monitor: Metric name to monitor.
        mode: 'min' or 'max'.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "eer",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        return current > (self.best_value + self.min_delta)

    def on_epoch_end(
        self, trainer, epoch, train_metrics, val_metrics
    ) -> bool:
        current = val_metrics.get(self.monitor)
        if current is None:
            return False

        if self._is_better(current):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(
                    "Early stopping: %s did not improve for %d epochs",
                    self.monitor,
                    self.patience,
                )
                return True

        return False


class TensorBoardCallback(Callback):
    """Log training metrics to TensorBoard.

    Args:
        log_dir: TensorBoard log directory.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

    def on_train_start(self, trainer) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available; skipping TB logging")

    def on_epoch_end(
        self, trainer, epoch, train_metrics, val_metrics
    ) -> bool:
        if self.writer is None:
            return False

        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"validate/{k}", v, epoch)

        return False

    def on_train_end(self, trainer) -> None:
        if self.writer:
            self.writer.close()
