"""Generic training loop for audio forgery localization models.

Reusable across FARA and future custom models. Handles gradient clipping,
mixed precision, checkpointing, and early stopping via callbacks.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core.metrics.evaluate import evaluate_localization

logger = logging.getLogger(__name__)


class Trainer:
    """Generic training loop with callback support.

    Args:
        model: The model to train (must return dict with logits/features).
        optimizer: PyTorch optimizer.
        loss_fn: Loss function accepting (model_output, frame_labels, boundary_labels).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config with grad_clip, etc.
        callbacks: List of Callback instances.
        use_amp: Whether to use automatic mixed precision.
        device: Device to train on (default: auto-detect).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        callbacks: Optional[List] = None,
        use_amp: bool = True,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks or []
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.grad_clip = getattr(config, "grad_clip", 1.0)
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.best_path: Optional[str] = None

    def _forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Run model forward pass on a batch.

        Subclass or override for models needing special input handling
        (e.g., WavLM feature extraction).
        """
        wavlm_states = batch["wavlm_hidden_states"].to(self.device)
        waveform = batch["waveforms"].to(self.device)
        return self.model(wavlm_states, waveform)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Returns dict of average loss components."""
        self.model.train()
        running = {}
        n_batches = 0

        for batch in self.train_loader:
            frame_labels = batch["frame_labels"].to(self.device)
            boundary_labels = batch.get(
                "boundary_labels",
                torch.zeros_like(frame_labels),
            ).to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self._forward_step(batch)
                    losses = self.loss_fn(output, frame_labels, boundary_labels)
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self._forward_step(batch)
                losses = self.loss_fn(output, frame_labels, boundary_labels)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

            # Accumulate
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation. Returns metrics dict including EER and F1."""
        self.model.eval()
        all_scores = []
        all_labels = []
        running = {}
        n_batches = 0

        for batch in self.val_loader:
            frame_labels = batch["frame_labels"].to(self.device)
            boundary_labels = batch.get(
                "boundary_labels",
                torch.zeros_like(frame_labels),
            ).to(self.device)

            output = self._forward_step(batch)
            losses = self.loss_fn(output, frame_labels, boundary_labels)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

            # Collect spoof scores for EER/F1 computation
            # Use softmax probability of spoof class as score
            spoof_probs = torch.softmax(output["spoof_logits"], dim=-1)[
                :, :, 1
            ]  # (B, T)
            lengths = batch["frame_lengths"].to(self.device)

            for i in range(spoof_probs.shape[0]):
                t = lengths[i].item()
                all_scores.append(spoof_probs[i, :t].cpu().numpy())
                all_labels.append(frame_labels[i, :t].cpu().numpy())

        avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}

        # Compute EER and F1 from aggregated predictions
        if all_scores:
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            # Filter padding
            valid = labels != -1
            if valid.sum() > 0:
                eval_metrics = evaluate_localization(scores[valid], labels[valid])
                avg_losses.update(eval_metrics)

        return avg_losses

    def fit(self, max_epochs: int, patience: int = 10) -> Optional[str]:
        """Run full training loop with early stopping.

        Args:
            max_epochs: Maximum number of epochs.
            patience: Stop if val metric doesn't improve for this many epochs.

        Returns:
            Path to best checkpoint, or None.
        """
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            logger.info(
                "Epoch %d — train_loss: %.5f | val_loss: %.5f | val_eer: %.4f | val_f1: %.4f",
                epoch,
                train_metrics.get("loss", 0),
                val_metrics.get("loss", 0),
                val_metrics.get("eer", 0),
                val_metrics.get("f1", 0),
            )

            # Check improvement
            monitor_val = val_metrics.get("eer", float("inf"))
            if monitor_val < self.best_metric:
                self.best_metric = monitor_val

            for cb in self.callbacks:
                should_stop = cb.on_epoch_end(
                    self, epoch, train_metrics, val_metrics
                )
                if should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    for cb2 in self.callbacks:
                        cb2.on_train_end(self)
                    return self.best_path

        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.best_path

    def _save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_metric": self.best_metric,
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        return ckpt.get("metrics", {})
