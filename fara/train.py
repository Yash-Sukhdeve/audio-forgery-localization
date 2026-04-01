"""FARA training entry point.

Usage:
    python -m fara.train --config configs/fara.yaml

Architecture:
  WavLM-Large (frozen) extracts 24-layer hidden states per frame.
  FARA (trainable) processes these via LearnableMask, SincNet, CMoE, etc.
  Boundary labels are generated on-the-fly from frame-level spoof labels.

Reference: Luo et al., IEEE/ACM TASLP 2026, Section IV.
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core.data.boundary import generate_boundary_labels
from core.data.collate import pad_collate
from core.data.partialspoof import PartialSpoofDataset
from core.metrics.evaluate import evaluate_localization
from core.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
)
from core.utils.config import load_config
from core.utils.seed import set_seed
from fara.losses.combined_loss import CombinedLoss
from fara.model.fara import FARA
from fara.model.wavlm_extractor import WavLMExtractor

logger = logging.getLogger(__name__)


class FARATrainer:
    """FARA-specific trainer with WavLM feature extraction.

    WavLM is frozen and runs outside the training graph.
    Only FARA parameters receive gradients.
    """

    def __init__(
        self,
        wavlm: WavLMExtractor,
        model: FARA,
        optimizer: torch.optim.Optimizer,
        loss_fn: CombinedLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        callbacks: list,
        device: str = "cuda",
    ):
        self.device = device
        self.wavlm = wavlm.to(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks

        self.grad_clip = getattr(config, "grad_clip", 1.0)
        self.use_amp = device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.log_interval = getattr(config, "log_interval", 50)

        self.current_epoch = 0
        self.best_metric = float("inf")
        self.best_path = None

    def _extract_features(self, batch: Dict) -> tuple:
        """Extract WavLM features and prepare inputs for FARA.

        Returns:
            wavlm_states: (B, T, N, D) WavLM hidden states.
            waveform: (B, T_samples) raw audio.
            frame_labels: (B, T) spoof labels.
            boundary_labels: (B, T) boundary labels.
        """
        waveform = batch["waveforms"].to(self.device)
        frame_labels = batch["frame_labels"].to(self.device)
        frame_lengths = batch["frame_lengths"]

        # Generate boundary labels on-the-fly
        boundary_labels = generate_boundary_labels(frame_labels)

        # Extract frozen WavLM features
        with torch.no_grad():
            wavlm_states = self.wavlm(waveform)  # (B, T_wavlm, N, D)

        return wavlm_states, waveform, frame_labels, boundary_labels, frame_lengths

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.wavlm.eval()  # Always eval mode for frozen backbone
        running = {}
        n_batches = 0

        for i, batch in enumerate(self.train_loader):
            wavlm_states, waveform, frame_labels, boundary_labels, _ = (
                self._extract_features(batch)
            )

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(wavlm_states, waveform)
                    # Align labels to model output time dimension
                    T = output["spoof_logits"].shape[1]
                    losses = self.loss_fn(
                        output, frame_labels[:, :T], boundary_labels[:, :T]
                    )
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(wavlm_states, waveform)
                T = output["spoof_logits"].shape[1]
                losses = self.loss_fn(
                    output, frame_labels[:, :T], boundary_labels[:, :T]
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

            if (i + 1) % self.log_interval == 0:
                avg_loss = running["loss"] / n_batches
                logger.info(
                    "  Step %d/%d — loss: %.5f",
                    i + 1, len(self.train_loader), avg_loss,
                )

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute EER/F1."""
        self.model.eval()
        self.wavlm.eval()
        all_scores, all_labels = [], []
        running = {}
        n_batches = 0

        for batch in self.val_loader:
            wavlm_states, waveform, frame_labels, boundary_labels, frame_lengths = (
                self._extract_features(batch)
            )

            output = self.model(wavlm_states, waveform)
            T = output["spoof_logits"].shape[1]
            losses = self.loss_fn(
                output, frame_labels[:, :T], boundary_labels[:, :T]
            )

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

            # Collect scores for EER
            spoof_probs = torch.softmax(output["spoof_logits"], dim=-1)[:, :, 1]
            for i in range(spoof_probs.shape[0]):
                t = min(frame_lengths[i].item(), T)
                all_scores.append(spoof_probs[i, :t].cpu().numpy())
                all_labels.append(frame_labels[i, :t].cpu().numpy())

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        if all_scores:
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            valid = labels != -1
            scores_valid = scores[valid]
            labels_valid = labels[valid]
            # Guard against NaN from AMP — replace with 0.5 (neutral)
            nan_mask = np.isnan(scores_valid) | np.isinf(scores_valid)
            if nan_mask.any():
                logger.warning(
                    "Found %d NaN/Inf in scores (%.2f%%), replacing with 0.5",
                    nan_mask.sum(), nan_mask.mean() * 100,
                )
                scores_valid = np.where(nan_mask, 0.5, scores_valid)
            if len(scores_valid) > 0:
                avg.update(evaluate_localization(scores_valid, labels_valid))

        return avg

    def fit(self, max_epochs: int) -> str:
        """Run full training loop."""
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            logger.info("=== Epoch %d/%d ===", epoch + 1, max_epochs)
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            logger.info(
                "Epoch %d — train_loss: %.5f | val_loss: %.5f | "
                "val_eer: %.4f | val_f1: %.4f",
                epoch,
                train_metrics.get("loss", 0),
                val_metrics.get("loss", 0),
                val_metrics.get("eer", 1.0),
                val_metrics.get("f1", 0),
            )

            for cb in self.callbacks:
                stop = cb.on_epoch_end(self, epoch, train_metrics, val_metrics)
                if stop:
                    logger.info("Early stopping at epoch %d", epoch)
                    for cb2 in self.callbacks:
                        cb2.on_train_end(self)
                    return self.best_path

        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.best_path

    def _save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None:
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
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt.get("metrics", {})


def main(config_path: str) -> None:
    """Run FARA training pipeline."""
    config = load_config(config_path)

    exp_dir = Path(config.output.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(exp_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    logger.info("FARA Training — config: %s", config_path)
    set_seed(config.training.seed)

    # Data
    train_loader, val_loader = _build_dataloaders(config)
    logger.info(
        "Data: %d train batches, %d val batches",
        len(train_loader), len(val_loader),
    )

    # WavLM (frozen backbone)
    logger.info("Loading WavLM-Large from %s", config.wavlm.checkpoint)
    wavlm = WavLMExtractor(config.wavlm.checkpoint, freeze=True)
    wavlm_params = sum(p.numel() for p in wavlm.parameters())
    logger.info("WavLM: %.2fM params (frozen)", wavlm_params / 1e6)

    # FARA model (trainable)
    model = FARA(
        d_model=config.model.d_model,
        n_wavlm_layers=config.model.n_wavlm_layers,
        mask_k=config.model.mask_k,
        sincnet_channels=config.model.sincnet_channels,
        sincnet_kernel=config.model.sincnet_kernel,
        sincnet_stride=config.model.sincnet_stride,
        num_experts=config.model.num_experts,
        num_classes=config.model.num_classes,
    )
    fara_params = sum(p.numel() for p in model.parameters())
    logger.info("FARA: %.2fM trainable params", fara_params / 1e6)

    # Optimizer (only FARA params, not WavLM)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )

    # Loss
    loss_fn = CombinedLoss(
        spoof_weight=config.loss.spoof_weight,
        boundary_weight=config.loss.boundary_weight,
        crl_weight=config.loss.crl_weight,
        beta=config.loss.beta,
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_dir=str(exp_dir / "checkpoints"),
            save_every_n_epochs=5,
            monitor="eer",
            mode="min",
        ),
        EarlyStoppingCallback(
            patience=config.training.patience,
            monitor="eer",
            mode="min",
        ),
        TensorBoardCallback(log_dir=str(exp_dir / "tb_logs")),
    ]

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = FARATrainer(
        wavlm=wavlm,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        callbacks=callbacks,
        device=device,
    )

    best_path = trainer.fit(max_epochs=config.training.max_epochs)

    if best_path:
        logger.info("Best checkpoint: %s", best_path)
        trainer._load_checkpoint(best_path)
        final = trainer.validate()
        logger.info("Final metrics: %s", final)


def _build_dataloaders(config):
    """Create train/val DataLoaders."""
    train_ds = PartialSpoofDataset(
        root=config.data.dataset_root,
        split="train",
        target_sr=config.data.target_sr,
        frame_duration_ms=config.data.frame_duration_ms,
    )
    val_ds = PartialSpoofDataset(
        root=config.data.dataset_root,
        split="dev",
        target_sr=config.data.target_sr,
        frame_duration_ms=config.data.frame_duration_ms,
    )
    return (
        DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
            drop_last=True,
        ),
        DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FARA model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
