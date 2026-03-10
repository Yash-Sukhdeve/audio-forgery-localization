"""BAM baseline wrapper.

Invokes BAM's train.py as a subprocess. Never modifies BAM repo code.

Reference: J. Zhong, B. Li, J. Yi, "Enhancing partially spoofed audio
localization with boundary-aware attention mechanism," Interspeech, 2024.
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from baselines.wrappers.base_wrapper import BaseWrapper


class BAMWrapper(BaseWrapper):
    """Wrapper for BAM baseline.

    Args:
        repo_dir: Path to cloned BAM repo.
        experiment_name: Name for this experiment run.
        data_dir: Path to prepared data directory (output of bam_data_prep.py).
    """

    def __init__(
        self,
        repo_dir: str,
        experiment_name: str = "bam_wavlm",
        data_dir: Optional[str] = None,
    ):
        super().__init__(repo_dir, experiment_name)
        self.data_dir = Path(data_dir) if data_dir else self.repo_dir / "data"

    def train(self, **kwargs) -> Path:
        """Train BAM on PartialSpoof."""
        defaults = {
            "max_epochs": 50,
            "batch_size": 8,
            "base_lr": 1e-5,
            "weight_decay": 1e-4,
            "samplerate": 16000,
            "resolution": 0.02,
            "gpu": "[0]",
        }
        defaults.update(kwargs)

        cmd = [
            sys.executable, str(self.repo_dir / "train.py"),
            "--exp_name", self.experiment_name,
            "--train_root", str(self.data_dir / "raw" / "train"),
            "--dev_root", str(self.data_dir / "raw" / "dev"),
            "--eval_root", str(self.data_dir / "raw" / "eval"),
            "--label_root", str(self.data_dir),
        ]

        for key, val in defaults.items():
            cmd.extend([f"--{key}", str(val)])

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.repo_dir))

        if result.returncode != 0:
            raise RuntimeError(f"BAM training failed with code {result.returncode}")

        # Find best checkpoint
        exp_dir = self.repo_dir / "exp" / self.experiment_name / "train"
        ckpts = sorted(exp_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {exp_dir}")

        return ckpts[-1]

    def evaluate(self, checkpoint: str, split: str = "eval", **kwargs) -> Dict[str, float]:
        """Evaluate BAM using its built-in evaluation."""
        cmd = [
            sys.executable, str(self.repo_dir / "train.py"),
            "--test_only",
            "--exp_name", self.experiment_name,
            "--eval_root", str(self.data_dir / "raw" / split),
            "--label_root", str(self.data_dir),
            "--checkpoint", str(checkpoint),
            "--resolution", str(kwargs.get("resolution", 0.02)),
            "--gpu", str(kwargs.get("gpu", "[0]")),
        ]

        result = subprocess.run(
            cmd, cwd=str(self.repo_dir), capture_output=True, text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"BAM eval failed: {result.stderr}")

        return self._parse_eval_output(result.stdout)

    def _parse_eval_output(self, stdout: str) -> Dict[str, float]:
        """Parse BAM's evaluation output for metrics."""
        metrics = {}
        for line in stdout.split("\n"):
            line_lower = line.lower()
            if "eer" in line_lower:
                try:
                    val = float(line.split(":")[-1].strip().replace("%", ""))
                    metrics["eer"] = val / 100.0 if val > 1 else val
                except (ValueError, IndexError):
                    pass
            if "f1" in line_lower:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics["f1"] = val
                except (ValueError, IndexError):
                    pass
        return metrics
