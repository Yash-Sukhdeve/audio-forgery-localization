#!/usr/bin/env python3
"""Parse FARA training logs and BAM eval results into dashboard_data.json.

Reads:
  - results/fara/fara_training.log
  - results/bam/eval_results_002.txt
Writes:
  - results/dashboard_data.json

Usage:
    python scripts/update_dashboard.py          # one-shot
    watch -n 30 python scripts/update_dashboard.py  # periodic via cron/watch
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FARA_LOG = PROJECT_ROOT / "results" / "fara" / "fara_training.log"
BAM_EVAL = PROJECT_ROOT / "results" / "bam" / "eval_results_002.txt"
OUTPUT = PROJECT_ROOT / "results" / "dashboard_data.json"

MAX_EPOCHS = 50
TOTAL_STEPS_PER_EPOCH = 3172


def parse_fara_log(log_path: Path) -> dict:
    """Parse the FARA training log file and extract all metrics."""
    result = {
        "epochs_completed": [],
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": TOTAL_STEPS_PER_EPOCH,
        "max_epochs": MAX_EPOCHS,
        "train_losses": [],
        "val_losses": [],
        "val_eers": [],
        "val_f1s": [],
        "step_losses": [],
        "best_eer": None,
        "best_eer_epoch": None,
        "start_time": None,
        "last_time": None,
        "training_active": False,
        "wavlm_params": None,
        "fara_params": None,
    }

    if not log_path.exists():
        return result

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().split("\n")

    # Regex patterns
    re_epoch_header = re.compile(r"=== Epoch (\d+)/(\d+) ===")
    re_step = re.compile(r"Step (\d+)/(\d+) .* loss: ([\d.]+)")
    re_epoch_summary = re.compile(
        r"Epoch (\d+) .* train_loss: ([\d.]+) \| val_loss: ([\d.]+) "
        r"\| val_eer: ([\d.]+) \| val_f1: ([\d.]+)"
    )
    re_best_eer = re.compile(r"New best eer: ([\d.]+) \(epoch (\d+)\)")
    re_timestamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    re_params = re.compile(r"(WavLM|FARA): ([\d.]+)M")

    current_epoch_display = 0
    latest_step = 0
    latest_step_loss = 0.0
    epoch_step_losses = []

    for line in lines:
        # Extract timestamps
        ts_match = re_timestamp.search(line)
        if ts_match:
            ts_str = ts_match.group(1)
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if result["start_time"] is None:
                    result["start_time"] = ts.isoformat()
                result["last_time"] = ts.isoformat()
            except ValueError:
                pass

        # Param counts
        pm = re_params.search(line)
        if pm:
            name, count = pm.group(1), pm.group(2)
            if name == "WavLM":
                result["wavlm_params"] = f"{count}M"
            elif name == "FARA":
                result["fara_params"] = f"{count}M"

        # Epoch header
        eh = re_epoch_header.search(line)
        if eh:
            current_epoch_display = int(eh.group(1))
            epoch_step_losses = []

        # Step progress
        sm = re_step.search(line)
        if sm:
            step_num = int(sm.group(1))
            total = int(sm.group(2))
            loss = float(sm.group(3))
            latest_step = step_num
            latest_step_loss = loss
            epoch_step_losses.append({"step": step_num, "loss": loss})

        # Epoch summary (validation)
        es = re_epoch_summary.search(line)
        if es:
            epoch_idx = int(es.group(1))
            train_loss = float(es.group(2))
            val_loss = float(es.group(3))
            val_eer = float(es.group(4))
            val_f1 = float(es.group(5))
            result["epochs_completed"].append(epoch_idx)
            result["train_losses"].append(
                {"epoch": epoch_idx, "value": train_loss}
            )
            result["val_losses"].append(
                {"epoch": epoch_idx, "value": val_loss}
            )
            result["val_eers"].append(
                {"epoch": epoch_idx, "value": val_eer}
            )
            result["val_f1s"].append(
                {"epoch": epoch_idx, "value": val_f1}
            )
            epoch_step_losses = []

        # Best EER
        be = re_best_eer.search(line)
        if be:
            result["best_eer"] = float(be.group(1))
            result["best_eer_epoch"] = int(be.group(2))

    # Current state
    n_completed = len(result["epochs_completed"])
    result["current_epoch"] = current_epoch_display
    result["current_step"] = latest_step
    result["current_step_loss"] = latest_step_loss
    result["step_losses"] = epoch_step_losses

    # Determine if training is still active (last log < 5 min ago)
    if result["last_time"]:
        last = datetime.fromisoformat(result["last_time"])
        result["training_active"] = (datetime.now() - last) < timedelta(minutes=5)

    # Estimate time remaining
    if result["start_time"] and result["last_time"] and n_completed > 0:
        start = datetime.fromisoformat(result["start_time"])
        last = datetime.fromisoformat(result["last_time"])
        elapsed = (last - start).total_seconds()
        # Fraction of work done: completed epochs + partial current
        frac_current = latest_step / TOTAL_STEPS_PER_EPOCH if TOTAL_STEPS_PER_EPOCH > 0 else 0
        total_frac = (n_completed + frac_current) / MAX_EPOCHS
        if total_frac > 0:
            est_total = elapsed / total_frac
            remaining_s = est_total - elapsed
            result["estimated_remaining_s"] = max(0, remaining_s)
            result["estimated_remaining_human"] = format_duration(remaining_s)
        else:
            result["estimated_remaining_s"] = None
            result["estimated_remaining_human"] = "N/A"
    else:
        result["estimated_remaining_s"] = None
        result["estimated_remaining_human"] = "N/A"

    return result


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds <= 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def parse_bam_eval(eval_path: Path) -> dict:
    """Parse BAM 0.02s evaluation results."""
    result = {
        "eer": None,
        "accuracy": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "boundary_eer": None,
        "boundary_f1": None,
        "best_checkpoint": None,
    }

    if not eval_path.exists():
        return result

    text = eval_path.read_text(encoding="utf-8", errors="replace")

    patterns = {
        "eer": r"EER:\s+([\d.]+)%",
        "accuracy": r"Accuracy:\s+([\d.]+)%",
        "f1": r"(?<!Boundary\s)F1:\s+([\d.]+)%",
        "precision": r"(?<!Boundary\s)Precision:\s+([\d.]+)%",
        "recall": r"(?<!Boundary\s)Recall:\s+([\d.]+)%",
        "boundary_eer": r"Boundary EER:\s+([\d.]+)%",
        "boundary_f1": r"Boundary F1:\s+([\d.]+)%",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    ckpt_m = re.search(r"Best Checkpoint:\s+(.+)", text)
    if ckpt_m:
        result["best_checkpoint"] = ckpt_m.group(1).strip()

    return result


def count_files(directory: Path, exclude_pycache: bool = True) -> int:
    """Count non-pycache Python/config files in a directory."""
    if not directory.exists():
        return 0
    count = 0
    for f in directory.iterdir():
        if exclude_pycache and f.name == "__pycache__":
            continue
        if f.is_file():
            count += 1
    return count


def build_file_inventory() -> list:
    """Build the file inventory for the dashboard."""
    dirs = [
        ("fara/model", "FARA model components"),
        ("fara/losses", "Loss functions"),
        ("core", "Core training infrastructure"),
        ("tests/fara", "FARA test suite"),
        ("scripts", "Utility scripts"),
        ("configs", "Configuration files"),
    ]
    inventory = []
    for rel_path, desc in dirs:
        full = PROJECT_ROOT / rel_path
        n = count_files(full)
        inventory.append({
            "path": rel_path,
            "description": desc,
            "file_count": n,
        })
    return inventory


def main() -> None:
    print(f"[update_dashboard] Parsing FARA log: {FARA_LOG}")
    fara_data = parse_fara_log(FARA_LOG)

    print(f"[update_dashboard] Parsing BAM eval: {BAM_EVAL}")
    bam_data = parse_bam_eval(BAM_EVAL)

    file_inventory = build_file_inventory()

    dashboard = {
        "updated_at": datetime.now().isoformat(),
        "fara_training": fara_data,
        "bam_002_eval": bam_data,
        "bam_016": {
            "eer": 8.33,
            "f1": 92.23,
            "published_eer": 8.43,
            "status": "Complete",
        },
        "file_inventory": file_inventory,
        "science_audit": {
            "verified_equations": 10,
            "critical_issues_fixed": 1,
            "issue_detail": "Fabricated citation removed",
            "engineering_assumptions": 19,
            "author_contact": "Hongxia Wang (hxwang@scu.edu.cn)",
        },
        "phases": [
            {"id": 0, "name": "Infrastructure", "status": "complete"},
            {"id": 1, "name": "BAM Baseline (0.16s)", "status": "complete",
             "detail": "EER: 8.33%"},
            {"id": 2, "name": "FARA Components", "status": "complete",
             "detail": "96/96 tests pass"},
            {"id": "bam_retrain", "name": "BAM Retrain (0.02s)",
             "status": "complete",
             "detail": "Best EER: 0.83% (val), Eval done"},
            {"id": 3, "name": "FARA Training", "status": "in_progress",
             "detail": "Live metrics"},
            {"id": 4, "name": "Cross-dataset Eval", "status": "pending"},
            {"id": 5, "name": "CFPRF + PSDS", "status": "pending"},
            {"id": 6, "name": "Analysis", "status": "pending"},
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"[update_dashboard] Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
