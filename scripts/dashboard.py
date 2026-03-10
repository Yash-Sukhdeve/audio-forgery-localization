#!/usr/bin/env python3
"""Live web-based training dashboard — reusable across all models.

Reads TensorBoard event files to display real-time training metrics
in a web browser. Saves all visualizations as static HTML for archival.

Usage:
    # Monitor BAM training
    python scripts/dashboard.py \
        --exp_dir baselines/repos/BAM/exp/bam_wavlm_ps/train \
        --name BAM --max_epochs 50

    # Monitor FARA training (future)
    python scripts/dashboard.py --exp_dir fara/exp/train --name FARA

    # Save plots only (no server)
    python scripts/dashboard.py --exp_dir <path> --name BAM --save_only

    # Custom port
    python scripts/dashboard.py --exp_dir <path> --port 8051
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# TensorBoard reader
# ---------------------------------------------------------------------------

def find_latest_version(exp_dir: str) -> Optional[str]:
    """Find the latest lightning_logs/version_N directory."""
    logs_dir = Path(exp_dir) / "lightning_logs"
    if not logs_dir.exists():
        if list(Path(exp_dir).glob("events.out.tfevents.*")):
            return exp_dir
        return None
    versions = sorted(
        logs_dir.glob("version_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return str(versions[-1]) if versions else None


def read_tensorboard(version_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Read all scalar events from a TensorBoard log directory.

    Returns: {tag: [{"step": int, "wall_time": float, "value": float}, ...]}
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(version_dir, size_guidance={"scalars": 0})
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [
            {"step": e.step, "wall_time": e.wall_time, "value": e.value}
            for e in events
        ]
    return data


# ---------------------------------------------------------------------------
# Log file parser (supplements TensorBoard with validation metrics)
# ---------------------------------------------------------------------------

LOG_PATTERN = re.compile(
    r"Epoch \[(\d+)\]:\s*(binary\s+)?(train|validate|test)\s+"
    r"(eer|acc|precision|recall|F1)\s+([\d.eE\-+]+)"
)

PROGRESS_PATTERN = re.compile(r"Epoch (\d+):\s+(\d+)%")


def parse_log_file(log_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse structured training log file."""
    data = defaultdict(list)
    if not os.path.exists(log_path):
        return data
    with open(log_path) as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                epoch = int(m.group(1))
                prefix = "b_" if m.group(2) else ""
                phase = m.group(3)
                metric = m.group(4)
                value = float(m.group(5))
                key = f"{phase}/{prefix}{phase}_{metric}"
                data[key].append({"step": epoch, "wall_time": 0, "value": value})
    return data


def parse_log_progress(log_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Get current epoch and progress % from the end of the log file."""
    if not os.path.exists(log_path):
        return None, None
    epoch, pct = None, None
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 16384))
        tail = f.read().decode("utf-8", errors="replace")
    for m in PROGRESS_PATTERN.finditer(tail):
        epoch = int(m.group(1))
        pct = int(m.group(2))
    return epoch, pct


def find_log_file(exp_dir: str) -> Optional[str]:
    """Auto-detect the training log file."""
    version_dir = find_latest_version(exp_dir)
    if version_dir:
        log = Path(version_dir) / "TrainerFn.FITTING.log"
        if log.exists():
            return str(log)
    return None


def find_checkpoint_dir(exp_dir: str) -> Optional[str]:
    version_dir = find_latest_version(exp_dir)
    if version_dir:
        ckpt_dir = Path(version_dir) / "checkpoints"
        if ckpt_dir.exists():
            return str(ckpt_dir)
    return None


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#2196F3",
    "validate": "#FF9800",
    "test": "#4CAF50",
    "boundary": "#9C27B0",
    "loss": "#F44336",
    "spoof": "#2196F3",
    "eer": "#E91E63",
    "f1": "#4CAF50",
    "acc": "#FF9800",
}


def _color_for(tag: str) -> str:
    lower = tag.lower()
    for key, color in COLORS.items():
        if key in lower:
            return color
    return "#607D8B"


def build_loss_figure(tb_data: Dict, model_name: str) -> go.Figure:
    """Build loss curves plot."""
    loss_tags = sorted(t for t in tb_data if "loss" in t.lower() and "hp_" not in t.lower())
    if not loss_tags:
        return go.Figure().update_layout(title="No loss data yet")

    fig = go.Figure()
    for tag in loss_tags:
        events = tb_data[tag]
        steps = [e["step"] for e in events]
        values = [e["value"] for e in events]
        is_epoch = "epoch" in tag.lower()
        fig.add_trace(go.Scatter(
            x=steps, y=values,
            mode="lines+markers" if is_epoch else "lines",
            name=tag.replace("_", " "),
            line=dict(color=_color_for(tag), width=2 if is_epoch else 1),
            opacity=1.0 if is_epoch else 0.6,
        ))

    fig.update_layout(
        title=f"{model_name} — Loss Curves",
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def build_epoch_metrics_figure(
    tb_data: Dict,
    log_data: Dict,
    model_name: str,
    phase: str,
    title_suffix: str,
) -> go.Figure:
    """Build per-epoch metric plots (EER, F1, Acc, etc.)."""
    # Collect relevant tags
    all_data = {}

    # TensorBoard epoch-level metrics
    for tag, events in tb_data.items():
        if phase in tag.lower() and "loss" not in tag.lower() and "hp_" not in tag.lower():
            all_data[tag] = events

    # Log file metrics (supplement)
    for tag, events in log_data.items():
        if phase in tag and tag not in all_data:
            all_data[tag] = events

    if not all_data:
        return go.Figure().update_layout(title=f"No {phase} metrics yet")

    # Separate into spoof-detection and boundary-detection
    spoof_tags = sorted(t for t in all_data if "b_" not in t.lower() and "binary" not in t.lower())
    boundary_tags = sorted(t for t in all_data if "b_" in t.lower() or "binary" in t.lower())

    has_boundary = bool(boundary_tags)
    fig = make_subplots(
        rows=2 if has_boundary else 1,
        cols=1,
        subplot_titles=(
            [f"Spoof Detection ({phase.title()})", f"Boundary Detection ({phase.title()})"]
            if has_boundary
            else [f"Spoof Detection ({phase.title()})"]
        ),
        vertical_spacing=0.15,
    )

    for tag in spoof_tags:
        events = all_data[tag]
        epochs = [e["step"] for e in events]
        values = [e["value"] for e in events]
        short_name = tag.split("/")[-1].replace(f"{phase}_", "").replace("_", " ").upper()
        is_eer = "eer" in tag.lower()
        fig.add_trace(
            go.Scatter(
                x=epochs, y=[v * 100 for v in values],
                mode="lines+markers",
                name=short_name,
                line=dict(color=_color_for(tag), width=2),
                marker=dict(size=6),
                hovertemplate=f"{short_name}: %{{y:.2f}}%<br>Epoch: %{{x}}<extra></extra>",
            ),
            row=1, col=1,
        )

    for tag in boundary_tags:
        events = all_data[tag]
        epochs = [e["step"] for e in events]
        values = [e["value"] for e in events]
        short_name = tag.split("/")[-1].replace(f"b_{phase}_", "").replace("_", " ").upper()
        fig.add_trace(
            go.Scatter(
                x=epochs, y=[v * 100 for v in values],
                mode="lines+markers",
                name=f"B-{short_name}",
                line=dict(color=_color_for(tag), width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            ),
            row=2 if has_boundary else 1, col=1,
        )

    fig.update_layout(
        title=f"{model_name} — {title_suffix}",
        template="plotly_dark",
        height=500 if has_boundary else 350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_yaxes(title_text="Metric (%)", row=1, col=1)
    if has_boundary:
        fig.update_yaxes(title_text="Metric (%)", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2 if has_boundary else 1, col=1)

    return fig


def build_eer_comparison_figure(
    tb_data: Dict, log_data: Dict, model_name: str
) -> go.Figure:
    """Build EER trend for both train and validation."""
    fig = go.Figure()

    for phase, color, dash in [("train", "#2196F3", None), ("validate", "#FF9800", "dash")]:
        # Try TB first
        eer_tag = None
        for tag in tb_data:
            if phase in tag.lower() and "eer" in tag.lower() and "b_" not in tag.lower() and "binary" not in tag.lower():
                eer_tag = tag
                break

        events = tb_data.get(eer_tag, []) if eer_tag else []

        # Fall back to log
        if not events:
            for tag in log_data:
                if phase in tag and "eer" in tag and "b_" not in tag:
                    events = log_data[tag]
                    break

        if events:
            epochs = [e["step"] for e in events]
            values = [e["value"] * 100 for e in events]
            fig.add_trace(go.Scatter(
                x=epochs, y=values,
                mode="lines+markers",
                name=f"{phase.title()} EER",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=8),
            ))

    # Add target line
    fig.add_hline(
        y=8.43, line_dash="dot", line_color="red", opacity=0.5,
        annotation_text="BAM Target: 8.43%",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{model_name} — EER Trend (Target: 8.43%)",
        xaxis_title="Epoch",
        yaxis_title="EER (%)",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML page builder
# ---------------------------------------------------------------------------

def build_html_page(
    figures: Dict[str, go.Figure],
    model_name: str,
    status: Dict[str, Any],
    auto_refresh: int = 10,
) -> str:
    """Build a full HTML dashboard page."""
    fig_divs = ""
    for fig_id, fig in figures.items():
        fig_divs += f'<div id="{fig_id}" class="plot-container">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>\n'

    elapsed_str = format_duration(status.get("elapsed", 0))
    eta_str = format_duration(status.get("eta", 0))
    epoch = status.get("epoch", 0)
    max_epochs = status.get("max_epochs", 50)
    progress = status.get("progress", 0)
    overall_pct = (epoch + progress / 100) / max_epochs * 100 if max_epochs > 0 else 0

    checkpoints_html = ""
    for ckpt in status.get("checkpoints", []):
        checkpoints_html += f'<tr><td>{ckpt["name"]}</td><td>{ckpt["size"]}</td><td>{ckpt["modified"]}</td></tr>\n'
    if not checkpoints_html:
        checkpoints_html = '<tr><td colspan="3" class="dim">No checkpoints yet</td></tr>'

    # Build best metrics summary
    best_metrics = status.get("best_metrics", {})
    best_html = ""
    for name, val in best_metrics.items():
        best_html += f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value">{val}</div></div>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="{auto_refresh}">
<title>{model_name} Training Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0d1117;
        color: #c9d1d9;
        padding: 20px;
    }}
    .header {{
        background: linear-gradient(135deg, #161b22, #1c2333);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }}
    .header h1 {{
        font-size: 24px;
        color: #58a6ff;
        margin-bottom: 12px;
    }}
    .header .subtitle {{
        color: #8b949e;
        font-size: 14px;
    }}
    .progress-container {{
        margin: 16px 0;
    }}
    .progress-bar {{
        height: 24px;
        background: #21262d;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
    }}
    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        border-radius: 12px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: white;
        min-width: 50px;
    }}
    .stats-row {{
        display: flex;
        gap: 20px;
        margin: 12px 0;
        flex-wrap: wrap;
    }}
    .stat {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 20px;
        min-width: 140px;
    }}
    .stat-label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
    .stat-value {{ font-size: 22px; font-weight: bold; color: #58a6ff; margin-top: 4px; }}
    .metric-cards {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 16px 0;
    }}
    .metric-card {{
        background: #161b22;
        border: 1px solid #238636;
        border-radius: 8px;
        padding: 12px 18px;
        min-width: 120px;
        text-align: center;
    }}
    .metric-label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; }}
    .metric-value {{ font-size: 20px; font-weight: bold; color: #3fb950; margin-top: 4px; }}
    .plot-container {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 20px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
    @media (max-width: 1200px) {{
        .grid {{ grid-template-columns: 1fr; }}
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }}
    th, td {{
        padding: 8px 14px;
        text-align: left;
        border-bottom: 1px solid #21262d;
    }}
    th {{
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .dim {{ color: #484f58; }}
    .ckpt-section {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    .ckpt-section h3 {{ color: #d2a8ff; margin-bottom: 10px; }}
    .footer {{
        text-align: center;
        color: #484f58;
        font-size: 12px;
        margin-top: 20px;
        padding: 10px;
    }}
    .timestamp {{ color: #8b949e; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
    <h1>{model_name} Training Dashboard</h1>
    <div class="subtitle">Auto-refreshes every {auto_refresh}s &mdash; Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class="progress-container">
        <div class="progress-bar">
            <div class="progress-fill" style="width: {max(overall_pct, 2):.1f}%">{overall_pct:.1f}%</div>
        </div>
    </div>

    <div class="stats-row">
        <div class="stat"><div class="stat-label">Epoch</div><div class="stat-value">{epoch} / {max_epochs}</div></div>
        <div class="stat"><div class="stat-label">Elapsed</div><div class="stat-value">{elapsed_str}</div></div>
        <div class="stat"><div class="stat-label">ETA</div><div class="stat-value">{eta_str}</div></div>
        <div class="stat"><div class="stat-label">Speed</div><div class="stat-value">{status.get('speed', 'N/A')}</div></div>
    </div>

    <div class="metric-cards">
        {best_html}
    </div>
</div>

<div class="plot-container">{figures.get('eer_trend', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}</div>

<div class="grid">
    <div class="plot-container">{figures.get('loss', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="plot-container">{figures.get('train_metrics', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}</div>
</div>

<div class="plot-container">{figures.get('val_metrics', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}</div>

<div class="ckpt-section">
    <h3>Checkpoints</h3>
    <table>
        <tr><th>File</th><th>Size</th><th>Modified</th></tr>
        {checkpoints_html}
    </table>
</div>

<div class="footer">
    Training Dashboard &mdash; Localization Project &mdash; Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Save static report
# ---------------------------------------------------------------------------

def save_report(html: str, output_dir: str, model_name: str) -> str:
    """Save dashboard as static HTML file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.lower()}_dashboard_{timestamp}.html"
    filepath = Path(output_dir) / filename

    # Also save/update a "latest" symlink
    latest = Path(output_dir) / f"{model_name.lower()}_dashboard_latest.html"

    with open(filepath, "w") as f:
        f.write(html)

    # Update latest
    with open(latest, "w") as f:
        f.write(html)

    return str(filepath)


def save_individual_plots(figures: Dict[str, go.Figure], output_dir: str, model_name: str):
    """Save each plot as a standalone interactive HTML."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for fig_id, fig in figures.items():
        filepath = plots_dir / f"{model_name.lower()}_{fig_id}.html"
        fig.write_html(str(filepath), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def get_status(
    tb_data: Dict,
    log_path: Optional[str],
    max_epochs: int,
    exp_dir: str,
) -> Dict[str, Any]:
    """Compute training status from available data."""
    status = {"max_epochs": max_epochs, "epoch": 0, "progress": 0}

    # Epoch from TB
    if "epoch" in tb_data and tb_data["epoch"]:
        status["epoch"] = int(tb_data["epoch"][-1]["value"])

    # Progress from log
    if log_path:
        epoch, pct = parse_log_progress(log_path)
        if epoch is not None:
            status["epoch"] = max(status["epoch"], epoch)
        if pct is not None:
            status["progress"] = pct

    # Timing
    start_time = None
    if "epoch" in tb_data and tb_data["epoch"]:
        start_time = tb_data["epoch"][0]["wall_time"]
    elapsed = time.time() - start_time if start_time else 0
    status["elapsed"] = elapsed

    epoch_count = max(status["epoch"] + 1, 1)
    epoch_rate = elapsed / epoch_count
    status["eta"] = epoch_rate * max(0, max_epochs - status["epoch"] - 1)
    status["speed"] = f"{epoch_rate / 60:.1f} min/epoch" if epoch_rate > 0 else "N/A"

    # Checkpoints
    ckpt_dir = find_checkpoint_dir(exp_dir)
    ckpts = []
    if ckpt_dir and os.path.exists(ckpt_dir):
        for p in sorted(Path(ckpt_dir).glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            ckpts.append({
                "name": p.name,
                "size": f"{p.stat().st_size / 1024 / 1024:.1f} MB",
                "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    status["checkpoints"] = ckpts

    # Best metrics
    best = {}
    for tag, events in tb_data.items():
        if not events or "hp_" in tag or "epoch" == tag:
            continue
        values = [e["value"] for e in events]
        lower = tag.lower()
        if "eer" in lower and "b_" not in lower and "binary" not in lower:
            phase = "Train" if "train" in lower else "Val" if "validate" in lower else "Test"
            best[f"Best {phase} EER"] = f"{min(values) * 100:.2f}%"
        elif "f1" in lower and "b_" not in lower and "binary" not in lower:
            phase = "Train" if "train" in lower else "Val" if "validate" in lower else "Test"
            best[f"Best {phase} F1"] = f"{max(values) * 100:.2f}%"
        elif "loss_epoch" in lower:
            best["Best Loss"] = f"{min(values):.6f}"

    status["best_metrics"] = best
    return status


# ---------------------------------------------------------------------------
# Flask server
# ---------------------------------------------------------------------------

def run_server(args):
    from flask import Flask, Response

    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.route("/")
    def index():
        version_dir = find_latest_version(args.exp_dir)
        if not version_dir:
            return "<h1>No training data found yet. Waiting...</h1>", 200

        tb_data = read_tensorboard(version_dir)
        log_path = args.log_file or find_log_file(args.exp_dir)
        log_data = parse_log_file(log_path) if log_path else {}
        status = get_status(tb_data, log_path, args.max_epochs, args.exp_dir)

        figures = {
            "eer_trend": build_eer_comparison_figure(tb_data, log_data, args.name),
            "loss": build_loss_figure(tb_data, args.name),
            "train_metrics": build_epoch_metrics_figure(tb_data, log_data, args.name, "train", "Train Metrics"),
            "val_metrics": build_epoch_metrics_figure(tb_data, log_data, args.name, "validate", "Validation Metrics"),
        }

        html = build_html_page(figures, args.name, status, auto_refresh=args.refresh)

        # Save report on every refresh
        save_report(html, args.output_dir, args.name)
        save_individual_plots(figures, args.output_dir, args.name)

        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status():
        version_dir = find_latest_version(args.exp_dir)
        if not version_dir:
            return json.dumps({"status": "waiting"})
        tb_data = read_tensorboard(version_dir)
        log_path = args.log_file or find_log_file(args.exp_dir)
        status = get_status(tb_data, log_path, args.max_epochs, args.exp_dir)
        return json.dumps(status, default=str)

    print(f"\n{'='*60}")
    print(f"  {args.name} Training Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Output: {args.output_dir}")
    print(f"  Refresh: {args.refresh}s")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live web training dashboard — reusable for all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment dir with lightning_logs/version_N/")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Training log file (auto-detected if not set)")
    parser.add_argument("--name", type=str, default="Model",
                        help="Model name for display")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Total epochs for progress calculation")
    parser.add_argument("--refresh", type=int, default=10,
                        help="Auto-refresh interval in seconds")
    parser.add_argument("--port", type=int, default=8050,
                        help="Web server port (default: 8050)")
    parser.add_argument("--output_dir", type=str, default="results/dashboards",
                        help="Directory to save HTML reports and plots")
    parser.add_argument("--save_only", action="store_true",
                        help="Save report once and exit (no server)")

    global args
    args = parser.parse_args()

    version_dir = find_latest_version(args.exp_dir)
    if not version_dir:
        print(f"Warning: No TensorBoard logs found in {args.exp_dir}")
        if args.save_only:
            sys.exit(1)
        print("Starting server anyway — will show data when available.\n")

    if args.save_only:
        tb_data = read_tensorboard(version_dir)
        log_path = args.log_file or find_log_file(args.exp_dir)
        log_data = parse_log_file(log_path) if log_path else {}
        status = get_status(tb_data, log_path, args.max_epochs, args.exp_dir)

        figures = {
            "eer_trend": build_eer_comparison_figure(tb_data, log_data, args.name),
            "loss": build_loss_figure(tb_data, args.name),
            "train_metrics": build_epoch_metrics_figure(tb_data, log_data, args.name, "train", "Train Metrics"),
            "val_metrics": build_epoch_metrics_figure(tb_data, log_data, args.name, "validate", "Validation Metrics"),
        }

        html = build_html_page(figures, args.name, status, auto_refresh=9999)
        path = save_report(html, args.output_dir, args.name)
        save_individual_plots(figures, args.output_dir, args.name)
        print(f"Report saved: {path}")
        print(f"Plots saved: {args.output_dir}/plots/")
    else:
        run_server(args)


if __name__ == "__main__":
    main()
