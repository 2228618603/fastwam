#!/usr/bin/env python3
"""Parse a live FastWAM training log and publish loss curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("load-giga/code/live"))
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--gif", action="store_true", help="Also write an animated loss GIF.")
    parser.add_argument("--gif-tail", type=int, default=300, help="Maximum recent train points to animate.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def normalize_log_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    # Rich may wrap long metric names in narrow terminals.
    text = re.sub(r"loss_acti\s+on", "loss_action", text)
    text = re.sub(r"loss_vid\s+eo", "loss_video", text)
    text = re.sub(r"infer_ps\s+nr", "infer_psnr", text)
    text = re.sub(r"infer_ss\s+im", "infer_ssim", text)
    text = re.sub(r"action_l\s+1", "action_l1", text)
    text = re.sub(r"action_l\s+2", "action_l2", text)
    return re.sub(r"\s+", " ", text)


def _last_by_step(rows: list[dict]) -> list[dict]:
    by_step = {}
    for row in rows:
        by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def parse_metrics(log_path: Path) -> tuple[list[dict], list[dict]]:
    if not log_path.exists():
        return [], []
    text = normalize_log_text(log_path.read_text(errors="replace"))
    train_rows = []
    train_pattern = re.compile(
        rf"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<max_step>\d+)\s+"
        rf"loss=(?P<loss>{FLOAT})(?P<body>.{{0,800}}?)"
        rf"lr=(?P<lr>{FLOAT})\s+speed=(?P<steps_per_sec>{FLOAT})\s+step/s,\s+"
        rf"(?P<samples_per_sec>{FLOAT})\s+samples/s",
        re.DOTALL,
    )
    for match in train_pattern.finditer(text):
        body = match.group("body")
        row = {
            "epoch": int(match.group("epoch")),
            "step": int(match.group("step")),
            "max_step": int(match.group("max_step")),
            "loss": float(match.group("loss")),
            "lr": float(match.group("lr")),
            "steps_per_sec": float(match.group("steps_per_sec")),
            "samples_per_sec": float(match.group("samples_per_sec")),
        }
        for key in ("loss_action", "loss_video"):
            key_match = re.search(rf"{key}=({FLOAT})", body)
            row[key] = None if key_match is None else float(key_match.group(1))
        train_rows.append(row)

    eval_rows = []
    eval_pattern = re.compile(
        rf"step=(?P<step>\d+)\s+val_loss=(?P<val_loss>{FLOAT})\s+"
        rf"infer_psnr=(?P<infer_psnr>{FLOAT})\s+infer_ssim=(?P<infer_ssim>{FLOAT})"
        rf"(?P<body>.{{0,300}}?)(?=epoch=|step=|$)",
        re.DOTALL,
    )
    for match in eval_pattern.finditer(text):
        body = match.group("body")
        row = {
            "step": int(match.group("step")),
            "val_loss": float(match.group("val_loss")),
            "infer_psnr": float(match.group("infer_psnr")),
            "infer_ssim": float(match.group("infer_ssim")),
        }
        for key in ("action_l1", "action_l2"):
            key_match = re.search(rf"{key}=({FLOAT})", body)
            row[key] = None if key_match is None else float(key_match.group(1))
        eval_rows.append(row)

    return _last_by_step(train_rows), _last_by_step(eval_rows)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(tmp, path)


def write_plot(path: Path, train_rows: list[dict], eval_rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.png")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("FastWAM AgileX BS16 Training Monitor")

    ax = axes[0]
    if train_rows:
        steps = [r["step"] for r in train_rows]
        ax.plot(steps, [r["loss"] for r in train_rows], label="train/loss", linewidth=2)
        if any(r.get("loss_action") is not None for r in train_rows):
            ax.plot(steps, [r.get("loss_action") for r in train_rows], label="loss_action", alpha=0.8)
        if any(r.get("loss_video") is not None for r in train_rows):
            ax.plot(steps, [r.get("loss_video") for r in train_rows], label="loss_video", alpha=0.8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Waiting for training loss...", ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    if train_rows:
        steps = [r["step"] for r in train_rows]
        ax.plot(steps, [r["samples_per_sec"] for r in train_rows], label="samples/sec", color="tab:green")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Waiting for throughput...", ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_ylabel("throughput")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    if eval_rows:
        steps = [r["step"] for r in eval_rows]
        ax.plot(steps, [r["val_loss"] for r in eval_rows], label="eval/val_loss", color="tab:red")
        ax2 = ax.twinx()
        ax2.plot(steps, [r["infer_psnr"] for r in eval_rows], label="infer_psnr", color="tab:purple")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax2.set_ylabel("PSNR")
    else:
        ax.text(0.5, 0.5, "No eval metrics yet", ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_ylabel("eval loss")
    ax.grid(True, alpha=0.25)

    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    os.replace(tmp, path)


def write_gif(path: Path, train_rows: list[dict], gif_tail: int = 300) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.gif")
    rows = train_rows[-max(int(gif_tail), 1):]
    if not rows:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.text(0.5, 0.5, "Waiting for training loss...", ha="center", va="center")
        ax.set_axis_off()
        placeholder = path.with_suffix(".placeholder.png")
        fig.savefig(placeholder, dpi=120)
        plt.close(fig)
        from PIL import Image

        Image.open(placeholder).save(path, save_all=True, append_images=[], duration=1000, loop=0)
        placeholder.unlink(missing_ok=True)
        return

    from matplotlib.animation import FuncAnimation, PillowWriter

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    loss_action = [r.get("loss_action") for r in rows]
    loss_video = [r.get("loss_video") for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title("FastWAM Loss, Updated From Log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(min(steps), max(steps) if max(steps) > min(steps) else min(steps) + 1)
    finite_values = [v for v in losses + loss_action + loss_video if v is not None]
    y_min = min(finite_values)
    y_max = max(finite_values)
    pad = max((y_max - y_min) * 0.08, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)
    line_total, = ax.plot([], [], label="loss", linewidth=2)
    line_action, = ax.plot([], [], label="loss_action", alpha=0.8)
    line_video, = ax.plot([], [], label="loss_video", alpha=0.8)
    marker = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend()

    def update(frame_idx: int):
        end = frame_idx + 1
        xs = steps[:end]
        line_total.set_data(xs, losses[:end])
        line_action.set_data(xs, loss_action[:end])
        line_video.set_data(xs, loss_video[:end])
        latest = rows[frame_idx]
        marker.set_text(
            f"step {latest['step']} / {latest['max_step']}\n"
            f"loss {latest['loss']:.4f}\n"
            f"samples/s {latest['samples_per_sec']:.2f}"
        )
        return line_total, line_action, line_video, marker

    frame_count = min(len(rows), 160)
    if len(rows) > frame_count:
        frame_indices = np.linspace(0, len(rows) - 1, frame_count).round().astype(int).tolist()
    else:
        frame_indices = list(range(len(rows)))

    def update_from_sparse(i: int):
        return update(frame_indices[i])

    anim = FuncAnimation(fig, update_from_sparse, frames=len(frame_indices), interval=220, blit=False)
    anim.save(tmp, writer=PillowWriter(fps=5))
    plt.close(fig)
    os.replace(tmp, path)


def write_html(path: Path, status: dict) -> None:
    latest = status.get("latest_train") or {}
    latest_eval = status.get("latest_eval") or {}
    generated = status.get("generated_at", "")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <title>FastWAM Loss Monitor</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #202124; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #d7dce2; border-radius: 8px; padding: 12px; background: #fff; }}
    .label {{ font-size: 12px; color: #5f6368; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #d7dce2; border-radius: 8px; }}
    code {{ background: #f1f3f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FastWAM Loss Monitor</h1>
  <p>Updated: <code>{generated}</code>. This page refreshes every 15 seconds.</p>
  <div class="grid">
    <div class="card"><div class="label">step</div><div class="value">{latest.get("step", "-")}/{latest.get("max_step", "-")}</div></div>
    <div class="card"><div class="label">loss</div><div class="value">{latest.get("loss", "-")}</div></div>
    <div class="card"><div class="label">loss_action</div><div class="value">{latest.get("loss_action", "-")}</div></div>
    <div class="card"><div class="label">loss_video</div><div class="value">{latest.get("loss_video", "-")}</div></div>
    <div class="card"><div class="label">samples/sec</div><div class="value">{latest.get("samples_per_sec", "-")}</div></div>
    <div class="card"><div class="label">latest val_loss</div><div class="value">{latest_eval.get("val_loss", "-")}</div></div>
  </div>
  <img src="loss_curve.png?ts={int(time.time())}" alt="loss curve">
  <p>Animated file: <code>loss_curve.gif</code>. Raw files: <code>train_metrics.csv</code>, <code>eval_metrics.csv</code>, <code>status.json</code>.</p>
</body>
</html>
"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(html)
    os.replace(tmp, path)


def publish(log_path: Path, out_dir: Path, write_animation: bool = False, gif_tail: int = 300) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = parse_metrics(log_path)
    write_csv(
        out_dir / "train_metrics.csv",
        train_rows,
        ["epoch", "step", "max_step", "loss", "loss_action", "loss_video", "lr", "steps_per_sec", "samples_per_sec"],
    )
    write_csv(
        out_dir / "eval_metrics.csv",
        eval_rows,
        ["step", "val_loss", "infer_psnr", "infer_ssim", "action_l1", "action_l2"],
    )
    write_plot(out_dir / "loss_curve.png", train_rows, eval_rows)
    if write_animation:
        write_gif(out_dir / "loss_curve.gif", train_rows, gif_tail=gif_tail)
    status = {
        "log": str(log_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_points": len(train_rows),
        "eval_points": len(eval_rows),
        "latest_train": train_rows[-1] if train_rows else None,
        "latest_eval": eval_rows[-1] if eval_rows else None,
    }
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "status.json")
    write_html(out_dir / "index.html", status)
    return status


def main() -> None:
    args = parse_args()
    while True:
        status = publish(args.log, args.out_dir, write_animation=args.gif, gif_tail=args.gif_tail)
        print(
            f"[{status['generated_at']}] train_points={status['train_points']} "
            f"eval_points={status['eval_points']} latest={status['latest_train']}"
        )
        if args.once:
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
