#!/usr/bin/env python3
"""Load a FastWAM checkpoint, run one validation inference, and write a valid mp4."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_batched_eval_sample(sample: dict[str, Any]) -> dict[str, Any]:
    video = sample["video"]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    prompt = sample["prompt"]
    if isinstance(prompt, str):
        prompt = [prompt]

    action = sample.get("action")
    if isinstance(action, torch.Tensor) and action.ndim == 2:
        action = action.unsqueeze(0)

    proprio = sample.get("proprio")
    if isinstance(proprio, torch.Tensor) and proprio.ndim == 2:
        proprio = proprio.unsqueeze(0)

    context = sample.get("context")
    context_mask = sample.get("context_mask")
    if isinstance(context, torch.Tensor) and context.ndim == 2:
        context = context.unsqueeze(0)
    if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)

    action_horizon = None
    if isinstance(action, torch.Tensor):
        action_horizon = int(action.shape[1])

    return {
        "video": video,
        "prompt": prompt,
        "action": action,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "action_horizon": action_horizon,
    }


def write_mp4_cv2(frames: list[Image.Image], output_path: Path, fps: int = 8) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path = output_path
    tmp_dir = Path(tempfile.mkdtemp(prefix="fastwam_verify_mp4_"))
    output_path = tmp_dir / output_path.name
    first = np.asarray(frames[0].convert("RGB"))
    height, width = first.shape[:2]
    if height % 2:
        height += 1
    if width % 2:
        width += 1
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open {output_path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            if rgb.shape[0] != height or rgb.shape[1] != width:
                padded = np.zeros((height, width, 3), dtype=np.uint8)
                padded[: rgb.shape[0], : rgb.shape[1]] = rgb
                rgb = padded
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    probe_mp4(output_path)
    shutil.copyfile(output_path, final_output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def probe_mp4(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open written mp4: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"OpenCV cannot read first frame from mp4: {path}")
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "file_size": path.stat().st_size,
        "first_frame_std": float(frame.std()),
    }


def main() -> None:
    args = parse_args()
    register_default_resolvers()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/data/chw/fastwam/checkpoints")

    cfg = OmegaConf.load(args.config)
    cfg.model.load_text_encoder = False
    cfg.model.action_dit_pretrained_path = "/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(str(cfg.mixed_precision)))
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    checkpoint_payload = model.load_checkpoint(str(args.checkpoint))
    model.eval()

    weight_checks = {}
    model_state = model.mot.state_dict()
    for key in [
        "mixtures.video.blocks.0.self_attn.q.weight",
        "mixtures.video.blocks.29.ffn.2.weight",
        "mixtures.action.blocks.0.self_attn.q.weight",
    ]:
        current = model_state[key].detach().cpu()
        expected = checkpoint_payload["mot"][key].detach().cpu()
        weight_checks[key] = float((current - expected).abs().max().item())

    val_ds = instantiate(cfg.data.val)
    sample = to_batched_eval_sample(val_ds[int(args.sample_index)])
    video0 = sample["video"][0]
    action = sample["action"][0] if isinstance(sample["action"], torch.Tensor) else None
    proprio = sample["proprio"][0, 0] if isinstance(sample["proprio"], torch.Tensor) else None
    input_image = video0[:, 0].unsqueeze(0)
    _, num_frames, _, _ = video0.shape

    infer_kwargs = {
        "input_image": input_image,
        "num_frames": int(num_frames),
        "action": action,
        "action_horizon": sample["action_horizon"],
        "proprio": proprio,
        "text_cfg_scale": 1.0,
        "action_cfg_scale": 1.0,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "tiled": False,
    }
    if sample["context"] is not None:
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = sample["context"][0]
        infer_kwargs["context_mask"] = sample["context_mask"][0]
    else:
        infer_kwargs["prompt"] = sample["prompt"][0]

    with torch.no_grad():
        pred = model.infer(**infer_kwargs)
    frames = pred["video"]
    write_mp4_cv2(frames, args.output_mp4, fps=8)
    video_probe = probe_mp4(args.output_mp4)

    pred_action = pred.get("action")
    report = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "output_mp4": str(args.output_mp4),
        "sample_index": int(args.sample_index),
        "num_frames": len(frames),
        "num_inference_steps": int(args.num_inference_steps),
        "weight_max_abs_diff": weight_checks,
        "all_checked_weights_exact": all(value == 0.0 for value in weight_checks.values()),
        "pred_action_shape": None if pred_action is None else list(pred_action.shape),
        "video_probe": video_probe,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
