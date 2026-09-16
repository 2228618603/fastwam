#!/usr/bin/env python3
"""Small offline AgileX action evaluation for FastWAM.

The default run is intentionally conservative: single visible GPU, batch size 1,
50 validation windows, and action-only inference. This avoids competing with
ongoing 8-GPU training jobs while still giving a quick signal.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_REQUESTED_DATASET = Path(
    "/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_071"
)
DEFAULT_RESOLVED_DATASET = Path(
    "/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_010000.pt"
)
DEFAULT_RUN_CONFIG = Path("/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/config.yaml")
DEFAULT_OUTPUT_DIR = Path("/mnt/data/chw/fastwam/evaluate_results/agilex_empty_box_giga_init/eval_step_010000_50w")
DEFAULT_STATS = Path("/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json")
DEFAULT_TEXT_CACHE = Path("/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711")
DEFAULT_ACTION_DIT = Path("/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


ACTION_GROUPS: dict[str, list[int]] = {
    "left_arm_joints": list(range(0, 6)),
    "left_gripper": [6],
    "right_arm_joints": list(range(7, 13)),
    "right_gripper": [13],
    "all_joints": list(range(0, 6)) + list(range(7, 13)),
    "all_grippers": [6, 13],
}
JOINT_GROUPS: dict[str, list[int]] = {
    "left_arm_joints": ACTION_GROUPS["left_arm_joints"],
    "right_arm_joints": ACTION_GROUPS["right_arm_joints"],
    "all_joints": ACTION_GROUPS["all_joints"],
}
RAD_TO_DEG = 180.0 / np.pi


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def resolve_dataset_root(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_REQUESTED_DATASET and DEFAULT_RESOLVED_DATASET.exists():
        return DEFAULT_RESOLVED_DATASET
    raise FileNotFoundError(f"dataset root does not exist: {path}")


def load_config(args: argparse.Namespace) -> DictConfig:
    register_default_resolvers()
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    else:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    cfg.data.train.dataset_dirs = [str(args.dataset_root_resolved)]
    cfg.data.val.dataset_dirs = [str(args.dataset_root_resolved)]
    cfg.data.train.pretrained_norm_stats = str(args.stats)
    cfg.data.val.pretrained_norm_stats = str(args.stats)
    cfg.data.train.text_embedding_cache_dir = str(args.text_cache)
    cfg.data.val.text_embedding_cache_dir = str(args.text_cache)
    cfg.data.train.use_text_embed_cache = bool(args.use_text_embed_cache)
    cfg.data.val.use_text_embed_cache = bool(args.use_text_embed_cache)
    cfg.data.train.is_training_set = False
    cfg.data.val.is_training_set = False
    cfg.data.train.skip_padding_as_possible = False
    cfg.data.val.skip_padding_as_possible = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.resume = str(args.checkpoint)
    cfg.model.load_text_encoder = not bool(args.use_text_embed_cache)
    cfg.model.action_dit_pretrained_path = str(args.action_dit)
    cfg.model.compile_training_denoise = False
    cfg.model.mot_checkpoint_mixed_attn = False
    return cfg


def check_paths(args: argparse.Namespace, cfg: DictConfig) -> dict[str, Any]:
    paths = {
        "repo_root": REPO_ROOT,
        "config": args.config,
        "dataset_root_requested": args.dataset_root,
        "dataset_root_resolved": args.dataset_root_resolved,
        "checkpoint": args.checkpoint,
        "stats": args.stats,
        "text_cache": args.text_cache,
        "action_dit": args.action_dit,
    }
    status = {}
    for name, path in paths.items():
        if path is None:
            status[name] = {"path": None, "exists": None}
            continue
        status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
    missing = [name for name, item in status.items() if item["exists"] is False and name != "dataset_root_requested"]
    if missing:
        raise FileNotFoundError(f"missing required paths: {missing}")
    result = {
        "stage": "check",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": args.device,
        "paths": status,
        "config": {
            "mixed_precision": str(cfg.mixed_precision),
            "model_target": str(cfg.model.get("_target_")),
            "val_target": str(cfg.data.val.get("_target_")),
            "num_frames": int(cfg.data.val.num_frames),
            "action_video_freq_ratio": int(cfg.data.val.action_video_freq_ratio),
        },
    }
    write_json(args.output_dir / "stage0_check.json", result)
    return result


def build_dataset(cfg: DictConfig):
    dataset = instantiate(cfg.data.val)
    return dataset


def build_model(cfg: DictConfig, args: argparse.Namespace):
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if str(args.device).startswith("cuda") and args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction), device=torch.device(args.device))

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/data/chw/fastwam/checkpoints")
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(str(cfg.mixed_precision)))
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    model.load_checkpoint(str(args.checkpoint))
    model.eval()
    return model


def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    required = ["video", "action", "proprio", "prompt", "image_is_pad", "action_is_pad", "proprio_is_pad"]
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"sample missing keys: {missing}")
    video = sample["video"]
    action = sample["action"]
    proprio = sample["proprio"]
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"video must be [C,T,H,W], got {type(video)} {getattr(video, 'shape', None)}")
    if not isinstance(action, torch.Tensor) or action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"action must be [T,14], got {type(action)} {getattr(action, 'shape', None)}")
    if not isinstance(proprio, torch.Tensor) or proprio.ndim != 2 or proprio.shape[1] != 14:
        raise ValueError(f"proprio must be [T,14], got {type(proprio)} {getattr(proprio, 'shape', None)}")
    return {
        "video_shape": list(video.shape),
        "action_shape": list(action.shape),
        "proprio_shape": list(proprio.shape),
        "has_gt_action": "gt_action" in sample,
        "has_context": "context" in sample,
        "prompt": sample["prompt"],
    }


def to_model_kwargs(model: Any, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    video = sample["video"]
    input_image = video[:, 0].unsqueeze(0)
    action = sample["action"]
    proprio = sample["proprio"][0]
    kwargs = {
        "prompt": sample["prompt"],
        "input_image": input_image,
        "action_horizon": int(action.shape[0]),
        "proprio": proprio,
        "num_inference_steps": int(args.num_inference_steps),
        "sigma_shift": args.sigma_shift,
        "seed": int(args.seed),
        "rand_device": args.rand_device,
        "tiled": False,
        "compile_action_infer": False,
    }
    if "context" in sample:
        kwargs["prompt"] = None
        kwargs["context"] = sample["context"]
        kwargs["context_mask"] = sample["context_mask"]

    params = inspect.signature(model.infer_action).parameters
    if "num_video_frames" in params:
        kwargs["num_video_frames"] = int(video.shape[1])
    if "compile_action_infer" not in params:
        kwargs.pop("compile_action_infer", None)
    if "sigma_shift" not in params:
        kwargs.pop("sigma_shift", None)
    return kwargs


def infer_one(model: Any, sample: dict[str, Any], args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    kwargs = to_model_kwargs(model, sample, args)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.infer_action(**kwargs)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    latency = time.perf_counter() - start
    pred = output["action"]
    if pred.ndim != 2 or pred.shape[1] != 14:
        raise ValueError(f"predicted action must be [T,14], got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise ValueError("predicted action contains NaN or Inf")
    return pred.detach().cpu(), latency


def denormalize_action_norm(dataset: Any, pred_norm: torch.Tensor) -> torch.Tensor:
    processor = dataset.lerobot_dataset.processor
    normalizer = processor.normalizer.normalizers["action"]["default"]
    return normalizer.backward(pred_norm)


def as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def per_window_metrics(gt: np.ndarray, pred: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    gt = as_float_array(gt, name=f"{prefix}gt")
    pred = as_float_array(pred, name=f"{prefix}pred")
    if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 14:
        raise ValueError(f"expected matching [T,14] actions, got gt={gt.shape}, pred={pred.shape}")
    diff = pred - gt
    abs_diff = np.abs(diff)
    key = f"{prefix}_" if prefix else ""
    result = {
        f"{key}mse_all": float(np.mean(diff**2)),
        f"{key}mae_all": float(np.mean(abs_diff)),
    }
    for name, dims in ACTION_GROUPS.items():
        group = diff[:, dims]
        result[f"{key}mse_{name}"] = float(np.mean(group**2))
        result[f"{key}mae_{name}"] = float(np.mean(np.abs(group)))
    if prefix == "raw":
        for name, dims in JOINT_GROUPS.items():
            group = diff[:, dims]
            mae_rad = float(np.mean(np.abs(group)))
            rmse_rad = float(np.sqrt(np.mean(group**2)))
            result[f"{key}{name}_mae_rad"] = mae_rad
            result[f"{key}{name}_rmse_rad"] = rmse_rad
            result[f"{key}{name}_mae_deg"] = mae_rad * RAD_TO_DEG
            result[f"{key}{name}_rmse_deg"] = rmse_rad * RAD_TO_DEG
            result[f"{key}{name}_p95_abs_deg"] = float(np.percentile(np.abs(group) * RAD_TO_DEG, 95))
    return result


@dataclass
class RunningActionMetrics:
    count_windows: int = 0
    count_elements: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    per_dim_sum_sq: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float64))
    per_dim_sum_abs: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float64))
    group_sum_sq: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in ACTION_GROUPS})
    group_sum_abs: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in ACTION_GROUPS})
    group_count: dict[str, int] = field(default_factory=lambda: {k: 0 for k in ACTION_GROUPS})
    joint_sum_sq: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in JOINT_GROUPS})
    joint_sum_abs: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in JOINT_GROUPS})
    joint_count: dict[str, int] = field(default_factory=lambda: {k: 0 for k in JOINT_GROUPS})

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        gt = as_float_array(gt, name="gt")
        pred = as_float_array(pred, name="pred")
        if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 14:
            raise ValueError(f"expected matching [T,14] actions, got gt={gt.shape}, pred={pred.shape}")
        diff = pred - gt
        abs_diff = np.abs(diff)
        self.count_windows += 1
        self.count_elements += int(diff.size)
        self.sum_sq += float(np.sum(diff**2))
        self.sum_abs += float(np.sum(abs_diff))
        self.per_dim_sum_sq += np.sum(diff**2, axis=0)
        self.per_dim_sum_abs += np.sum(abs_diff, axis=0)
        for name, dims in ACTION_GROUPS.items():
            group = diff[:, dims]
            self.group_sum_sq[name] += float(np.sum(group**2))
            self.group_sum_abs[name] += float(np.sum(np.abs(group)))
            self.group_count[name] += int(group.size)
        for name, dims in JOINT_GROUPS.items():
            group = diff[:, dims]
            self.joint_sum_sq[name] += float(np.sum(group**2))
            self.joint_sum_abs[name] += float(np.sum(np.abs(group)))
            self.joint_count[name] += int(group.size)

    def summary(self, *, prefix: str = "", include_angles: bool = False) -> dict[str, float | int]:
        if self.count_windows == 0:
            return {"count_windows": 0}
        key = f"{prefix}_" if prefix else ""
        result: dict[str, float | int] = {
            "count_windows": self.count_windows,
            f"{key}mse_all": self.sum_sq / self.count_elements,
            f"{key}mae_all": self.sum_abs / self.count_elements,
        }
        for name in ACTION_GROUPS:
            denom = max(1, self.group_count[name])
            result[f"{key}mse_{name}"] = self.group_sum_sq[name] / denom
            result[f"{key}mae_{name}"] = self.group_sum_abs[name] / denom
        if include_angles:
            for name in JOINT_GROUPS:
                denom = max(1, self.joint_count[name])
                mae_rad = self.joint_sum_abs[name] / denom
                rmse_rad = float(np.sqrt(self.joint_sum_sq[name] / denom))
                result[f"{key}{name}_mae_rad"] = mae_rad
                result[f"{key}{name}_rmse_rad"] = rmse_rad
                result[f"{key}{name}_mae_deg"] = mae_rad * RAD_TO_DEG
                result[f"{key}{name}_rmse_deg"] = rmse_rad * RAD_TO_DEG
        return result

    def per_dim(self, *, prefix: str = "") -> list[dict[str, float | int]]:
        denom = max(1, self.count_windows * 32)
        key = f"{prefix}_" if prefix else ""
        return [
            {
                "dim": int(i),
                f"{key}mse": float(self.per_dim_sum_sq[i] / denom),
                f"{key}mae": float(self.per_dim_sum_abs[i] / denom),
            }
            for i in range(14)
        ]


def select_indices(dataset: Any, args: argparse.Namespace) -> list[int]:
    total = len(dataset)
    indices = list(range(total))
    if args.window_strategy == "shuffle":
        rng = random.Random(args.seed)
        rng.shuffle(indices)
    start = max(int(args.start_index), 0)
    indices = indices[start : start + int(args.max_windows)]
    if not indices:
        raise ValueError(f"selected zero windows from dataset of length {total}")
    return indices


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_data(args: argparse.Namespace, cfg: DictConfig) -> dict[str, Any]:
    dataset = build_dataset(cfg)
    sample = dataset[int(args.sample_index)]
    result = {
        "stage": "data",
        "dataset_len": len(dataset),
        "sample_index": int(args.sample_index),
        "sample": validate_sample(sample),
        "dataset_root_requested": str(args.dataset_root),
        "dataset_root_resolved": str(args.dataset_root_resolved),
    }
    write_json(args.output_dir / "stage1_data_contract.json", result)
    return result


def run_smoke(args: argparse.Namespace, cfg: DictConfig) -> dict[str, Any]:
    dataset = build_dataset(cfg)
    sample = dataset[int(args.sample_index)]
    validate_sample(sample)
    model = build_model(cfg, args)
    pred_norm, latency = infer_one(model, sample, args)
    gt_norm = sample["action"].detach().cpu()
    raw_gt = denormalize_action_norm(dataset, gt_norm)
    raw_pred = denormalize_action_norm(dataset, pred_norm)
    result = {
        "stage": "smoke",
        "sample_index": int(args.sample_index),
        "latency_sec": latency,
        "gt_norm_shape": list(gt_norm.shape),
        "pred_norm_shape": list(pred_norm.shape),
        "metrics_norm": per_window_metrics(gt_norm.numpy(), pred_norm.numpy(), prefix="norm"),
        "metrics_raw": per_window_metrics(raw_gt.numpy(), raw_pred.numpy(), prefix="raw"),
    }
    write_json(args.output_dir / "stage2_smoke_one_window.json", result)
    return result


def run_eval(args: argparse.Namespace, cfg: DictConfig) -> dict[str, Any]:
    dataset = build_dataset(cfg)
    model = build_model(cfg, args)
    indices = select_indices(dataset, args)
    per_window_path = args.output_dir / "per_window.jsonl"
    worst_path = args.output_dir / "worst_windows.jsonl"
    for path in (per_window_path, worst_path):
        if path.exists():
            path.unlink()

    norm_metrics = RunningActionMetrics()
    raw_metrics = RunningActionMetrics()
    per_episode: dict[int, RunningActionMetrics] = defaultdict(RunningActionMetrics)
    latencies: list[float] = []
    worst: list[dict[str, Any]] = []

    for ordinal, index in enumerate(indices):
        sample = dataset[int(index)]
        pred_norm, latency = infer_one(model, sample, args)
        gt_norm = sample["action"].detach().cpu()
        norm_metrics.update(gt_norm.numpy(), pred_norm.numpy())

        row = {
            "ordinal": int(ordinal),
            "dataset_index": int(index),
            "latency_sec": float(latency),
            **per_window_metrics(gt_norm.numpy(), pred_norm.numpy(), prefix="norm"),
        }
        raw_gt = denormalize_action_norm(dataset, gt_norm)
        raw_pred = denormalize_action_norm(dataset, pred_norm)
        raw_metrics.update(raw_gt.numpy(), raw_pred.numpy())
        row.update(per_window_metrics(raw_gt.numpy(), raw_pred.numpy(), prefix="raw"))
        episode = int(sample.get("episode_index", -1))
        row["episode_index"] = episode
        per_episode[episode].update(gt_norm.numpy(), pred_norm.numpy())
        latencies.append(float(latency))
        append_jsonl(per_window_path, row)
        worst.append(row)
        worst = sorted(worst, key=lambda item: item["norm_mse_all"], reverse=True)[: int(args.keep_worst)]
        print(
            f"[eval] {ordinal + 1}/{len(indices)} index={index} "
            f"norm_mse={row['norm_mse_all']:.6f} latency={latency:.2f}s",
            flush=True,
        )

    for row in worst:
        append_jsonl(worst_path, row)

    latency_payload = {
        "count": len(latencies),
        "mean_sec": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p50_sec": percentile(latencies, 50),
        "p95_sec": percentile(latencies, 95),
        "p99_sec": percentile(latencies, 99),
        "max_sec": max(latencies) if latencies else 0.0,
    }
    summary = {
        "stage": "eval",
        "count_windows": len(indices),
        "checkpoint": str(args.checkpoint),
        "config": None if args.config is None else str(args.config),
        "task": args.task,
        "dataset_root_requested": str(args.dataset_root),
        "dataset_root_resolved": str(args.dataset_root_resolved),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "device": args.device,
        "metrics_norm": norm_metrics.summary(prefix="norm"),
        "metrics_raw": raw_metrics.summary(prefix="raw", include_angles=True),
        "latency": latency_payload,
    }
    write_json(args.output_dir / "args.json", vars(args))
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_dim.json", {"norm": norm_metrics.per_dim(prefix="norm"), "raw": raw_metrics.per_dim(prefix="raw")})
    write_json(
        args.output_dir / "per_episode.json",
        {str(ep): metrics.summary(prefix="norm") for ep, metrics in sorted(per_episode.items())},
    )
    write_json(args.output_dir / "latency.json", latency_payload)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "data", "smoke", "eval"], default="eval")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--task", default="agilex_empty_box_uncond_3cam384")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_REQUESTED_DATASET)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--text-cache", type=Path, default=DEFAULT_TEXT_CACHE)
    parser.add_argument("--action-dit", type=Path, default=DEFAULT_ACTION_DIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--keep-worst", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--window-strategy", choices=["sequential", "shuffle"], default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--use-text-embed-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("override", nargs="*", help="Optional OmegaConf dotlist overrides, e.g. model.load_text_encoder=true")
    args = parser.parse_args()
    args.dataset_root_resolved = resolve_dataset_root(args.dataset_root)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    if args.stage == "check":
        result = check_paths(args, cfg)
    elif args.stage == "data":
        check_paths(args, cfg)
        result = run_data(args, cfg)
    elif args.stage == "smoke":
        check_paths(args, cfg)
        result = run_smoke(args, cfg)
    else:
        check_paths(args, cfg)
        result = run_eval(args, cfg)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
