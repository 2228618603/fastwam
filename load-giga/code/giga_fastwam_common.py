#!/usr/bin/env python3
"""Shared utilities for GigaWorld Pro -> FastWAM video expert migration."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEMANTIC_REJECT_PREFIXES = (
    "patch_short.",
    "patch_mid.",
    "patch_long.",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_shape(shape: Any) -> list[int]:
    return [int(x) for x in list(shape)]


def count_params(shapes: dict[str, list[int]]) -> int:
    total = 0
    for shape in shapes.values():
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n
    return total


def load_giga_shapes(giga_transformer_dir: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    index_path = giga_transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = dict(index["weight_map"])
    shapes: dict[str, list[int]] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        shard_path = giga_transformer_dir / shard
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for key in keys:
                if key not in available:
                    raise KeyError(f"{key} listed in index but absent from {shard_path}")
                shapes[key] = normalize_shape(f.get_slice(key).get_shape())
    return shapes, weight_map


def load_fastwam_video_shapes(fastwam_root: Path, proprio_dim: int = 14) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if str(fastwam_root) not in sys.path:
        sys.path.insert(0, str(fastwam_root))
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    cfg = OmegaConf.load(fastwam_root / "configs/model/fastwam.yaml")
    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    video_cfg["action_dim"] = proprio_dim
    video_cfg["use_gradient_checkpointing"] = False
    with torch.device("meta"):
        model = WanVideoDiT(**video_cfg)
    shapes = {key: normalize_shape(value.shape) for key, value in model.state_dict().items()}
    return shapes, dict(video_cfg)


def map_giga_key_to_fastwam(key: str) -> tuple[str | None, str | None]:
    if any(key.startswith(prefix) for prefix in SEMANTIC_REJECT_PREFIXES):
        return None, "semantic_reject"
    if "lora" in key.lower():
        return None, "semantic_reject"
    if "gan" in key.lower():
        return None, "semantic_reject"
    if key == "patch_embedding.weight":
        return "patch_embedding.weight", "shape_sensitive"
    if key == "patch_embedding.bias":
        return "patch_embedding.bias", None

    replacements = (
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("norm_out.scale_shift_table", "head.modulation"),
        ("proj_out.", "head.head."),
    )
    for src, dst in replacements:
        if key.startswith(src) or key == src:
            return key.replace(src, dst, 1), None

    match = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if not match:
        return None, "unmapped"
    layer, suffix = match.groups()
    block_replacements = (
        ("attn1.to_q.", "self_attn.q."),
        ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."),
        ("attn1.to_out.0.", "self_attn.o."),
        ("attn1.norm_q.", "self_attn.norm_q."),
        ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_q.", "cross_attn.q."),
        ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."),
        ("attn2.to_out.0.", "cross_attn.o."),
        ("attn2.norm_q.", "cross_attn.norm_q."),
        ("attn2.norm_k.", "cross_attn.norm_k."),
        ("ffn.net.0.proj.", "ffn.0."),
        ("ffn.net.2.", "ffn.2."),
        ("norm2.", "norm3."),
    )
    if suffix == "scale_shift_table":
        return f"blocks.{layer}.modulation", None
    for src, dst in block_replacements:
        if suffix.startswith(src):
            return f"blocks.{layer}.{suffix.replace(src, dst, 1)}", None
    return None, "unmapped"


def build_mapping_report(
    giga_shapes: dict[str, list[int]],
    fastwam_shapes: dict[str, list[int]],
) -> dict[str, Any]:
    mapped: dict[str, dict[str, Any]] = {}
    shape_mismatch: dict[str, dict[str, Any]] = {}
    semantic_reject: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, dict[str, Any]] = {}
    duplicate_targets: dict[str, list[str]] = {}
    target_to_sources: dict[str, list[str]] = {}

    for source_key, source_shape in sorted(giga_shapes.items()):
        target_key, reason = map_giga_key_to_fastwam(source_key)
        if target_key is None:
            bucket = semantic_reject if reason == "semantic_reject" else unmapped
            bucket[source_key] = {"shape": source_shape, "reason": reason}
            continue
        target_shape = fastwam_shapes.get(target_key)
        if target_shape is None:
            unmapped[source_key] = {
                "target_key": target_key,
                "source_shape": source_shape,
                "reason": "target_absent",
            }
            continue
        if source_shape != target_shape:
            shape_mismatch[source_key] = {
                "target_key": target_key,
                "source_shape": source_shape,
                "target_shape": target_shape,
                "reason": reason or "shape_mismatch",
            }
            continue
        mapped[source_key] = {"target_key": target_key, "shape": source_shape}
        target_to_sources.setdefault(target_key, []).append(source_key)

    for target_key, sources in target_to_sources.items():
        if len(sources) > 1:
            duplicate_targets[target_key] = sources

    mapped_target_keys = {entry["target_key"] for entry in mapped.values()}
    missing_targets = {
        key: shape
        for key, shape in sorted(fastwam_shapes.items())
        if key not in mapped_target_keys
    }

    mapped_shapes = {entry["target_key"]: entry["shape"] for entry in mapped.values()}
    return {
        "mapped": mapped,
        "mapped_summary": {
            "keys": len(mapped),
            "params": count_params(mapped_shapes),
        },
        "shape_mismatch": shape_mismatch,
        "semantic_reject": semantic_reject,
        "unmapped": unmapped,
        "duplicate_targets": duplicate_targets,
        "missing_targets": missing_targets,
        "missing_summary": {
            "keys": len(missing_targets),
            "params": count_params(missing_targets),
        },
    }


def load_tensor_from_index(
    giga_transformer_dir: Path,
    weight_map: dict[str, str],
    key: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    shard = weight_map[key]
    with safe_open(giga_transformer_dir / shard, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
