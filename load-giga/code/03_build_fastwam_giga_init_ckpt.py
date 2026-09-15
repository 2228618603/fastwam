#!/usr/bin/env python3
"""Build a FastWAM resume checkpoint with GigaWorld Pro video expert weights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from giga_fastwam_common import dtype_from_name, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--mapped-video-pt", type=Path, required=True)
    parser.add_argument("--action-dit-pt", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--proprio-dim", type=int, default=14)
    return parser.parse_args()


def assert_no_block_missing(keys: list[str]) -> None:
    block_missing = [
        key
        for key in keys
        if key.startswith("blocks.")
        and (
            ".self_attn." in key
            or ".cross_attn." in key
            or ".ffn." in key
            or key.endswith(".modulation")
            or ".norm3." in key
        )
    ]
    if block_missing:
        sample = ", ".join(block_missing[:10])
        raise RuntimeError(f"Unexpected missing core block keys ({len(block_missing)}): {sample}")


def main() -> None:
    args = parse_args()
    fastwam_root = args.fastwam_root.resolve()
    if str(fastwam_root) not in sys.path:
        sys.path.insert(0, str(fastwam_root))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/data/chw/fastwam/checkpoints")

    from fastwam.runtime import create_fastwam

    dtype = dtype_from_name(args.dtype)
    cfg = OmegaConf.load(fastwam_root / "configs/model/fastwam.yaml")
    model_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=False)
    model_cfg.pop("_target_", None)
    model_cfg["proprio_dim"] = args.proprio_dim
    model_cfg["load_text_encoder"] = False
    model_cfg["action_dit_pretrained_path"] = str(args.action_dit_pt)
    model_cfg["model_dtype"] = dtype
    model_cfg["device"] = args.device
    model_cfg["mot_checkpoint_mixed_attn"] = False
    model_cfg["compile_training_denoise"] = False
    model_cfg["video_dit_config"] = dict(model_cfg["video_dit_config"])
    model_cfg["action_dit_config"] = dict(model_cfg["action_dit_config"])
    model_cfg["video_dit_config"]["action_dim"] = args.proprio_dim
    model_cfg["action_dit_config"]["action_dim"] = args.proprio_dim
    model_cfg["video_dit_config"]["use_gradient_checkpointing"] = False
    model_cfg["action_dit_config"]["use_gradient_checkpointing"] = False

    model = create_fastwam(**model_cfg)
    mapped_payload = torch.load(args.mapped_video_pt, map_location="cpu", weights_only=False)
    mapped_sd = mapped_payload["video_expert_state_dict"]
    mapped_sd = {key: value.to(dtype=dtype, device=args.device) for key, value in mapped_sd.items()}
    incompatible = model.video_expert.load_state_dict(mapped_sd, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected mapped keys while loading video expert: {unexpected_keys[:20]}")
    assert_no_block_missing(missing_keys)

    checkpoint = {
        "mot": {key: value.detach().cpu() for key, value in model.mot.state_dict().items()},
        "proprio_encoder": (
            None
            if model.proprio_encoder is None
            else {key: value.detach().cpu() for key, value in model.proprio_encoder.state_dict().items()}
        ),
        "step": 0,
        "torch_dtype": str(dtype),
        "meta": {
            "source_mapped_video": str(args.mapped_video_pt),
            "action_expert_source": str(args.action_dit_pt),
            "video_expert_base": "Wan-AI/Wan2.2-TI2V-5B",
            "tokenizer_model_id": model_cfg["tokenizer_model_id"],
            "load_text_encoder": False,
            "proprio_dim": args.proprio_dim,
            "forbidden_sources": [
                "libero/step_021700.pt",
                "robotwin/step_029355.pt",
                "robotwin_release/robotwin_uncond_3cam_384.pt",
            ],
        },
    }
    if checkpoint["proprio_encoder"] is None:
        raise RuntimeError("FastWAM checkpoint requires proprio_encoder, but model.proprio_encoder is None")

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_pt)
    report = {
        "output_pt": str(args.output_pt),
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "missing_key_count": len(missing_keys),
        "mapped_key_count": len(mapped_sd),
        "mapped_param_count": sum(t.numel() for t in mapped_sd.values()),
        "mot_key_count": len(checkpoint["mot"]),
        "proprio_encoder_keys": sorted(checkpoint["proprio_encoder"]),
        "meta": checkpoint["meta"],
    }
    write_json(args.output_report, report)
    print(f"wrote {args.output_pt}")
    print(f"wrote {args.output_report}")
    print(f"unexpected_keys={len(unexpected_keys)} missing_keys={len(missing_keys)}")
    print(f"mot_keys={len(checkpoint['mot'])} dtype={checkpoint['torch_dtype']}")


if __name__ == "__main__":
    main()
