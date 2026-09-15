#!/usr/bin/env python3
"""Convert GigaWorld Pro tensors into a FastWAM video expert state dict."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from giga_fastwam_common import (
    build_mapping_report,
    load_fastwam_video_shapes,
    load_giga_shapes,
    load_tensor_from_index,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--giga-transformer-dir", type=Path, required=True)
    parser.add_argument("--inspect-report", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--proprio-dim", type=int, default=14)
    parser.add_argument("--dtype", default="source", choices=("source", "bf16", "fp16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_report = read_json(args.inspect_report)
    giga_shapes, weight_map = load_giga_shapes(args.giga_transformer_dir)
    fastwam_shapes, _ = load_fastwam_video_shapes(args.fastwam_root, proprio_dim=args.proprio_dim)
    mapping = build_mapping_report(giga_shapes, fastwam_shapes)
    if mapping["mapped"] != inspect_report.get("mapped"):
        raise RuntimeError("Current mapping differs from inspect report. Re-run Stage 3 inspect first.")

    dtype = None
    if args.dtype != "source":
        from giga_fastwam_common import dtype_from_name

        dtype = dtype_from_name(args.dtype)

    state_dict: dict[str, torch.Tensor] = {}
    for source_key, entry in mapping["mapped"].items():
        target_key = entry["target_key"]
        tensor = load_tensor_from_index(args.giga_transformer_dir, weight_map, source_key, dtype=dtype)
        expected_shape = tuple(fastwam_shapes[target_key])
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"Unexpected shape drift for {source_key}: {tuple(tensor.shape)} != {expected_shape}")
        if target_key in state_dict:
            raise RuntimeError(f"Duplicate target key: {target_key}")
        state_dict[target_key] = tensor.contiguous()

    meta = {
        "source": str(args.giga_transformer_dir),
        "target": "FastWAM video_expert",
        "direct_load": False,
        "dtype": args.dtype,
        "mapped_keys": len(state_dict),
        "mapped_params": sum(t.numel() for t in state_dict.values()),
        "skipped_reason": {
            "patch_embedding.weight": "shape mismatch; Giga in_channels=148, FastWAM in_dim=48",
            "patch_short.*": "Giga FunControl/history memory branch absent in FastWAM",
            "patch_mid.*": "Giga FunControl/history memory branch absent in FastWAM",
            "patch_long.*": "Giga FunControl/history memory branch absent in FastWAM",
        },
    }
    payload = {"video_expert_state_dict": state_dict, "meta": meta}
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_pt)

    report = {
        "output_pt": str(args.output_pt),
        "meta": meta,
        "mapped_keys": sorted(state_dict),
        "mapped_summary": mapping["mapped_summary"],
        "shape_mismatch": mapping["shape_mismatch"],
        "semantic_reject": mapping["semantic_reject"],
        "unmapped": mapping["unmapped"],
        "missing_targets": mapping["missing_targets"],
    }
    write_json(args.output_report, report)
    print(f"wrote {args.output_pt}")
    print(f"wrote {args.output_report}")
    print(f"mapped_keys={meta['mapped_keys']} mapped_params={meta['mapped_params']}")
    print(f"has_patch_weight={'patch_embedding.weight' in state_dict}")
    print(f"has_patch_bias={'patch_embedding.bias' in state_dict}")


if __name__ == "__main__":
    main()
