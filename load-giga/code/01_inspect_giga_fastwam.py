#!/usr/bin/env python3
"""Audit GigaWorld Pro transformer keys against FastWAM video expert keys."""

from __future__ import annotations

import argparse
from pathlib import Path

from giga_fastwam_common import (
    build_mapping_report,
    count_params,
    load_fastwam_video_shapes,
    load_giga_shapes,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--giga-transformer-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--proprio-dim", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    giga_cfg = read_json(args.giga_transformer_dir / "config.json")
    giga_shapes, _ = load_giga_shapes(args.giga_transformer_dir)
    fastwam_shapes, fastwam_video_cfg = load_fastwam_video_shapes(args.fastwam_root, proprio_dim=args.proprio_dim)
    mapping = build_mapping_report(giga_shapes, fastwam_shapes)

    payload = {
        "source": str(args.giga_transformer_dir),
        "target": "FastWAM WanVideoDiT video expert",
        "giga_config": {
            "_class_name": giga_cfg.get("_class_name"),
            "model_type": giga_cfg.get("model_type"),
            "in_channels": giga_cfg.get("in_channels"),
            "out_channels": giga_cfg.get("out_channels"),
            "num_layers": giga_cfg.get("num_layers"),
            "num_attention_heads": giga_cfg.get("num_attention_heads"),
            "attention_head_dim": giga_cfg.get("attention_head_dim"),
            "ffn_dim": giga_cfg.get("ffn_dim"),
            "text_dim": giga_cfg.get("text_dim"),
            "patch_size": giga_cfg.get("patch_size"),
            "weight_keys": len(giga_shapes),
            "params": count_params(giga_shapes),
        },
        "fastwam_video_config": {
            "in_dim": fastwam_video_cfg.get("in_dim"),
            "out_dim": fastwam_video_cfg.get("out_dim"),
            "num_layers": fastwam_video_cfg.get("num_layers"),
            "num_heads": fastwam_video_cfg.get("num_heads"),
            "attn_head_dim": fastwam_video_cfg.get("attn_head_dim"),
            "ffn_dim": fastwam_video_cfg.get("ffn_dim"),
            "text_dim": fastwam_video_cfg.get("text_dim"),
            "patch_size": fastwam_video_cfg.get("patch_size"),
            "state_keys": len(fastwam_shapes),
            "params": count_params(fastwam_shapes),
        },
        **mapping,
    }
    write_json(args.output_json, payload)

    shape_mismatch = payload["shape_mismatch"]
    semantic_reject = payload["semantic_reject"]
    md = [
        "# GigaWorld Pro -> FastWAM Inspect Report",
        "",
        f"- source: `{payload['source']}`",
        f"- target: `{payload['target']}`",
        "",
        "## Config",
        "",
        f"- Giga class/model: `{payload['giga_config']['_class_name']}` / `{payload['giga_config']['model_type']}`",
        f"- Giga in/out: `{payload['giga_config']['in_channels']}` -> `{payload['giga_config']['out_channels']}`",
        f"- FastWAM in/out: `{payload['fastwam_video_config']['in_dim']}` -> `{payload['fastwam_video_config']['out_dim']}`",
        f"- layers/hidden heads: `{payload['giga_config']['num_layers']}` layers, `{payload['giga_config']['num_attention_heads']}` heads, head dim `{payload['giga_config']['attention_head_dim']}`",
        "",
        "## Transfer Summary",
        "",
        f"- mapped keys: `{payload['mapped_summary']['keys']}`",
        f"- mapped params: `{payload['mapped_summary']['params']}`",
        f"- shape mismatches: `{len(shape_mismatch)}`",
        f"- semantic rejects: `{len(semantic_reject)}`",
        f"- unmapped: `{len(payload['unmapped'])}`",
        f"- missing FastWAM targets after mapping: `{payload['missing_summary']['keys']}`",
        "",
        "## Required Rejections",
        "",
    ]
    for key in ("patch_embedding.weight", "patch_short.weight", "patch_mid.weight", "patch_long.weight"):
        location = "shape_mismatch" if key in shape_mismatch else "semantic_reject" if key in semantic_reject else "not_found"
        md.append(f"- `{key}`: `{location}`")
    md.extend(["", "## Shape Mismatch Sample", ""])
    for key, entry in list(shape_mismatch.items())[:40]:
        md.append(f"- `{key}` -> `{entry['target_key']}`: `{entry['source_shape']}` vs `{entry['target_shape']}`")
    md.extend(["", "## Semantic Reject Sample", ""])
    for key, entry in list(semantic_reject.items())[:40]:
        md.append(f"- `{key}`: `{entry['reason']}`")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md) + "\n")

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"mapped_keys={payload['mapped_summary']['keys']} mapped_params={payload['mapped_summary']['params']}")
    print(f"shape_mismatch={len(shape_mismatch)} semantic_reject={len(semantic_reject)} unmapped={len(payload['unmapped'])}")


if __name__ == "__main__":
    main()
