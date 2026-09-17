"""把 GigaWorld-Policy-0.5 的预训练 MoT 权重转成 FastWAM 的 `mot` state_dict。

背景与决策见 `GWP05_TRANSFER_PLAN.md`。要点：

* GWP-0.5 与 FastWAM 是同一族 MoT（Wan2.2-TI2V-5B visual expert 30/3072/14336/24x128
  + 1024 维 action expert），**维度逐位对齐，不需要任何形状手术**，差别只在 key 命名。
* 1645 / 1664 个 GWP 张量可一对一映射；弃用的 19 个是动作/状态 I/O 头
  （`action_encoder` / `action_decoder` / `state_encoder` / `action_scale_shift_table`），
  它们的维度和层数都对不上（16 vs 14 维、3 层 vs 1 层、多一个 embodiment 轴）。
* 对应地，FastWAM 侧有 4 个张量保持自有初始化
  （`mixtures.action.action_encoder.{weight,bias}` / `mixtures.action.head.{weight,bias}`）——
  这 4 个在 baseline 里**本来就是随机初始化**（`ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES`），
  所以 A/B 在这一点上是公平的。

产出 `{"mot": {...}}`，可直接用 task config 的 `resume:` 加载
（`trainer.py:327` -> `FastWAM.load_checkpoint()` -> `payload["mot"]`），**FastWAM 零代码改动**。

⚠️ `load_checkpoint` 用的是 `strict=False`，会**静默吞掉**任何 key 拼写错误。所以本脚本内建
两道验证门，默认全开，失败即抛错：

  V1 覆盖率  每个映射后的 target key 必须存在于真实 `MoT.state_dict()` 且 shape 一致；
             未命中的 target key 必须恰好是预期集合；弃用的 GWP key 必须恰好是预期集合。
  V2 血缘余弦 视觉专家每个张量 vs 本地 Wan2.2 **同名** key 的 cos-sim，应显著高于
             **错位配对**（block i vs block i+1）的对照组。GWP 的 visual expert 是
             Wan2.2 -> GigaWorld-1 -> GWP-0.5 一路微调下来的，所以同名配对必然高度相关，
             而错配 / 转置 / 层偏移会让相关性塌到 0 附近。
             这一门不需要下载 Diffusers 版 Wan2.2 —— 本地
             `checkpoints/Wan-AI/Wan2.2-TI2V-5B` 就是原始/FastWAM 命名。

用法::

    python scripts/convert_gwp05_to_fastwam.py \\
      --gwp-dir /mnt/data/zzd/giga-world-policy/Giga-World-Policy-0.5 \\
      --mode video_only \\
      --out checkpoints/gwp05_video_only.pt

    python scripts/convert_gwp05_to_fastwam.py --mode full --out checkpoints/gwp05_full.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

# --------------------------------------------------------------------------------------
# 映射规则
# --------------------------------------------------------------------------------------

#: block 内 attention 子模块的叶子名映射。GWP(diffusers) -> FastWAM(DiffSynth)。
#: `attn1` = self-attention（加 RoPE），`attn2` = cross-attention（吃 encoder_hidden_states），
#: 这一点由源码锁定：`transformer_wa_casual_mot.py` 的 `_expert_qkv` 用 attn1 且调 rope，
#: `_expert_cross_attn_output` 用 attn2。
_ATTN_LEAF = {
    "to_q": "q",
    "to_k": "k",
    "to_v": "v",
    "to_out.0": "o",
    "norm_q": "norm_q",
    "norm_k": "norm_k",
}

_ATTN_MODULE = {"attn1": "self_attn", "attn2": "cross_attn"}

#: block 内非 attention 的映射。
#:
#: `norm2 -> norm3` 是唯一的换名，而且它是被结构**唯一确定**的，不是猜的：
#:   GWP  ActionExpertBlock: norm1(affine=False) norm2(affine=True) norm3(affine=False)
#:   FastWAM DiTBlock:       norm1(affine=False) norm2(affine=False) norm3(affine=True)
#: 两边各只有一个带参数的 LayerNorm，所以只有一种可能的对应。
_BLOCK_MISC = {
    "ffn.net.0.proj": "ffn.0",
    "ffn.net.2": "ffn.2",
    "norm2": "norm3",
    "scale_shift_table": "modulation",
}

#: 顶层：visual 侧。
_TOP_VISUAL = {
    "patch_embedding": "patch_embedding",
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "proj_out": "head.head",
    "scale_shift_table": "head.modulation",
}

#: 顶层：action 侧。
_TOP_ACTION = {
    "action_condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "action_condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "action_condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "action_condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "action_condition_embedder.time_proj": "time_projection.1",
}

#: 弃用的 GWP 前缀（动作/状态 I/O 头，维度与层数都对不上）。
_DROP_PREFIXES = ("action_encoder.", "action_decoder.", "state_encoder.")
_DROP_EXACT = ("action_scale_shift_table",)

#: 对应地，FastWAM 侧预期保持自有初始化的 target key。
_EXPECTED_UNFILLED_ACTION = (
    "mixtures.action.action_encoder.weight",
    "mixtures.action.action_encoder.bias",
    "mixtures.action.head.weight",
    "mixtures.action.head.bias",
)

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(visual|action)_expert\.(.+)$")

_EXPERT_TO_MIXTURE = {"visual": "video", "action": "action"}


def _map_block_key(idx: str, expert: str, rest: str) -> str | None:
    """把 `blocks.{i}.{expert}_expert.{rest}` 映射成 `mixtures.{m}.blocks.{i}.{...}`。"""
    mixture = _EXPERT_TO_MIXTURE[expert]
    prefix = f"mixtures.{mixture}.blocks.{idx}."

    parts = rest.split(".")
    if parts[0] in _ATTN_MODULE:
        module = _ATTN_MODULE[parts[0]]
        tail = ".".join(parts[1:])
        for src_leaf, dst_leaf in _ATTN_LEAF.items():
            if tail == src_leaf:  # 无参数后缀（不会出现，但保持完备）
                return f"{prefix}{module}.{dst_leaf}"
            if tail.startswith(src_leaf + "."):
                param = tail[len(src_leaf) + 1 :]
                return f"{prefix}{module}.{dst_leaf}.{param}"
        return None

    for src_leaf, dst_leaf in _BLOCK_MISC.items():
        if rest == src_leaf:
            return f"{prefix}{dst_leaf}"
        if rest.startswith(src_leaf + "."):
            param = rest[len(src_leaf) + 1 :]
            return f"{prefix}{dst_leaf}.{param}"
    return None


def _map_top_key(key: str) -> str | None:
    for table, mixture in ((_TOP_VISUAL, "video"), (_TOP_ACTION, "action")):
        for src_leaf, dst_leaf in table.items():
            if key == src_leaf:
                return f"mixtures.{mixture}.{dst_leaf}"
            if key.startswith(src_leaf + "."):
                param = key[len(src_leaf) + 1 :]
                return f"mixtures.{mixture}.{dst_leaf}.{param}"
    return None


def map_gwp_key(key: str) -> str | None:
    """GWP-0.5 key -> FastWAM `mot` key。返回 None 表示该 key 应被弃用。"""
    if key in _DROP_EXACT or any(key.startswith(p) for p in _DROP_PREFIXES):
        return None
    match = _BLOCK_RE.match(key)
    if match is not None:
        return _map_block_key(match.group(1), match.group(2), match.group(3))
    return _map_top_key(key)


# --------------------------------------------------------------------------------------
# 目标清单（真实 MoT.state_dict 的 key + shape）
# --------------------------------------------------------------------------------------


def _is_unresolved(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def build_target_manifest(
    model_config_path: Path, action_dim: int
) -> dict[str, tuple[int, ...]]:
    """在 meta device 上构造 video/action expert，取 `mixtures.*` 的 key -> shape。

    用 meta device 是为了不分配 6 B 参数、也不需要 GPU 和 Wan2.2 权重文件；
    我们只要结构。
    """
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    cfg = OmegaConf.load(str(model_config_path))
    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)

    # 从 video_cfg 补齐 action_cfg 里那些 `${video_dit_config.x}` 形式的插值
    for key in ("num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim", "eps"):
        if _is_unresolved(action_cfg.get(key)):
            action_cfg[key] = video_cfg[key]

    # 剩下的未解析项：action_dim 由命令行给（agilex = 14），gradient checkpointing 与结构无关
    video_cfg["action_dim"] = action_dim
    action_cfg["action_dim"] = action_dim
    for c in (video_cfg, action_cfg):
        if _is_unresolved(c.get("use_gradient_checkpointing")):
            c["use_gradient_checkpointing"] = False

    manifest: dict[str, tuple[int, ...]] = {}
    with torch.device("meta"):
        video = WanVideoDiT(**video_cfg)
        action = ActionDiT(**action_cfg)
    for prefix, module in (("mixtures.video.", video), ("mixtures.action.", action)):
        for key, tensor in module.state_dict().items():
            manifest[prefix + key] = tuple(tensor.shape)
    return manifest


# --------------------------------------------------------------------------------------
# safetensors 读取
# --------------------------------------------------------------------------------------


def _shard_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No .safetensors under {root}")
    return paths


def read_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for path in _shard_paths(root):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def iter_tensors(root: Path, keys: Iterable[str] | None = None):
    wanted = None if keys is None else set(keys)
    for path in _shard_paths(root):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if wanted is not None and key not in wanted:
                    continue
                yield key, f.get_tensor(key)


# --------------------------------------------------------------------------------------
# V1 覆盖率
# --------------------------------------------------------------------------------------


def verify_coverage(
    gwp_shapes: dict[str, tuple[int, ...]],
    manifest: dict[str, tuple[int, ...]],
    mode: str,
) -> dict[str, str]:
    """返回 {gwp_key: target_key}，同时做全部断言。"""
    mapping: dict[str, str] = {}
    dropped: list[str] = []
    unmapped: list[str] = []
    shape_errors: list[str] = []
    missing_target: list[str] = []

    for key in sorted(gwp_shapes):
        target = map_gwp_key(key)
        if target is None:
            dropped.append(key)
            continue
        if mode == "video_only" and not target.startswith("mixtures.video."):
            dropped.append(key)
            continue
        if target not in manifest:
            missing_target.append(f"{key} -> {target}")
            continue
        if manifest[target] != gwp_shapes[key]:
            shape_errors.append(
                f"{key} {gwp_shapes[key]} -> {target} {manifest[target]}"
            )
            continue
        mapping[key] = target

    # 未被映射规则识别的 key（既没命中弃用名单，也没命中任何映射表）
    for key in sorted(gwp_shapes):
        if key in mapping:
            continue
        if key in dropped or any(f"{key} " in e for e in shape_errors + missing_target):
            continue
        if map_gwp_key(key) is None:
            continue
        unmapped.append(key)

    problems = []
    if missing_target:
        problems.append(f"映射到不存在的 target key ({len(missing_target)}):\n  " + "\n  ".join(missing_target[:20]))
    if shape_errors:
        problems.append(f"shape 不一致 ({len(shape_errors)}):\n  " + "\n  ".join(shape_errors[:20]))
    if unmapped:
        problems.append(f"未被任何规则识别 ({len(unmapped)}):\n  " + "\n  ".join(unmapped[:20]))

    # 弃用集合必须恰好是预期的（video_only 模式下额外弃用整个 action 侧）
    expected_drop = {
        k for k in gwp_shapes if k in _DROP_EXACT or any(k.startswith(p) for p in _DROP_PREFIXES)
    }
    if mode == "video_only":
        for k in gwp_shapes:
            t = map_gwp_key(k)
            if t is not None and t.startswith("mixtures.action."):
                expected_drop.add(k)
    if set(dropped) != expected_drop:
        extra = sorted(set(dropped) - expected_drop)
        missed = sorted(expected_drop - set(dropped))
        problems.append(f"弃用集合与预期不符。多弃: {extra[:10]}　少弃: {missed[:10]}")

    # 未被填充的 target key 必须恰好是预期集合
    filled = set(mapping.values())
    unfilled = sorted(set(manifest) - filled)
    if mode == "full":
        expected_unfilled = set(_EXPECTED_UNFILLED_ACTION)
    else:
        expected_unfilled = {k for k in manifest if k.startswith("mixtures.action.")}
    if set(unfilled) != expected_unfilled:
        extra = sorted(set(unfilled) - expected_unfilled)
        missed = sorted(expected_unfilled - set(unfilled))
        problems.append(
            f"未填充的 target key 与预期不符。多出 {len(extra)} 个: {extra[:10]}　"
            f"本应未填充却被填充 {len(missed)} 个: {missed[:10]}"
        )

    print(f"[V1] mode={mode}")
    print(f"[V1]   GWP 张量总数        : {len(gwp_shapes)}")
    print(f"[V1]   已映射              : {len(mapping)}")
    print(f"[V1]   弃用                : {len(dropped)}")
    print(f"[V1]   target 清单总数      : {len(manifest)}")
    print(f"[V1]   未填充的 target     : {len(unfilled)}  {unfilled if len(unfilled) <= 6 else ''}")

    if problems:
        raise SystemExit("[V1] ✗ 覆盖率验证失败：\n\n" + "\n\n".join(problems))
    print("[V1] ✓ 覆盖率验证通过")
    return mapping


# --------------------------------------------------------------------------------------
# V2 血缘余弦
# --------------------------------------------------------------------------------------


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.reshape(-1).to(torch.float32)
    b32 = b.reshape(-1).to(torch.float32)
    denom = a32.norm() * b32.norm()
    if denom == 0:
        return float("nan")
    return float(torch.dot(a32, b32) / denom)


def verify_ancestry(
    gwp_dir: Path,
    wan_dir: Path,
    mapping: dict[str, str],
    layers: list[int],
    threshold: float,
) -> None:
    """视觉专家 vs 本地 Wan2.2 的血缘检查。

    GWP-0.5 的 visual expert 是 Wan2.2 -> GigaWorld-1 -> GWP-0.5 微调来的，所以
    **同名配对**的 cos-sim 必然显著高；而**错位配对**（block i vs block i+1，同一叶子名）
    是对照组。若映射表写错（转置、attn1/attn2 互换、层偏移），同名配对会塌到对照组水平。
    """
    layer_prefixes = tuple(f"mixtures.video.blocks.{i}." for i in layers)

    def selected(target: str) -> bool:
        if not target.startswith("mixtures.video."):
            return False
        if ".blocks." not in target:
            return True  # 顶层全查
        return target.startswith(layer_prefixes)

    sub = {g: t for g, t in mapping.items() if selected(t)}
    if not sub:
        raise SystemExit("[V2] 没有选中任何张量，检查 --ancestry-layers")

    gwp_tensors = dict(iter_tensors(gwp_dir, sub.keys()))

    # Wan2.2 侧：需要同名 key，以及错位配对用的 key
    wan_keys_needed: set[str] = set()
    control_of: dict[str, str] = {}
    for target in sub.values():
        wan_key = target[len("mixtures.video.") :]
        wan_keys_needed.add(wan_key)
        m = re.match(r"^blocks\.(\d+)\.(.+)$", wan_key)
        if m is not None:
            ctrl = f"blocks.{(int(m.group(1)) + 1) % 30}.{m.group(2)}"
            control_of[wan_key] = ctrl
            wan_keys_needed.add(ctrl)
    wan_tensors = dict(iter_tensors(wan_dir, wan_keys_needed))

    rows = []
    for gwp_key, target in sorted(sub.items(), key=lambda kv: kv[1]):
        wan_key = target[len("mixtures.video.") :]
        if wan_key not in wan_tensors:
            raise SystemExit(f"[V2] 本地 Wan2.2 缺少 key: {wan_key}")
        aligned = _cos(gwp_tensors[gwp_key], wan_tensors[wan_key])
        ctrl_key = control_of.get(wan_key)
        control = _cos(gwp_tensors[gwp_key], wan_tensors[ctrl_key]) if ctrl_key else float("nan")
        rows.append((target, aligned, control))

    print(f"[V2] 血缘检查：{len(rows)} 个张量（层 {layers} + 全部顶层）")
    print(f"[V2]   {'target':58s} {'同名 cos':>9s} {'错位 cos':>9s}")
    for target, aligned, control in rows:
        flag = "" if aligned >= threshold else "  <-- 低"
        ctrl_str = f"{control:9.4f}" if control == control else "        -"
        print(f"[V2]   {target:58s} {aligned:9.4f} {ctrl_str}{flag}")

    aligned_vals = [a for _, a, _ in rows]
    control_vals = [c for _, _, c in rows if c == c]
    a_min = min(aligned_vals)
    a_mean = sum(aligned_vals) / len(aligned_vals)
    c_mean = (sum(control_vals) / len(control_vals)) if control_vals else float("nan")
    print(f"[V2]   同名 cos : min={a_min:.4f}  mean={a_mean:.4f}")
    print(f"[V2]   错位 cos : mean={c_mean:.4f}  (对照组，应显著更低)")

    if a_min < threshold:
        low = [t for t, a, _ in rows if a < threshold]
        raise SystemExit(
            f"[V2] ✗ 血缘验证失败：{len(low)} 个张量的同名 cos < {threshold}。"
            f"映射可能写错。\n  " + "\n  ".join(low[:20])
        )
    if control_vals and a_mean <= c_mean + 0.1:
        raise SystemExit(
            f"[V2] ✗ 血缘验证失败：同名配对({a_mean:.4f})没有显著优于错位对照({c_mean:.4f})，"
            "说明这个检查本身没有判别力，不能作为映射正确的证据。"
        )
    print("[V2] ✓ 血缘验证通过")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _parse_dtype(name: str) -> torch.dtype:
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gwp-dir", default="/mnt/data/zzd/giga-world-policy/Giga-World-Policy-0.5")
    p.add_argument("--wan-dir", default="checkpoints/Wan-AI/Wan2.2-TI2V-5B",
                   help="本地 Wan2.2（原始/FastWAM 命名），用于 V2 血缘检查")
    p.add_argument("--model-config", default="configs/model/fastwam.yaml")
    p.add_argument("--action-dim", type=int, default=14, help="agilex 双臂 = 14")
    p.add_argument("--mode", choices=["video_only", "full"], required=True,
                   help="video_only = 只迁视觉专家（C1 臂）；full = 视觉 + 动作专家（C2 臂）")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--ancestry-layers", default="0,15,29")
    p.add_argument("--ancestry-threshold", type=float, default=0.5)
    p.add_argument("--skip-ancestry", action="store_true", help="跳过 V2（不建议）")
    p.add_argument("--dry-run", action="store_true", help="只验证，不写文件")
    args = p.parse_args()

    gwp_dir = Path(args.gwp_dir)
    out_path = Path(args.out)
    dtype = _parse_dtype(args.dtype)

    print(f"[INFO] GWP dir      : {gwp_dir}")
    print(f"[INFO] model config : {args.model_config}")
    print(f"[INFO] mode         : {args.mode}")
    print(f"[INFO] out          : {out_path}  (dtype={args.dtype})")

    manifest = build_target_manifest(Path(args.model_config), args.action_dim)
    gwp_shapes = read_shapes(gwp_dir)

    mapping = verify_coverage(gwp_shapes, manifest, args.mode)

    if not args.skip_ancestry:
        layers = [int(x) for x in args.ancestry_layers.split(",") if x.strip()]
        verify_ancestry(gwp_dir, Path(args.wan_dir), mapping, layers, args.ancestry_threshold)
    else:
        print("[V2] 已跳过（--skip-ancestry）")

    if args.dry_run:
        print("[INFO] --dry-run：验证通过，未写文件")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mot: dict[str, torch.Tensor] = {}
    for gwp_key, tensor in iter_tensors(gwp_dir, mapping.keys()):
        mot[mapping[gwp_key]] = tensor.to(dtype=dtype).contiguous()
    if len(mot) != len(mapping):
        raise SystemExit(f"[ERROR] 写出张量数 {len(mot)} != 映射数 {len(mapping)}")

    payload = {
        "mot": mot,
        "gwp05_transfer": {
            "source": str(gwp_dir),
            "mode": args.mode,
            "mapped": len(mot),
            "dropped": len(gwp_shapes) - len(mot),
            "unfilled_target": sorted(set(manifest) - set(mot)),
            "dtype": args.dtype,
            "action_dim": args.action_dim,
        },
    }
    torch.save(payload, str(out_path))
    size_gb = out_path.stat().st_size / 1024**3
    print(f"[INFO] ✓ 已写出 {out_path}  ({size_gb:.2f} GiB, {len(mot)} 张量)")
    print("[INFO] 用法：在 task config 里设 resume: " + str(out_path))
    print(json.dumps(payload["gwp05_transfer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
