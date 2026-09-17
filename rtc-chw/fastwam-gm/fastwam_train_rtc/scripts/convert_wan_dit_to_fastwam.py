#!/usr/bin/env python
"""把一份「Wan2.2-TI2V-5B 命名的 DiT 权重」转成 FastWAM 的 `mot` state_dict。

用途:在自有具身数据上对原生 Wan2.2 DiT 做过视频预训练之后,把那份权重当作
FastWAM 的 **video expert 初始化**,替换掉原生 Wan2.2 权重。

与 `convert_gwp05_to_fastwam.py` 的区别:GWP-0.5 是 diffusers 命名,需要一整张
映射表;这里的输入已经是 **DiffSynth / FastWAM 原生命名**(即 `blocks.*` /
`patch_embedding` / `text_embedding` / `time_embedding` / `time_projection` /
`head`),所以映射规则退化成「统一加 `mixtures.video.` 前缀」——
唯一需要做的事是**把两道验证门保留下来**。

为什么门不能省:`FastWAM.load_checkpoint()`(`fastwam.py:1417`)对 `mot` 用
`strict=False`,任何 key 拼写错误都会被**静默吞掉**,训练照样跑、loss 曲线照样
"看着正常",只是初始化其实是原生 Wan2.2 —— 而一次静默失败要烧掉整轮训练时间。

  V1 覆盖率   每个 target key 必须存在于真实 `MoT.state_dict()` 且 shape 一致;
              未命中的 target key 必须**恰好**是 action expert 的全集
              (本脚本只迁视觉专家,action expert 保持插值初始化)。
  V2 血缘余弦 每个张量 vs 本地原生 Wan2.2 **同名** key 的 cos-sim,应显著高于
              **错位配对**(block i vs block i+1)的对照组。输入是从 Wan2.2
              微调下来的,所以同名配对必然高度相关;而层偏移 / 转置 / 张量错配
              都会让相关性塌到 0 附近。这一门同时能验出「输入其实是从头训练的
              模型」这种情况。

产出 `{"mot": {...}}`,直接用 task config 的 `resume:` 加载,**FastWAM 零代码改动**
—— 与 `ab_B`(RoboTwin 热启动)、`ab_C1/C2`(GWP 迁移)走的是同一条路。

注意 payload 里**不含** `proprio_encoder`:`load_checkpoint` 会 warn 并保持当前
初始化(`fastwam.py:1427`),这与 baseline 的随机初始化一致,所以对照仍是单变量。

用法::

    python scripts/convert_wan_dit_to_fastwam.py \\
      --dit /mnt/data/zzd/DiffSynth-Studio/runs/agilex_emptybox_470_video_pretrain_100k_log100_20260904/step-60000.safetensors \\
      --out checkpoints/wanpre60k_video_only.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 复用 GWP 转换器里那两个与映射表无关的通用件:
#   build_target_manifest —— 在 meta device 上构造真实 MoT,拿 key -> shape
#   iter_tensors / read_shapes —— safetensors 读取(支持单文件或分片目录)
from convert_gwp05_to_fastwam import (  # noqa: E402
    build_target_manifest,
    iter_tensors,
    read_shapes,
)

VIDEO_PREFIX = "mixtures.video."

#: V2 抽查的 block 号。取首/中/末,能覆盖到「整体层偏移」这类错误。
ANCESTRY_LAYERS = (0, 7, 15, 22, 29)

#: V2 判据。只作用于 ndim>=2 的权重矩阵,理由见 `verify_ancestry` 的 docstring。
#: 实测同名 >= 0.981、错位 ~0.000,所以这两个阈值离实测值都很远,不会边界抖动。
SAME_FLOOR = 0.95
GAP_FLOOR = 0.50


# --------------------------------------------------------------------------------------
# 输入路径:允许单个 .safetensors 文件,也允许一个含分片的目录
# --------------------------------------------------------------------------------------


def _as_root(path: Path) -> Path:
    """`iter_tensors`/`read_shapes` 接受目录;单文件时用它的父目录会连带读进兄弟文件,
    所以单文件走独立实现。"""
    return path


def _read_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    if path.is_dir():
        return read_shapes(path)
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def _iter_tensors(path: Path, keys=None):
    if path.is_dir():
        yield from iter_tensors(path, keys)
        return
    wanted = None if keys is None else set(keys)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if wanted is not None and key not in wanted:
                continue
            yield key, f.get_tensor(key)


# --------------------------------------------------------------------------------------
# V1 覆盖率
# --------------------------------------------------------------------------------------


def verify_coverage(
    src_shapes: dict[str, tuple[int, ...]],
    manifest: dict[str, tuple[int, ...]],
) -> dict[str, str]:
    """返回 {src_key: target_key};任何不一致直接抛错。"""
    mapping: dict[str, str] = {}
    shape_errors: list[str] = []
    missing_target: list[str] = []

    for key in sorted(src_shapes):
        target = VIDEO_PREFIX + key
        if target not in manifest:
            missing_target.append(f"{key} -> {target}")
            continue
        if manifest[target] != src_shapes[key]:
            shape_errors.append(f"{target}: src{src_shapes[key]} != model{manifest[target]}")
            continue
        mapping[key] = target

    if missing_target:
        raise SystemExit(
            "[V1 FAIL] 以下 src key 映射后在 MoT.state_dict() 里不存在"
            f"({len(missing_target)} 个):\n  " + "\n  ".join(missing_target[:20])
        )
    if shape_errors:
        raise SystemExit(
            f"[V1 FAIL] shape 不一致({len(shape_errors)} 个):\n  " + "\n  ".join(shape_errors[:20])
        )

    # 未被填充的 target key 必须恰好是 action expert 的全集
    filled = set(mapping.values())
    unfilled = sorted(k for k in manifest if k not in filled)
    unexpected = [k for k in unfilled if not k.startswith("mixtures.action.")]
    if unexpected:
        raise SystemExit(
            "[V1 FAIL] video expert 有 target key 没被填充"
            f"({len(unexpected)} 个,说明输入缺张量):\n  " + "\n  ".join(unexpected[:20])
        )

    n_video = sum(1 for k in manifest if k.startswith(VIDEO_PREFIX))
    if len(filled) != n_video:
        raise SystemExit(f"[V1 FAIL] 填充 {len(filled)} != video expert 张量总数 {n_video}")

    print(
        f"[V1 OK] {len(mapping)} 个张量全部映射且 shape 一致 "
        f"(= video expert 全部 {n_video} 个);保持自有初始化的 {len(unfilled)} 个"
        f"全部是 mixtures.action.*"
    )
    return mapping


# --------------------------------------------------------------------------------------
# V2 血缘余弦
# --------------------------------------------------------------------------------------


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """float64 计算,避免 3072~14336 维点积在 fp32 上累积出 >1 的余弦值。"""
    x = a.flatten().to(torch.float64)
    y = b.flatten().to(torch.float64)
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((x @ y) / denom)


def verify_ancestry(src_path: Path, wan_ref: Path, mapping: dict[str, str]) -> None:
    """同名配对 vs 错位配对(block i vs block i+1)的 cos-sim 对比。

    ⚠️ 判据分两层,原因都在实测中出现过:

    **(a) 同名配对下限只对 ndim >= 2 收紧。** 1-D 偏置的余弦不是血缘信号 —— bias 范数
    很小,微调时相对移动可以很大,实测 `blocks.7.cross_attn.v.bias` 同名 cos 只有
    0.866(相对差 0.50),而同一份权重的全部 2-D 矩阵都 >= 0.981。拿 1-D 最小值当门
    会误杀。1-D 只用中位数做一个宽松的同源性检查。

    **(b) 错位对照只对 ndim == 2 的线性权重矩阵收紧。** 有两类张量天然在层间高度相关,
    错位对照对它们无意义:

      * RMSNorm / LayerNorm 权重(`norm_q.weight` 等):各元素都在 1.0 附近,任意两个
        之间 cos 都 ~1.0,实测错位对照最高 0.99972。
      * adaLN 的 `modulation`(`(1,6,3072)` 的 scale_shift_table):实测错位 0.61~0.86。

    真正的线性权重矩阵上这个判据极其锐利:实测同名 >= 0.981、错位中位数 0.0002
    (最小 -0.003),层偏移 / 转置 / 张量错配都会掉到 0 附近,相隔近两个数量级。
    """
    layer_prefixes = tuple(f"blocks.{i}." for i in ANCESTRY_LAYERS)
    top_keys = ("patch_embedding.weight", "head.head.weight", "time_embedding.0.weight")

    def selected(key: str) -> bool:
        return key.startswith(layer_prefixes) or key in top_keys

    probe = [k for k in mapping if selected(k)]
    if not probe:
        raise SystemExit("[V2 FAIL] 没有抽到任何探针张量")

    # 错位对照:blocks.{i}.rest 配 blocks.{i+1}.rest
    ctrl_pairs: dict[str, str] = {}
    for key in probe:
        if not key.startswith("blocks."):
            continue
        _, idx, rest = key.split(".", 2)
        cand = f"blocks.{int(idx) + 1}.{rest}"
        if cand in mapping:
            ctrl_pairs[key] = cand

    need = set(probe) | set(ctrl_pairs.values())
    src = {k: v for k, v in _iter_tensors(src_path, need)}
    ref = {k: v for k, v in _iter_tensors(wan_ref, need)}

    absent = need - set(ref)
    if absent:
        raise SystemExit(
            f"[V2 FAIL] 参考 Wan2.2 权重缺少 {len(absent)} 个探针 key: {sorted(absent)[:5]}"
        )

    strict = [k for k in probe if src[k].ndim >= 2]
    loose = [k for k in probe if src[k].ndim < 2]
    # 错位判据组:只要真正的线性权重矩阵(排除 modulation(1,6,3072) 与
    # patch_embedding(3072,48,1,2,2)),理由见 docstring (b)
    gap_group = [k for k in strict if src[k].ndim == 2 and k in ctrl_pairs]
    if not strict:
        raise SystemExit("[V2 FAIL] 没有抽到任何 ndim>=2 的探针张量")
    if not gap_group:
        raise SystemExit("[V2 FAIL] 没有抽到任何可做错位对照的 2-D 线性权重")

    same = {k: _cos(src[k], ref[k]) for k in probe}
    ctrl = {k: _cos(src[k], ref[v]) for k, v in ctrl_pairs.items()}

    def _stat(keys: list[str], table: dict[str, float]) -> str:
        vals = sorted(table[k] for k in keys if k in table)
        if not vals:
            return "n=0"
        return (f"n={len(vals)} min={vals[0]:.5f} "
                f"median={vals[len(vals) // 2]:.5f} max={vals[-1]:.5f}")

    print(f"[V2] 同名配对 ndim>=2 (下限判据组)  : {_stat(strict, same)}")
    print(f"[V2] 同名配对 ndim==1 (仅中位数检查): {_stat(loose, same)}")
    print(f"[V2] 错位对照 2-D 线性权重(判据组) : {_stat(gap_group, ctrl)}")
    print(f"[V2] 错位对照 其余(仅报告,不判据)  : "
          f"{_stat([k for k in probe if k in ctrl and k not in gap_group], ctrl)}"
          "  <- norm/modulation 等层间天然相关")

    worst = sorted(gap_group, key=lambda k: same[k] - ctrl[k])[:5]
    print("     错位判据组 gap 最小 5 个:")
    for k in worst:
        print(f"       {k:38s} same={same[k]:.5f}  shifted={ctrl[k]:.5f}  "
              f"gap={same[k] - ctrl[k]:+.5f}")

    same_min = min(same[k] for k in strict)
    if same_min < SAME_FLOOR:
        worst_key = min(strict, key=lambda k: same[k])
        raise SystemExit(
            f"[V2 FAIL] ndim>=2 张量的同名配对最低 cos {same_min:.5f} < {SAME_FLOOR} "
            f"({worst_key}) —— 输入不像是从这份 Wan2.2 权重微调出来的"
            "(可能是从头训练、或权重版本不对)"
        )

    bad_gap = [(k, same[k], ctrl[k]) for k in gap_group
               if same[k] - ctrl[k] < GAP_FLOOR]
    if bad_gap:
        detail = "\n  ".join(f"{k}: same={s:.5f} shifted={c:.5f}" for k, s, c in bad_gap[:10])
        raise SystemExit(
            f"[V2 FAIL] {len(bad_gap)} 个 2-D 线性权重的同名优势不足 {GAP_FLOOR} —— "
            f"疑似层偏移或张量错配:\n  {detail}"
        )

    loose_vals = sorted(same[k] for k in loose)
    if loose_vals and loose_vals[len(loose_vals) // 2] < SAME_FLOOR:
        raise SystemExit(
            f"[V2 FAIL] 1-D 张量的同名 cos 中位数 "
            f"{loose_vals[len(loose_vals) // 2]:.5f} < {SAME_FLOOR} —— 整体不同源"
        )

    min_gap = min(same[k] - ctrl[k] for k in gap_group)
    print(f"[V2 OK] {len(strict)} 个 ndim>=2 张量同名 cos >= {same_min:.5f};"
          f"{len(gap_group)} 个 2-D 线性权重的同名优势 >= {min_gap:+.5f};血缘成立")


# --------------------------------------------------------------------------------------


def _parse_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dit", required=True, type=Path,
                    help="输入权重:单个 .safetensors 文件,或含分片的目录")
    ap.add_argument("--out", required=True, type=Path, help="输出 .pt 路径")
    ap.add_argument("--model-config", type=Path, default=Path("configs/model/fastwam.yaml"))
    ap.add_argument("--action-dim", type=int, default=14, help="agilex 双臂 = 14")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--wan-ref", type=Path,
                    default=Path("checkpoints/Wan-AI/Wan2.2-TI2V-5B"),
                    help="V2 血缘校验用的原生 Wan2.2 权重目录")
    ap.add_argument("--skip-ancestry", action="store_true", help="跳过 V2(不建议)")
    args = ap.parse_args()

    dtype = _parse_dtype(args.dtype)

    print(f"[INFO] 输入      : {args.dit}")
    print(f"[INFO] model cfg : {args.model_config}  (action_dim={args.action_dim})")

    src_shapes = _read_shapes(args.dit)
    print(f"[INFO] 输入张量数: {len(src_shapes)}")

    manifest = build_target_manifest(args.model_config, args.action_dim)
    n_v = sum(1 for k in manifest if k.startswith(VIDEO_PREFIX))
    n_a = len(manifest) - n_v
    print(f"[INFO] MoT 结构  : video {n_v} + action {n_a} = {len(manifest)} 个张量")

    mapping = verify_coverage(src_shapes, manifest)

    if args.skip_ancestry:
        print("[V2 SKIP] 已按 --skip-ancestry 跳过血缘校验")
    else:
        verify_ancestry(args.dit, args.wan_ref, mapping)

    mot: dict[str, torch.Tensor] = {}
    n_param = 0
    for key, tensor in _iter_tensors(args.dit, mapping.keys()):
        mot[mapping[key]] = tensor.to(dtype).contiguous()
        n_param += tensor.numel()
    if len(mot) != len(mapping):
        raise SystemExit(f"[FAIL] 只读到 {len(mot)}/{len(mapping)} 个张量")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mot": mot}, str(args.out))
    size_gib = args.out.stat().st_size / 1024 ** 3
    print(f"\n[DONE] {args.out}  {len(mot)} 张量 / {n_param / 1e9:.4f} B 参数 / "
          f"{size_gib:.2f} GiB / {args.dtype}")

    sidecar = args.out.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "source": str(args.dit.resolve()),
        "model_config": str(args.model_config),
        "action_dim": args.action_dim,
        "dtype": args.dtype,
        "num_tensors": len(mot),
        "num_params": n_param,
        "mode": "video_expert_only",
        "action_expert": "保持 ActionDiT 插值初始化(未迁移)",
    }, ensure_ascii=False, indent=2))
    print(f"[DONE] provenance -> {sidecar}")


if __name__ == "__main__":
    main()
