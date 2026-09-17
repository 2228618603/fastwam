#!/usr/bin/env python
"""离线批量推理评测：在自己的数据上跑推理，把**原始结果存盘**，指标另算。

设计：推理（要 GPU、慢）与可视化（纯 CPU）分离。本脚本只负责推理 + 存盘，
出图交给 scripts/visualize_eval.py 读这些文件 —— 这样改图不用重跑推理。

仓库里原本没有这个能力：
  - trainer.evaluate() 每 eval_every 步只做 1 个样本/卡，**每步还换一批随机样本**
    （trainer.py:436-437 按 global_step+rank 播种），所以 checkpoint 之间不可比；
    且只存拼接 mp4 和标量，pred_action 用完就丢，没法事后分析
  - experiments/{libero,robotwin}/ 的 manager 依赖仿真器，真机数据用不上
  - scripts/dryrun_fastwam.py 只测速度，用合成输入

本脚本让**所有 checkpoint 评完全相同的样本**，把 checkpoint 选择从"靠窗口平均
绕过采样噪声"变成直接比较。

反归一化逻辑与 trainer.py:503-535 **完全一致**（走 processor 的
action_state_merger.backward -> normalizer.backward -> merger.forward），
所以这里的 action_l2 与训练日志里的 eval/action_l2 可直接对比。
注意 trainer 的 action_l2 其实是 **MSE**（diff.pow(2).mean()），单位是物理量的平方；
本脚本额外给出 RMSE 与逐维误差，更好解读。

用法（推荐：config 驱动）:
  source env.sh
  python scripts/offline_eval.py --config configs/eval/final_A_sweep.yaml
  python scripts/offline_eval.py --config configs/eval/smoke.yaml --num-samples 4

用法（旧式单 ckpt，与 run_final.sh 的调用保持兼容）:
  python scripts/offline_eval.py \
    --ckpt runs/agilex_uncond_3cam_384_1e-4/ab_A/checkpoints/weights/step_005000.pt \
    --task agilex_uncond_3cam_384_1e-4 \
    --num-samples 64 --out-dir eval_offline/ab_A_step5000

输出布局:
  --config 模式（可多 ckpt，按 ckpt 分片到多卡）:
    <out_dir>/config.resolved.yaml   实际生效配置快照（worker 读的就是它）
    <out_dir>/ckpt/<tag>/...         每个 ckpt 一个子目录，内部结构同下
    <out_dir>/compare.csv            每 ckpt 一行的聚合指标（含 step / epoch）
    <out_dir>/sweep.json             扫描级元信息 + 各 ckpt 聚合指标
    <out_dir>/logs/gpu*.log          各分片日志
  单 ckpt 目录内部（两种模式一致）:
    samples/sample_XXXX.npz   pred/gt action（归一化+物理量）、proprio、误差
    videos/sample_XXXX.mp4    可选，pred/VAE重建/GT 竖向拼接
    metrics.csv               每样本一行
    summary.json              聚合指标 + 运行元信息 + 时间标注参数
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

register_default_resolvers()

REPO_ROOT = Path(__file__).resolve().parent.parent

# 与 robot_video_dataset.py:23 的 DEFAULT_PROMPT 保持一致（preflight 校验 text-embed 缓存用）
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

# config 的完整默认值。--config 读进来的 YAML 只需写要改的项。
DEFAULTS = {
    "task": "agilex_uncond_3cam_384_1e-4",
    "out_dir": None,
    "checkpoints": None,          # str | list | {dir, glob, every} | list of those
    "highlight": None,            # 轨迹叠加图用的 ckpt tag，<=3 个
    "reference": None,            # 成对差分图的基准 ckpt tag，默认取最后一个
    "dataset": {
        "dataset_dir": None,      # None = 用 task 配置里的
        "split": "val",
        "episodes": None,         # None=沿用 val_set_proportion 划分；[..]=显式；"all"=全部
        "norm_stats": "./runs/_shared/dataset_stats.json",
    },
    "sampling": {
        "num_samples": 64,
        "per_episode": None,      # 设了就每 episode 等间隔取 N 个（覆盖更均衡）
        "stride": 0,
        "seed": 42,
        "skip_padded": False,     # 跳过 episode 尾部会被 padding 的窗口
    },
    "inference": {
        "mode": "joint",         # joint=动作+视频(慢)；action=只动作(快)
        "num_inference_steps": 10,
        "sigma_shift": None,
        "save_video": False,
        "action_crosscheck": True,   # infer_joint 是否额外跑一遍 infer_action 做校验
                                     # （trainer 默认开；关掉能省约 40% joint 耗时）
    },
    "annotate": {
        "steps_per_epoch": None,  # 用于把 ckpt step 换算成 epoch；None 则图上不标 epoch
        "fps": None,              # action 步 -> 秒；None 则从数据集 meta/info.json 读
        "video_fps": None,        # 导出 mp4 的真实播放速率；None 则 fps/action_video_freq_ratio
    },
    "runtime": {
        "gpus": [0],
        "overrides": [],          # 额外 hydra 覆盖
    },
}


# ---------------------------------------------------------------- config
def _merge(base: dict, over) -> dict:
    """把 over 深合并到 base 上（不改 base）。over 里的 None 视为"没写"，不覆盖。"""
    out = dict(base)
    if not over:
        return out
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def load_config(args) -> DictConfig:
    """DEFAULTS <- YAML <- 命令行，后者优先。"""
    cfg = dict(DEFAULTS)
    if args.config:
        raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"{args.config} 顶层必须是 mapping")
        cfg = _merge(cfg, raw)

    # 命令行覆盖（只有显式给了才覆盖，所以 argparse 的 default 全是 None）
    cli = {
        "task": args.task,
        "out_dir": args.out_dir,
        "dataset": {"split": args.split, "dataset_dir": args.dataset_dir,
                    "norm_stats": args.norm_stats,
                    "episodes": _parse_episodes(args.episodes)},
        "sampling": {"num_samples": args.num_samples, "stride": args.stride,
                     "per_episode": args.per_episode, "seed": args.seed,
                     "skip_padded": True if args.skip_padded else None},
        "inference": {"mode": args.mode, "num_inference_steps": args.num_inference_steps,
                      "sigma_shift": args.sigma_shift,
                      "save_video": True if args.save_video else None},
        "runtime": {"gpus": _parse_gpus(args.gpus), "overrides": args.overrides or None},
    }
    cfg = _merge(cfg, _prune_none(cli))
    if args.ckpt:
        cfg["checkpoints"] = list(args.ckpt)
    if cfg["out_dir"] is None:
        raise SystemExit("必须给 --out-dir 或在 config 里写 out_dir")
    if not cfg["checkpoints"]:
        raise SystemExit("必须给 --ckpt 或在 config 里写 checkpoints")
    return OmegaConf.create(cfg)


def _prune_none(d):
    """递归去掉值为 None 的键，这样它们不会覆盖 DEFAULTS / YAML。"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _prune_none(v)
            if v:
                out[k] = v
        elif v is not None:
            out[k] = v
    return out


def _parse_episodes(s):
    if s is None:
        return None
    if s.strip().lower() == "all":
        return "all"
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def _parse_gpus(s):
    """'0-7' / '0,2,5' / '3' -> [int]"""
    if s is None:
        return None
    out: list[int] = []
    for part in re.split(r"[,\s]+", s.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ---------------------------------------------------------------- checkpoint 解析
def ckpt_step(path) -> int | None:
    """从 step_077690.pt 之类的文件名里抠出训练步数。抠不到返回 None。"""
    m = re.search(r"(\d+)", Path(path).stem.replace("step_", "step"))
    return int(m.group(1)) if m else None


def ckpt_tag(path) -> str:
    return Path(path).stem


def resolve_checkpoints(spec) -> list[Path]:
    """把 config 里的 checkpoints 写法统一成排好序的路径列表。

    支持：单个路径字符串 / 路径列表 / {dir, glob, every} / 以上的混合列表。
    """
    if spec is None:
        return []
    if isinstance(spec, (str, Path)):
        spec = [spec]
    if isinstance(spec, DictConfig) or isinstance(spec, dict):
        spec = [spec]

    found: list[Path] = []
    for item in spec:
        if isinstance(item, (str, Path)):
            found.append(Path(item))
            continue
        item = OmegaConf.to_container(item, resolve=True) if isinstance(item, DictConfig) else dict(item)
        d = Path(item["dir"])
        pat = item.get("glob", "*.pt")
        every = int(item.get("every", 1) or 1)
        matched = sorted(d.glob(pat), key=lambda p: (ckpt_step(p) is None, ckpt_step(p) or 0, p.name))
        if not matched:
            raise SystemExit(f"checkpoints: {d}/{pat} 没匹配到任何文件")
        if every > 1:
            matched = matched[::every]
        found += matched

    # 去重（保序），再按 step 排序，图上 x 轴才是单调的
    uniq = list(dict.fromkeys(p.resolve() for p in found))
    return sorted(uniq, key=lambda p: (ckpt_step(p) is None, ckpt_step(p) or 0, p.name))


# ---------------------------------------------------------------- preflight
def preflight(cfg: DictConfig, ckpts: list[Path], dataset_dir: Path) -> None:
    """进推理前把能静默失真的东西全查一遍。用错统计量目前不会报错，只会给出错的数。"""
    print(">>> preflight ...")
    missing = [str(p) for p in ckpts if not p.is_file()]
    if missing:
        raise SystemExit("以下 checkpoint 不存在或不可读:\n  " + "\n  ".join(missing))
    print(f"    checkpoint {len(ckpts)} 份，全部可读")

    stats = Path(cfg.dataset.norm_stats)
    if not stats.is_file():
        raise SystemExit(
            f"归一化统计量不存在: {stats}\n"
            "  用错/缺失统计量会让指标完全失真且**不报错**。训练用的那份在 "
            "runs/_shared/dataset_stats.json 或 runs/<task>/<run>/dataset_stats.json"
        )
    print(f"    归一化统计量 {stats}")

    if not (dataset_dir / "meta" / "info.json").is_file():
        raise SystemExit(f"数据集目录不像 LeRobot 数据集（缺 meta/info.json）: {dataset_dir}")

    ds_cfg = _task_data_cfg(cfg)

    # shape_meta 要的列必须都在。缺列的话 robot_video_dataset.__getitem__:283-292 会
    # 静默换随机样本重试 5 次，最后才在很深的地方崩 —— 而那已经烧掉了 40 多秒建模型。
    feats = set(json.loads((dataset_dir / "meta" / "info.json").read_text())["features"])
    want = (
        [f"observation.images.{m['key']}" if m["key"] != "default" else "observation.images"
         for m in ds_cfg["shape_meta"]["images"]]
        + [f"observation.state.{m['key']}" if m["key"] != "default" else "observation.state"
           for m in ds_cfg["shape_meta"]["state"]]
        + [f"action.{m['key']}" if m["key"] != "default" else "action"
           for m in ds_cfg["shape_meta"]["action"]]
    )
    absent_cols = [k for k in want if k not in feats]
    if absent_cols:
        raise SystemExit(
            f"{dataset_dir} 缺 shape_meta 要求的列:\n  " + "\n  ".join(absent_cols)
            + f"\n  该数据集实际有: {sorted(feats)}\n"
            "  列名不匹配要么换数据集，要么先转名（参考 scripts/convert_agilex_to_fastwam.py）。"
        )
    print(f"    数据列齐全（{len(want)} 项）")

    # text-embed 缓存：缺一条就在第一个样本崩，提前查
    import hashlib
    tasks_file = dataset_dir / "meta" / "tasks.jsonl"
    cache_dir = Path(ds_cfg["text_embedding_cache_dir"])
    ctx_len = int(ds_cfg["context_len"])
    if tasks_file.is_file():
        absent = []
        for line in tasks_file.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)["task"]
            h = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode("utf-8")).hexdigest()
            f = cache_dir / f"{h}.t5_len{ctx_len}.wan22ti2v5b.pt"
            if not f.is_file():
                absent.append((task, f))
        if absent:
            msg = "\n".join(f"    - {t[:60]}... -> {f}" for t, f in absent)
            raise SystemExit(
                f"text embedding 缓存缺 {len(absent)} 条（推理第一个样本就会崩）:\n{msg}\n"
                f"  先跑: python scripts/precompute_text_embeds.py "
                f"--dataset-dir {dataset_dir} --out-dir {cache_dir}"
            )
        print(f"    text-embed 缓存 {cache_dir} 全部命中")

    eps = cfg.dataset.episodes
    if eps is not None and eps != "all":
        total = json.loads((dataset_dir / "meta" / "info.json").read_text())["total_episodes"]
        bad = [int(e) for e in eps if not 0 <= int(e) < total]
        if bad:
            raise SystemExit(f"episodes {bad} 越界（该数据集共 {total} 个 episode）")
        print(f"    episodes {list(map(int, eps))} 在范围内（共 {total} 个）")


def _task_data_cfg(cfg: DictConfig) -> dict:
    """只为 preflight 取几个数据配置字段，避免这时候就 compose 整个 hydra。"""
    with initialize(config_path="../configs", version_base="1.3"):
        c = compose(config_name="train", overrides=[f"task={cfg.task}"])
    node = c.data[cfg.dataset.split]
    return {
        "text_embedding_cache_dir": node.text_embedding_cache_dir,
        "context_len": node.context_len,
        "dataset_dirs": list(node.dataset_dirs),
        "num_frames": int(node.num_frames),
        "action_video_freq_ratio": int(node.action_video_freq_ratio),
        "shape_meta": OmegaConf.to_container(node.shape_meta, resolve=True),
    }


# ---------------------------------------------------------------- 反归一化
def denorm_action(processor, action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """把归一化的 action [T,D] 还原成物理量。逻辑照抄 trainer.py:503-535。

    必须经过 merger.backward -> normalizer.backward -> merger.forward 这一圈，
    因为 normalizer 是按 shape_meta 里的子 key（joint / gripper_position）分别
    统计的，不能对拼接后的 14 维直接反归一化。

    ⚠️ merger 的 _crop 断言 ndim==3（action_state_merger.py:56-57），**action 和
    state 都要有 batch 维**。漏掉 state 那一句是本脚本第一版唯一的、也是致命的 bug
    （.run_final.log:43-54 两次 AssertionError 都出在这）。
    """
    action_btd = (action.unsqueeze(0) if action.ndim == 2 else action).detach().to("cpu", torch.float32)
    state_btd = (proprio.unsqueeze(0) if proprio.ndim == 2 else proprio).detach().to("cpu", torch.float32)
    batch = {"action": action_btd, "state": state_btd}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {m["key"]: batch["action"][m["key"]].squeeze(0)
                   for m in processor.shape_meta["action"]},
        "state": {m["key"]: batch["state"][m["key"]].squeeze(0)
                  for m in processor.shape_meta["state"]},
    }
    merged = processor.action_state_merger.forward(merged)
    out = merged["action"]
    if out.ndim != 2:
        raise ValueError(f"反归一化后的 action 应为 [T,D]，得到 {tuple(out.shape)}")
    return out


def action_dim_names(info_path: Path) -> list[str]:
    """从数据集 info.json 拿 action 各维的真实名字，画图时好读。"""
    try:
        feats = json.loads(info_path.read_text())["features"]
        for key in ("action", "actions"):
            if key in feats and feats[key].get("names"):
                return [str(n) for n in feats[key]["names"]]
    except Exception:
        pass
    return []


def dataset_fps(info_path: Path) -> float | None:
    try:
        return float(json.loads(info_path.read_text())["fps"])
    except Exception:
        return None


# ---------------------------------------------------------------- 样本选取
class EpisodeIndex:
    """帧下标 -> (episode id, episode 内偏移)。

    episode_data_index 给的是各 episode 在拼接后数据集里的 [from, to) 帧区间
    （base_lerobot_dataset.py:134-137），episode_ids 是同序的原始 episode 编号。
    """

    def __init__(self, ds, num_frames: int):
        lds = ds.lerobot_dataset
        self.starts = [int(x) for x in lds.episode_data_index["from"].tolist()]
        self.ends = [int(x) for x in lds.episode_data_index["to"].tolist()]
        ids: list[int] = []
        for repo in lds.dataset_dirs:                      # 与帧拼接顺序一致
            ids += [int(e) for e in lds.episode_ids[str(repo)]]
        self.ids = ids
        self.num_frames = int(num_frames)
        if len(self.ids) != len(self.starts):
            raise RuntimeError(
                f"episode 数不一致: episode_ids={len(self.ids)} vs "
                f"episode_data_index={len(self.starts)}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def locate(self, idx: int) -> tuple[int, int, int]:
        """-> (episode id, episode 内帧号, episode 长度)"""
        j = bisect.bisect_right(self.ends, idx)
        j = min(j, len(self.ids) - 1)
        return self.ids[j], idx - self.starts[j], self.ends[j] - self.starts[j]

    def unpadded_ranges(self) -> list[tuple[int, int]]:
        """每个 episode 里**不会触发 padding** 的起始帧区间 [a, b)。

        窗口跨 num_frames 帧（obs 33 帧 / action 32 步），起点 f 需要 f..f+32 都在
        episode 内，即 f < ep_len - (num_frames-1)；再往后尾部会被复制填充
        （sliding_window_with_replication），误差会被人为压低。
        """
        out = []
        for a, b in zip(self.starts, self.ends):
            out.append((a, max(a + 1, b - (self.num_frames - 1))))
        return out


def pick_indices(ds, epi: EpisodeIndex, s: DictConfig) -> tuple[list[int], int]:
    """选样本下标。返回 (下标列表, 因 skip_padded 被排除的候选数)。

    等间隔而不是随机：保证覆盖整个 split，且**完全可复现** —— 这是多 ckpt
    可比性的前提（所有 ckpt 拿到的是同一批下标）。
    """
    n = len(ds)
    ranges = epi.unpadded_ranges() if s.skip_padded else list(zip(epi.starts, epi.ends))
    excluded = n - sum(b - a for a, b in ranges) if s.skip_padded else 0

    if s.per_episode:
        idxs: list[int] = []
        for a, b in ranges:
            k = min(int(s.per_episode), b - a)
            if k > 0:
                idxs += np.linspace(a, b - 1, k).astype(int).tolist()
    else:
        pool = np.concatenate([np.arange(a, b) for a, b in ranges]) if ranges else np.arange(n)
        if int(s.stride) > 0:
            idxs = pool[:: int(s.stride)][: int(s.num_samples)].tolist()
        else:
            k = min(int(s.num_samples), pool.size)
            idxs = pool[np.linspace(0, pool.size - 1, k).astype(int)].tolist()

    return sorted(dict.fromkeys(int(i) for i in idxs)), int(excluded)


# ---------------------------------------------------------------- 单 ckpt 评测
def load_and_verify(model, path: Path) -> dict:
    """加载权重，并确认**真的加载上了**。

    mot 走 load_state_dict(strict=False)（fastwam.py:1214），键名不匹配会被静默
    忽略 —— 那样跑出来的是随机初始化的分数，看起来还挺"正常"。这里比对键集合。
    """
    payload = model.load_checkpoint(str(path))
    have = set(model.mot.state_dict().keys())
    got = set(payload["mot"].keys()) if "mot" in payload else set()
    missing, unexpected = sorted(have - got), sorted(got - have)
    info = {
        "mot_total": len(have),
        "mot_loaded": len(have & got),
        "mot_missing": len(missing),
        "mot_unexpected": len(unexpected),
        "step_in_payload": payload.get("step"),
    }
    del payload
    if missing:
        raise SystemExit(
            f"{path.name}: mot 有 {len(missing)}/{len(have)} 个权重没被加载 —— "
            f"模型结构与 checkpoint 不匹配（--task 是否与训练时一致？）\n"
            f"  例如: {missing[:5]}"
        )
    if unexpected:
        print(f"    ⚠️ checkpoint 里有 {len(unexpected)} 个用不上的键（通常无害）")
    print(f"    权重校验 {info['mot_loaded']}/{info['mot_total']} 键全部匹配")
    return info


def eval_one_ckpt(model, ds, processor, idxs, epi, cfg, ckpt_path: Path, out: Path) -> dict:
    """在固定的一批样本上评一个 ckpt，结果落 out 目录，返回 summary。"""
    inf = cfg.inference
    ann = cfg.annotate
    (out / "samples").mkdir(parents=True, exist_ok=True)
    if inf.save_video and inf.mode == "joint":
        (out / "videos").mkdir(parents=True, exist_ok=True)

    tag = ckpt_tag(ckpt_path)
    print(f"\n>>> [{tag}] 加载权重 {ckpt_path}")
    t_load = time.perf_counter()
    weight_info = load_and_verify(model, ckpt_path)
    model.eval()
    print(f"    加载耗时 {time.perf_counter() - t_load:.1f}s")

    dim_names = action_dim_names(Path(ds.lerobot_dataset.dataset_dirs[0]) / "meta" / "info.json")
    started = _dt.datetime.now()
    rows: list[dict] = []
    t_start = time.perf_counter()

    for i, idx in enumerate(idxs):
        s = ds[idx]
        video = s["video"]                       # [C,T,H,W] in [-1,1]
        gt_action = s["action"]                  # [T,D] 归一化
        proprio = s["proprio"]                   # [T,D] 归一化
        first_frame = video[:, 0].unsqueeze(0).to(model.device, model.torch_dtype)

        kw = dict(
            input_image=first_frame,
            action_horizon=int(gt_action.shape[0]),
            proprio=proprio[0].to(model.device, model.torch_dtype),
            num_inference_steps=int(inf.num_inference_steps),
            sigma_shift=inf.sigma_shift,
            seed=int(cfg.sampling.seed),
            tiled=False,
            prompt=None,
            context=s["context"].to(model.device, model.torch_dtype),
            context_mask=s["context_mask"].to(model.device),
        )

        t0 = time.perf_counter()
        if inf.mode == "joint":
            # infer_joint 的参数名是 num_video_frames（必填），不是 num_frames；
            # infer_action 没有这个参数，所以只在 joint 分支加。
            kw["num_video_frames"] = int(video.shape[1])
            kw["test_action_with_infer_action"] = bool(inf.action_crosscheck)
            pred = model.infer_joint(**kw)
        else:
            pred = model.infer_action(**kw)
        infer_s = time.perf_counter() - t0

        pred_action = pred["action"].detach().to("cpu", torch.float32)   # [T,D] 归一化

        # --- 反归一化到物理量（与 trainer 一致）---
        pred_phys = denorm_action(processor, pred_action, proprio)
        gt_phys = denorm_action(processor, gt_action, proprio)
        diff = (pred_phys - gt_phys).numpy()

        ep_id, frame_in_ep, ep_len = epi.locate(idx)
        pad_frac = float(s["action_is_pad"].float().mean().item()) if "action_is_pad" in s else 0.0

        rec = {
            "idx": int(idx),
            "episode_index": int(ep_id),
            "frame_in_episode": int(frame_in_ep),
            "pad_frac": pad_frac,
            "infer_s": infer_s,
            # 与 trainer 的 eval/action_l2 同定义（MSE，物理量）
            "action_l2": float((diff ** 2).mean()),
            "action_l1": float(np.abs(diff).mean()),
            "action_rmse": float(np.sqrt((diff ** 2).mean())),
            "action_max_abs": float(np.abs(diff).max()),
        }

        npz = {
            "pred_action_norm": pred_action.numpy(),
            "gt_action_norm": gt_action.numpy(),
            "pred_action_phys": pred_phys.numpy(),
            "gt_action_phys": gt_phys.numpy(),
            "proprio_norm": proprio.numpy(),
            "abs_err_per_dim": np.abs(diff).mean(axis=0),      # [D]
            "abs_err_per_step": np.abs(diff).mean(axis=1),     # [T] 误差随预测步数
        }

        # --- 视频质量 ---
        if inf.mode == "joint":
            pred_v = pil_frames_to_video_tensor(pred["video"])
            gt_v = ((video.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).contiguous()
            with torch.no_grad():
                gt_batch = video.unsqueeze(0).to(model.device, model.torch_dtype)
                vae_v = pil_frames_to_video_tensor(
                    model._decode_latents(model._encode_video_latents(gt_batch, tiled=False),
                                          tiled=False))
            rec.update(
                psnr_rg=float(video_psnr(pred=pred_v, target=gt_v)),
                ssim_rg=float(video_ssim(pred=pred_v, target=gt_v)),
                psnr_rd=float(video_psnr(pred=pred_v, target=vae_v)),
                ssim_rd=float(video_ssim(pred=pred_v, target=vae_v)),
                psnr_dg=float(video_psnr(pred=vae_v, target=gt_v)),
                ssim_dg=float(video_ssim(pred=vae_v, target=gt_v)),
            )
            if inf.save_video:
                st = torch.cat([pred_v, vae_v, gt_v], dim=2)   # 竖向拼 pred/VAE/GT
                frames = [Image.fromarray(
                    (st[:, t].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
                    for t in range(st.shape[1])]
                # fps 用真实播放速率（9 帧跨 32/30 s -> 7.5 fps），不是 trainer 的 8
                save_mp4(frames, str(out / "videos" / f"sample_{i:04d}.mp4"),
                         fps=float(ann.video_fps))

        np.savez_compressed(out / "samples" / f"sample_{i:04d}.npz", **npz)
        rows.append(rec)

        if (i + 1) % 5 == 0 or i == len(idxs) - 1:
            el = time.perf_counter() - t_start
            eta = el / (i + 1) * (len(idxs) - i - 1)
            print(f"    [{tag}] [{i+1}/{len(idxs)}] l2={rec['action_l2']:.4f} "
                  f"rmse={rec['action_rmse']:.4f} {infer_s:.2f}s/样本 ETA {eta/60:.1f}min")

    # ---------------- 汇总 ----------------
    keys = sorted({k for r in rows for k in r})
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def agg(k):
        v = [r[k] for r in rows if k in r]
        if not v:
            return None
        a = np.array(v, dtype=float)
        n = a.size
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if n > 1 else 0.0),
                "sem": float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "min": float(a.min()), "max": float(a.max()),
                "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                "n": int(n)}

    per_dim = np.stack([np.load(out / "samples" / f"sample_{i:04d}.npz")["abs_err_per_dim"]
                        for i in range(len(rows))])
    per_step = np.stack([np.load(out / "samples" / f"sample_{i:04d}.npz")["abs_err_per_step"]
                         for i in range(len(rows))])

    step = ckpt_step(ckpt_path)
    spe = ann.steps_per_epoch
    horizon = int(per_step.shape[1])
    summary = {
        "ckpt": str(ckpt_path),
        "ckpt_tag": tag,
        "step": step,
        "epoch": (step / float(spe)) if (step is not None and spe) else None,
        "weight_check": weight_info,
        "task": cfg.task,
        "split": cfg.dataset.split,
        "dataset_dir": str(ds.lerobot_dataset.dataset_dirs[0]),
        "episodes": sorted({int(r["episode_index"]) for r in rows}),
        "num_episodes_in_split": len(epi),
        "mode": inf.mode,
        "num_samples": len(rows),
        "num_inference_steps": int(inf.num_inference_steps),
        "sigma_shift": inf.sigma_shift,
        "action_crosscheck": bool(inf.action_crosscheck),
        "dataset_total": len(ds),
        "action_dim_names": dim_names,
        "num_padded_samples": int(sum(1 for r in rows if r["pad_frac"] > 0)),
        # 时间标注：action 步 <-> 秒，以及本次评测的时间戳
        "annotate": {
            "fps": float(ann.fps),
            "video_fps": float(ann.video_fps),
            "action_dt_s": 1.0 / float(ann.fps),
            "horizon_steps": horizon,
            "horizon_s": horizon / float(ann.fps),
            "steps_per_epoch": int(spe) if spe else None,
        },
        "eval_started_at": started.isoformat(timespec="seconds"),
        "eval_finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "eval_wall_s": round(time.perf_counter() - t_start, 1),
        "metrics": {k: agg(k) for k in keys
                    if k not in ("idx", "episode_index", "frame_in_episode")},
        "abs_err_per_dim_mean": per_dim.mean(0).tolist(),
        "abs_err_per_step_mean": per_step.mean(0).tolist(),
        "abs_err_per_step_p90": np.percentile(per_step, 90, axis=0).tolist(),
        "sample_idxs": [int(r["idx"]) for r in rows],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    m = summary["metrics"]
    print(f"    [{tag}] 完成 {len(rows)} 样本 / {summary['eval_wall_s']/60:.1f} 分钟  "
          f"action_l2={m['action_l2']['mean']:.4f} rmse={m['action_rmse']['mean']:.4f}"
          + (f" psnr_rd={m['psnr_rd']['mean']:.2f}" if "psnr_rd" in m else ""))
    return summary


# ---------------------------------------------------------------- worker
def build_dataset_and_model(cfg: DictConfig, device: str):
    ov = [f"task={cfg.task}"] + list(cfg.runtime.overrides or [])
    with initialize(config_path="../configs", version_base="1.3"):
        hcfg = compose(config_name="train", overrides=ov)

    split = cfg.dataset.split
    ds_kwargs = {"pretrained_norm_stats": str(cfg.dataset.norm_stats)}
    if cfg.dataset.dataset_dir:
        ds_kwargs["dataset_dirs"] = [str(cfg.dataset.dataset_dir)]
    eps = cfg.dataset.episodes
    if eps == "all":
        # val_set_proportion<1e-6 走的就是"全部 episode"那条既有分支
        ds_kwargs["val_set_proportion"] = 0.0
    elif eps is not None:
        ds_kwargs["episodes"] = [int(e) for e in eps]

    print(f">>> 构建数据集 (split={split}) {ds_kwargs.get('dataset_dirs', '')} ...")
    ds = instantiate(hcfg.data[split], **ds_kwargs)
    processor = ds.lerobot_dataset.processor

    print(f">>> 构建模型 (task={cfg.task}) ...")
    t0 = time.perf_counter()
    model = instantiate(hcfg.model, model_dtype=torch.bfloat16, device=device)
    print(f"    模型就绪，耗时 {time.perf_counter() - t0:.1f}s")
    return ds, processor, model, hcfg


def fill_annotate(cfg: DictConfig, info_path: Path, num_frames: int, freq_ratio: int) -> None:
    """annotate 里没写的项从数据集 meta 推出来，让"时间标注"永远有值。"""
    if cfg.annotate.fps is None:
        fps = dataset_fps(info_path)
        if fps is None:
            raise SystemExit(f"读不到 fps（{info_path}），请在 config 里写 annotate.fps")
        cfg.annotate.fps = fps
    if cfg.annotate.video_fps is None:
        # 视频帧是每 freq_ratio 帧抽 1 帧，真实播放速率就是 fps/freq_ratio
        cfg.annotate.video_fps = float(cfg.annotate.fps) / max(1, int(freq_ratio))


def run_worker(cfg: DictConfig, ckpts: list[Path], nested: bool) -> int:
    # launcher 传下来的线程上限（单卡直跑时不设，用满整机）
    nt = os.environ.get("FASTWAM_EVAL_THREADS")
    if nt:
        torch.set_num_threads(int(nt))
        print(f">>> CPU 线程限为 {nt}")

    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(out_root))

    dcfg = _task_data_cfg(cfg)
    dataset_dir = Path(cfg.dataset.dataset_dir or dcfg["dataset_dirs"][0])
    fill_annotate(cfg, dataset_dir / "meta" / "info.json",
                  dcfg["num_frames"], dcfg["action_video_freq_ratio"])

    ds, processor, model, _ = build_dataset_and_model(cfg, "cuda")
    epi = EpisodeIndex(ds, dcfg["num_frames"])
    idxs, excluded = pick_indices(ds, epi, cfg.sampling)
    print(f">>> split 共 {len(ds):,} 帧 / {len(epi)} 个 episode {epi.ids}")
    print(f">>> 本次评测 {len(idxs)} 个样本"
          + (f"（skip_padded 排除了 {excluded:,} 个候选）" if excluded else ""))

    failed = []
    for p in ckpts:
        out = (out_root / "ckpt" / ckpt_tag(p)) if nested else out_root
        try:
            eval_one_ckpt(model, ds, processor, idxs, epi, cfg, p, out)
        except SystemExit:
            raise
        except Exception as e:                       # 一个 ckpt 挂了不拖累其余
            import traceback
            print(f"    ❌ [{ckpt_tag(p)}] 失败: {e}")
            traceback.print_exc()
            failed.append(ckpt_tag(p))
    if failed:
        print(f">>> 有 {len(failed)} 个 ckpt 失败: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------- launcher
def run_launcher(cfg: DictConfig, ckpts: list[Path], todo: list[Path] | None = None) -> int:
    """按 ckpt 轮转分片到各卡，每卡一个子进程；跑完合并。

    样本下标只由 (split, episodes, num_samples, per_episode, stride, seed) 决定，
    所以分片**不影响可比性** —— 每个 worker 算出来的 idxs 完全一样。

    `todo` 是本次真正要推理的子集（--skip-done 用）；合并时仍按完整的 `ckpts` 走。
    """
    out_root = Path(cfg.out_dir)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    resolved = out_root / "config.resolved.yaml"
    OmegaConf.save(cfg, resolved)

    run_list = ckpts if todo is None else todo
    gpus = [int(g) for g in cfg.runtime.gpus]
    shards: list[list[Path]] = [[] for _ in gpus]
    for i, p in enumerate(run_list):
        shards[i % len(gpus)].append(p)                # 轮转：各卡负载均衡

    procs = []
    # 每个 worker 只能用 总核数/卡数 个线程。不限的话 8 个进程各起 80 个 torch 线程
    # （80 核机器 -> load average 260+），实测每样本从 1.36s 掉到 16s，慢 12 倍。
    # 瓶颈是 CPU 侧：VAE 解码后的 PIL 转换、PSNR/SSIM、h264 编码都在 CPU 上。
    nthreads = max(1, (os.cpu_count() or 8) // max(1, len(gpus)))
    print(f">>> 每个 worker 限 {nthreads} 个 CPU 线程（共 {os.cpu_count()} 核 / {len(gpus)} 卡）")
    for g, shard in zip(gpus, shards):
        if not shard:
            continue
        log = out_root / "logs" / f"gpu{g}.log"
        # PYTHONUNBUFFERED：子进程 stdout 是文件时 Python 会按块缓冲，日志会一直是空的，
        # 那这些 log 就失去了"看进度"的意义（实测踩到）
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS=str(nthreads), MKL_NUM_THREADS=str(nthreads),
                   FASTWAM_EVAL_THREADS=str(nthreads))
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--config", str(resolved), "--ckpt", *[str(p) for p in shard]]
        fh = log.open("w")
        procs.append((g, shard, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                                 cwd=str(REPO_ROOT)), fh, log))
        print(f">>> GPU {g}: {len(shard)} 个 ckpt -> {log}", flush=True)

    print(f">>> {len(procs)} 个分片已启动，等待完成（进度看各自的 log）...", flush=True)
    bad = []
    for g, shard, proc, fh, log in procs:
        rc = proc.wait()
        fh.close()
        status = "ok" if rc == 0 else f"rc={rc}"
        print(f"    GPU {g} 结束 ({status})  {log}", flush=True)
        if rc != 0:
            bad.append((g, log))

    merged = merge_results(cfg, ckpts)
    if bad:
        # 沿用 run_final.sh 的作风：失败只报告，已成功的部分照样合并可用
        print(f"\n⚠️ {len(bad)} 个分片非零退出，看日志: "
              + ", ".join(str(l) for _, l in bad))
    return 0 if merged and not bad else 1


def merge_results(cfg: DictConfig, ckpts: list[Path]) -> int:
    """把各 ckpt 的 summary.json 汇成 compare.csv + sweep.json。"""
    out_root = Path(cfg.out_dir)
    got, absent = [], []
    for p in ckpts:
        f = out_root / "ckpt" / ckpt_tag(p) / "summary.json"
        (got if f.is_file() else absent).append((p, f))
    if not got:
        print("!! 没有任何 ckpt 产出 summary.json，无法合并")
        return 0

    summaries = [json.loads(f.read_text()) for _, f in got]
    summaries.sort(key=lambda s: (s.get("step") is None, s.get("step") or 0))

    metric_keys = [k for k in ("action_rmse", "action_l2", "action_l1", "action_max_abs",
                               "psnr_rd", "ssim_rd", "psnr_rg", "ssim_rg",
                               "psnr_dg", "ssim_dg", "infer_s")
                   if any(k in s["metrics"] for s in summaries)]
    rows = []
    for s in summaries:
        r = {"ckpt_tag": s["ckpt_tag"], "step": s.get("step"), "epoch": s.get("epoch"),
             "num_samples": s["num_samples"]}
        for k in metric_keys:
            a = s["metrics"].get(k)
            r[f"{k}_mean"] = a["mean"] if a else ""
            r[f"{k}_sem"] = a["sem"] if a else ""
            r[f"{k}_p90"] = a["p90"] if a else ""
        rows.append(r)

    cmp_csv = out_root / "compare.csv"
    with cmp_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # reference / highlight 可能指向没跑成的 ckpt（分片失败或人为中断都很常见），
    # 那就退到实际存在的上面 —— 否则出图阶段会静默少画一张。
    tags = [s["ckpt_tag"] for s in summaries]
    ref = cfg.get("reference")
    if ref not in tags:
        if ref:
            print(f"    ⚠️ reference={ref} 没有结果，改用 {tags[-1]}")
        ref = tags[-1]
    hi = [t for t in (cfg.get("highlight") or []) if t in tags]
    if len(hi) < 2 and len(tags) >= 2:
        # 首 / 中 / 末各一个：跨度最大，最看得出训练带来的变化
        auto = sorted({tags[0], tags[len(tags) // 2], tags[-1]}, key=tags.index)
        print(f"    ⚠️ highlight 里可用的不足 2 个，自动改用 {auto}")
        hi = auto
    sweep = {
        "out_dir": str(out_root),
        "task": cfg.task,
        "split": cfg.dataset.split,
        "dataset_dir": summaries[0]["dataset_dir"],
        "episodes": summaries[0]["episodes"],
        "mode": summaries[0]["mode"],
        "num_samples": summaries[0]["num_samples"],
        "num_inference_steps": summaries[0]["num_inference_steps"],
        "annotate": summaries[0]["annotate"],
        "sample_idxs": summaries[0]["sample_idxs"],
        "reference": ref,
        "highlight": hi,
        "metric_keys": metric_keys,
        "merged_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "missing": [ckpt_tag(p) for p, _ in absent],
        "ckpts": [{"tag": s["ckpt_tag"], "step": s.get("step"), "epoch": s.get("epoch"),
                   "dir": f"ckpt/{s['ckpt_tag']}",
                   "metrics": s["metrics"]} for s in summaries],
    }
    (out_root / "sweep.json").write_text(json.dumps(sweep, indent=2, ensure_ascii=False))

    # 所有 ckpt 必须评的是同一批样本，否则整个对比失去意义
    ref_idxs = summaries[0]["sample_idxs"]
    mismatch = [s["ckpt_tag"] for s in summaries if s["sample_idxs"] != ref_idxs]
    if mismatch:
        print(f"!! 警告：以下 ckpt 的样本下标与其他不同，对比不可信: {mismatch}")

    print(f"\n>>> 合并完成 {len(summaries)}/{len(ckpts)} 个 ckpt")
    if absent:
        print(f"    缺失: {[ckpt_tag(p) for p, _ in absent]}")
    best = min(summaries, key=lambda s: s["metrics"]["action_rmse"]["mean"])
    print(f"    action_rmse 最优: {best['ckpt_tag']} = "
          f"{best['metrics']['action_rmse']['mean']:.4f}")
    print(f"    {cmp_csv}\n    出图: python scripts/visualize_eval.py --sweep {out_root}")
    return len(summaries)


# ---------------------------------------------------------------- 主入口
def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="FastWAM 离线批量评测（config 驱动，支持多 ckpt / 多卡分片）")
    ap.add_argument("--config", help="eval YAML，见 configs/eval/*.yaml")
    ap.add_argument("--worker", action="store_true",
                    help="内部用：作为单卡 worker 运行（由 launcher 起）")
    ap.add_argument("--merge-only", action="store_true",
                    help="不推理，只把已有的 ckpt/*/summary.json 重新合并")
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过已有 summary.json 的 ckpt（中断续跑用）；合并时仍算全部")

    ap.add_argument("--ckpt", nargs="*", help="weights/step_XXXXXX.pt，可给多个")
    ap.add_argument("--task", help="决定模型结构与数据配置，必须与训练时一致")
    ap.add_argument("--out-dir")
    ap.add_argument("--split", choices=["val", "train"])
    ap.add_argument("--dataset-dir", help="换一份 LeRobot 数据集目录")
    ap.add_argument("--episodes", help="'36,97' 或 'all'；不给则沿用 val_set_proportion 划分")
    ap.add_argument("--norm-stats", help="dataset_stats.json，必须与训练时同一份")
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--per-episode", type=int, help="每个 episode 等间隔取 N 个")
    ap.add_argument("--stride", type=int)
    ap.add_argument("--skip-padded", action="store_true", help="跳过 episode 尾部的 padding 窗口")
    ap.add_argument("--mode", choices=["joint", "action"])
    ap.add_argument("--num-inference-steps", type=int)
    ap.add_argument("--sigma-shift", type=float)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gpus", help="'0-7' / '0,2,5'；多于 1 张且多于 1 个 ckpt 时自动分片")
    ap.add_argument("--overrides", nargs="*", help="额外 hydra 覆盖")
    args = ap.parse_args()

    cfg = load_config(args)
    ckpts = resolve_checkpoints(cfg.checkpoints)
    if not ckpts:
        raise SystemExit("解析后 checkpoint 列表为空")

    # 布局：--config 一律用 ckpt/<tag>/ 子目录；旧式单 ckpt 直接写 out-dir（兼容 run_final.sh）
    nested = bool(args.config) or len(ckpts) > 1

    if args.merge_only:
        return 0 if merge_results(cfg, ckpts) else 1

    if args.worker:
        return run_worker(cfg, ckpts, nested=True)

    dcfg = _task_data_cfg(cfg)
    preflight(cfg, ckpts, Path(cfg.dataset.dataset_dir or dcfg["dataset_dirs"][0]))

    todo = ckpts
    if args.skip_done:
        root = Path(cfg.out_dir) / "ckpt"
        todo = [p for p in ckpts if not (root / ckpt_tag(p) / "summary.json").is_file()]
        done = len(ckpts) - len(todo)
        print(f">>> --skip-done: 已有结果 {done} 个，本次跑 {len(todo)} 个")
        if not todo:
            print(">>> 没有需要跑的 ckpt，直接合并")
            return 0 if merge_results(cfg, ckpts) else 1

    gpus = [int(g) for g in cfg.runtime.gpus]
    if len(gpus) > 1 and len(todo) > 1:
        print(f">>> {len(todo)} 个 ckpt 分片到 {len(gpus)} 张卡 {gpus}")
        return run_launcher(cfg, ckpts, todo=todo)

    if len(gpus) == 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    rc = run_worker(cfg, todo, nested=nested)
    if nested:
        merge_results(cfg, ckpts)
    else:
        print(f"\n    结果目录: {cfg.out_dir}")
        print(f"    出图:     python scripts/visualize_eval.py {cfg.out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
