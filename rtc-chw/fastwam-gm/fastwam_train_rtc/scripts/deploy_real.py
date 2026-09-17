#!/usr/bin/env python
"""FastWAM Agilex 双臂真机部署推理。

对端是 openpi 仓库的 `RobotArmService`(跑在机器人 PC 上,封装 ROS2 观测 + Piper SDK),
本脚本跑在 GPU 机上:取观测 -> 拼图 -> `infer_action` -> 反归一化 -> 安全层 -> 下发 14 维绝对指令。

    source env.sh

    # 0) 无硬件:验证部署侧预处理与训练**逐元素一致**(最先跑这个)
    python scripts/deploy_real.py selfcheck --config configs/deploy/agilex_real.yaml

    # 1) 无硬件:实测推理延迟,据此定 replan_steps
    python scripts/deploy_real.py benchmark --config configs/deploy/agilex_real.yaml

    # 2) 真机:先小步试(2 秒),确认方向对了再放开
    python scripts/deploy_real.py run --config configs/deploy/agilex_real.yaml \
        --endpoint tcp://192.168.1.10:9900 --max-steps 60

    # 3) 真机:正式 rollout
    python scripts/deploy_real.py run --config configs/deploy/agilex_real.yaml \
        --endpoint tcp://192.168.1.10:9900 --num-episodes 5 --record-dir runs/deploy/$(date +%m%d_%H%M)

时间尺度(fps=30,决定了所有控制参数)
--------------------------------------
1 个动作步 = 1/30 s = **33.3 ms**;action_horizon = 32 步 = 1.067 s。

本机实测(A800,bf16,`benchmark` 子命令,分阶段线程 pre=16/infer=80):

    num_inference_steps=10   推理 413 ms + 预处理 4 ms = **416 ms** = 12.5 个控制周期
    num_inference_steps= 5   推理 247 ms + 预处理 4 ms = **250 ms** =  7.5 个控制周期

⚠️ `OFFLINE_EVAL_DESIGN.md` 里的 0.42 s 是**只算 infer_action、不含预处理**的数
(离线评测的拼图在 dataloader worker 里,没计入)。部署必须把预处理算进预算。
线程配置影响很大,详见 `inference.torch_threads_*` 的注释;**换机器要重跑 benchmark**。

推理仍然比我们想执行的 chunk 长,这是本脚本全部设计张力的来源:

  sync (默认)  执行 replan_steps 步 -> 驻留 -> 取新观测 -> 阻塞推理 -> 下一 chunk
               只用 chunk 前 10 步(误差 1.4~3.5 度);replan=10/settle=3 时占空比 39%
               (10 步去噪)或 49%(5 步),动作一顿一顿
  async        后台推理,算完按**实际经过步数**对齐拼接,连续 30Hz
               但只能从 chunk 第 13 步(10 步去噪)或第 8 步(5 步)接入

误差随 chunk 步号线性增长(离线实测:首步 1.37 度 -> 末步 8.55 度,
见 eval_offline/*/figs/14_angle_error_deg.png),所以默认 sync。
若离线复测确认 5 步去噪精度不掉,async + 5 步(第 8 步接入,误差约 3.0 度)是更好的终态。

⚠️ 六个必须对齐的契约(错了大多**不报错**,只是动作全乱)
--------------------------------------------------------
1. **两套 14 维排布不同**,别混:
       proprio = [12 关节(左 j0-5, 右 j0-5), 2 夹爪(左, 右)]
       action  = [左 j0-5, **左夹爪**, 右 j0-5, **右夹爪**]     <- 夹爪插在中间
   service 的 step 用的正是 action 排布(`robot_arm_service.py:467-468`)。
2. **夹爪单位**:训练数据是 **0~1 归一化**(满开 0.07 m);service 两侧都用 **米**。
   换算 `frac = m / 0.07`,见 `units.gripper_travel_m`。
3. **相机改名 + 顺序**:service 给 `cam_top`,训练要 `cam_high`;且顺序必须是
   cam_high -> cam_left_wrist -> cam_right_wrist(拼图位置硬编码在
   `robot_video_dataset.py:170-194`,上 256x320 = 顶部,下 128x320 = 双腕并排)。
4. **JPEG 不翻通道**:service 用 cv2 编码出的 JPEG 颜色本身是对的,PIL 解出来直接是 RGB。
   详见 `deploy_robot_client.py` 顶部。
5. **归一化统计量**必须是训练那一份(`runs/_shared/dataset_stats.json`)。
6. **权重要真加载上**:`load_state_dict(strict=False)`(`fastwam.py:1214`)键名不匹配会被静默
   忽略,跑出来是随机初始化的分数还看着挺"正常"。本模型应为 1649/1649,不匹配直接退出。

⚠️ sync 模式必须重取观测
------------------------
`step` 返回的 obs 是"刚下发指令、手臂还没走到"时刻采的 —— 这与训练语义一致(实测状态 ->
未来 32 步指令)。但推理阻塞的 0.4 s 里手臂还在继续走向上一个目标,所以**规划前要重发保持
指令再取一次新 obs**(`control.settle_steps`,默认 3 步 = 100 ms),否则拿的是过期状态。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

# 契约(排布/单位/预处理/安全层/节拍)的**唯一真值来源**,与 fastwam_server.py /
# fastwam_client.py 共用同一份实现 —— 这些常量错了大多不报错、只是动作全乱,
# 所以绝不能存在第二份拷贝。详见 src/fastwam/deploy/。
from fastwam.deploy.control import Pacer, SafetyFilter as _SharedSafetyFilter
from fastwam.deploy.layout import (
    ACT_GRIP_IDX,
    ACT_JOINT_IDX,
    DEFAULT_INSTRUCTION,
    PROP_GRIP_SLICE,
    PROP_JOINT_SLICE,
    grip_frac_to_m,
    grip_m_to_frac,
    obs_to_physical,
    to_action_layout,
    to_service_action,
)
from fastwam.deploy.policy import FastWAMPolicy, ObsPipeline, denorm_action

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 deploy_robot_client 可导入
from deploy_robot_client import RobotArmClient  # noqa: E402

register_default_resolvers()

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "task": "agilex_final_3cam_384_1e-4",
    "checkpoint": None,
    "norm_stats": "./runs/_shared/dataset_stats.json",
    "instruction": DEFAULT_INSTRUCTION,
    "robot": {
        "endpoint": "tcp://127.0.0.1:9900",
        "recv_timeout_s": 30.0,
        "go_zero_on_exit": False,  # True 会让双臂打到全零位,可能扫过工作区,默认关
        # 训练名 -> service 名
        "cameras": {
            "cam_high": "cam_top",
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
        },
    },
    "units": {
        # 夹爪满开度(米)。训练数据是 0~1 归一化,service 两侧用米。已确认 = 0.07。
        "gripper_travel_m": 0.07,
    },
    "inference": {
        "device": "cuda",
        "num_inference_steps": 10,  # 与离线评测一致;改了要先用 offline_eval 复测精度
        "sigma_shift": None,
        "seed": None,  # None = 每次重新采噪声;给定值则每次 replan 用同一份噪声
        "action_horizon": None,  # None -> num_frames - 1 = 32
        "compile_action_infer": False,
        # torch.compile 的 mode。**异步部署必须用 "default"**：
        #   reduce-overhead / max-autotune 会启用 Inductor 的 CUDA Graph Trees，而它的状态存在
        #   `threading.local()` 里、只在 import 那个线程初始化（cudagraph_trees.py:277-292 是顶层
        #   语句）。`run_async` 把推理丢进 ThreadPoolExecutor，工作线程上查不到那个 TLS key，
        #   直接 `AssertionError`（torch/_inductor/cudagraph_trees.py:325）。
        #   sync 模式 / dryrun / offline_eval 都在主线程，所以以前从没暴露过。
        # 代价：CUDA Graphs 是本路径加速的主要来源（32 个 action token，瓶颈在 kernel launch
        #   而非算力，A800 实测 413 -> 131 ms），换成 "default" 只剩算子融合，收益小得多。
        "compile_mode": "reduce-overhead",
        # torch CPU 线程数,**两个阶段要的值相反**(实测,80 核 A800 机,10 步去噪):
        #   预处理(3 相机 resize + 拼图,全是 CPU 小张量):16 线程 58ms / 80 线程 290ms  <- 少的好
        #   infer_action(去噪循环里的 CPU 侧算子)         :16 线程 564ms / 80 线程 428ms  <- 多的好
        # 合计:定 16 = 622ms,定 80 = 718ms,**分阶段 = 416ms**(预处理降到 4ms)。
        # 0/None = 不限(用满整机)。⚠️ 换机器或负载变了要重跑 benchmark 子命令。
        "torch_threads_pre": 16,
        "torch_threads_infer": 0,
        # 跳过读 Wan2.2 底座 DiT(18.8 GB)与 ActionDiT(2.0 GB)预训练权重。
        # 部署时 checkpoint 的 mot 是 1649/1649 全覆盖,底座读完立刻被覆写 —— 纯浪费。
        # 已实测:开与不开,mot 权重逐比特相同、动作输出 max|Δ|=0,建模还快 17s。
        # 收益:最小部署集从 34.2 GB 降到 13.4 GB(只剩 checkpoint 12 GB + VAE 1.4 GB)。
        # 万一 checkpoint 没能全覆盖,load_and_verify 会因缺键直接退出,不会静默跑随机权重。
        # ⚠️ VAE 不受影响,永远要读(相机图得经它编码成 latent)。
        "skip_base_dit_load": True,
    },
    # ── Training-time RTC 动作前缀条件（arXiv:2512.05964）────────────────────────
    # enabled=false（默认）时**完全不走** RTC 代码路径，与既有非 RTC 部署逐比特一致。
    # 只有 control.mode=async 时才生效：sync 模式每次都从静止重新规划、没有 chunk 重叠，
    # 前缀条件无从谈起（论文的异步执行框架是它的前提）。
    #
    # ⚠️ checkpoint 必须是带 rtc.enabled=true 训练出来的，否则模型没见过前缀条件，
    #    效果会明显变差。脚本启动时会用 model.rtc.enabled 交叉校验并拒绝不匹配的组合。
    "rtc": {
        "enabled": False,
        # 前缀长度 d 怎么定：
        #   measured = 用最近若干次推理的实测耗时换算（论文相对 inference-time RTC 的
        #              主要增益之一就是「d 是运行时输入，不必保守估计」）
        #   fixed    = 固定用 fixed_delay，便于复现和排查
        "delay_mode": "measured",  # measured | fixed
        "fixed_delay": 5,
        # measured 模式：取最近 latency_window 次推理耗时的分位数，向上取整成控制步。
        # 偏大一点更安全：d 估小了会导致切入点落在前缀之外，接缝失去保护。
        "latency_percentile": 0.9,
        "latency_window": 20,
        # d 的上下限。上限必须满足论文的时序约束 d <= H - s，且不超过训练时的 max_delay-1。
        "min_delay": 1,
        "max_delay": 11,
    },
    "control": {
        "mode": "sync",  # sync | async
        "hz": 30.0,  # 必须等于数据集 fps
        "replan_steps": 10,  # sync:每个 chunk 执行多少步。交接文档建议 <= 10
        "settle_steps": 3,  # sync:规划前保持指令驻留几步以取到"已稳定"的观测
        "max_steps": 0,  # 每个 episode 的步数上限,0 = 不限
        "num_episodes": 1,
    },
    "safety": {
        "enabled": True,
        # 下面两个来自训练数据实测的步间动作差分(60 个 episode / 129,762 个转移):
        #   12 关节 p99 = 0.059 rad, p99.9 = 0.097 rad, max = 0.815 rad(离群)
        #   2 夹爪  p99 = 0.124,     p99.9 = 0.198
        # 取 p99.9:放过示教里 99.9% 的真实速度,挡住 46 度的单步跳变。
        "max_delta_joint": 0.10,  # rad / 33.3ms ≈ 172 度/s
        "max_delta_gripper": 0.20,  # 归一化分数 / 33.3ms
        # 绝对限位 = 训练 action 的 global_min/max 再放这个余量
        "limit_margin_joint": 0.10,
        "limit_margin_gripper": 0.05,
        # chunk 首步与实测状态的关节差:超过 warn 只计数告警,超过 abort 直接中止 episode
        "warn_first_step_jump": 0.15,
        "abort_first_step_jump": 0.60,
    },
    "preflight": {
        # 起始位姿分布:从数据集 parquet 的第 0 帧统计,用来判断真机初始位姿是否 OOD。
        # auto = 算并缓存到 runs/_shared/start_pose_stats.json;null = 跳过这项检查
        "start_pose_stats": "auto",
        "start_pose_episodes": 80,
        "dataset_dir": None,  # None -> 用 task 配置里的
        "abort_on_gripper_out_of_range": True,
        "gripper_range_margin": 0.15,
    },
    "record": {
        "dir": None,
        "save_video": True,  # 存模型每次真正看到的 384x320 拼图
    },
    "runtime": {"overrides": []},
}


# ══════════════════════════════════════════════════════════════════ config
def _merge(base: dict, over) -> dict:
    """把 over 深合并到 base 上(不改 base)。over 里的 None 视为"没写",不覆盖。"""
    out = dict(base)
    if not over:
        return out
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def _prune_none(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _prune_none(v)
            if v:
                out[k] = v
        elif v is not None:
            out[k] = v
    return out


def load_config(args) -> DictConfig:
    """DEFAULTS <- YAML <- 命令行,后者优先。与 offline_eval.py 同一套写法。"""
    cfg = dict(DEFAULTS)
    if getattr(args, "config", None):
        raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if not isinstance(raw, dict):
            raise SystemExit(f"{args.config} 顶层必须是 mapping")
        cfg = _merge(cfg, raw)

    cli = {
        "task": getattr(args, "task", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "norm_stats": getattr(args, "norm_stats", None),
        "instruction": getattr(args, "instruction", None),
        "robot": {"endpoint": getattr(args, "endpoint", None)},
        "units": {"gripper_travel_m": getattr(args, "gripper_travel_m", None)},
        "inference": {
            "num_inference_steps": getattr(args, "num_inference_steps", None),
            "device": getattr(args, "device", None),
            "torch_threads_pre": getattr(args, "torch_threads_pre", None),
            "torch_threads_infer": getattr(args, "torch_threads_infer", None),
        },
        "control": {
            "mode": getattr(args, "mode", None),
            "replan_steps": getattr(args, "replan_steps", None),
            "max_steps": getattr(args, "max_steps", None),
            "num_episodes": getattr(args, "num_episodes", None),
            "hz": getattr(args, "hz", None),
        },
        "safety": {"enabled": False if getattr(args, "no_safety", False) else None},
        "record": {
            "dir": getattr(args, "record_dir", None),
            "save_video": False if getattr(args, "no_save_video", False) else None,
        },
        "runtime": {"overrides": getattr(args, "overrides", None) or None},
    }
    cfg = _merge(cfg, _prune_none(cli))
    if not cfg["checkpoint"]:
        raise SystemExit("必须给 --checkpoint 或在 config 里写 checkpoint")
    return OmegaConf.create(cfg)


def task_data_cfg(cfg: DictConfig) -> dict:
    """只取几个数据/时间尺度字段,避免过早 compose 整个 hydra。"""
    with initialize(config_path="../configs", version_base="1.3"):
        c = compose(config_name="train", overrides=[f"task={cfg.task}"])
    node = c.data["val"]
    return {
        "dataset_dirs": [str(d) for d in node.dataset_dirs],
        "num_frames": int(node.num_frames),
        "action_video_freq_ratio": int(node.action_video_freq_ratio),
        "video_size": [int(x) for x in node.video_size],
        "concat_multi_camera": str(node.concat_multi_camera),
        "context_len": int(node.context_len),
        "text_embedding_cache_dir": str(node.text_embedding_cache_dir),
        "shape_meta": OmegaConf.to_container(node.shape_meta, resolve=True),
    }


# ══════════════════════════════════════ 单位/排布/流水线/安全层/节拍器
# 这些实现全部搬到了 `src/fastwam/deploy/`，与 `fastwam_server.py`（GPU 侧）和
# `fastwam_client.py`（控制环侧）共用同一份 —— 见本文件顶部 import 处的说明。
# 本文件保留的是**单进程编排**：selfcheck / benchmark / rtc-check 与 sync/async 控制环。


class SafetyFilter(_SharedSafetyFilter):
    """共享 SafetyFilter 的适配器,保留本脚本原有的 (stats, DictConfig) 构造签名。

    共享版改成吃 numpy 的 lo/hi 数组,是为了让**无 torch** 的 `fastwam_client.py` 也能用
    (限位由 server 在 ping 的 info 里下发)。本脚本手里本来就有完整的 stats,直接取出来即可。
    """

    def __init__(self, stats: dict, s: DictConfig):
        super().__init__(
            action_min=stats["action"]["default"]["global_min"].numpy().astype(np.float32),
            action_max=stats["action"]["default"]["global_max"].numpy().astype(np.float32),
            max_delta_joint=float(s.max_delta_joint),
            max_delta_gripper=float(s.max_delta_gripper),
            limit_margin_joint=float(s.limit_margin_joint),
            limit_margin_gripper=float(s.limit_margin_gripper),
            warn_first_step_jump=float(s.warn_first_step_jump),
            abort_first_step_jump=float(s.abort_first_step_jump),
            enabled=bool(s.enabled),
        )



# ═══════════════════════════════════════════════════════════════════ 落盘
class Recorder:
    """每步指令 + 每次规划的完整 chunk + 模型真正看到的拼图,便于事后离线复盘。"""

    def __init__(self, out_dir: Path | None, save_video: bool, replan_hz: float):
        self.dir = Path(out_dir) if out_dir else None
        self.save_video = bool(save_video)
        self.replan_hz = max(replan_hz, 1e-6)
        self.steps: list[dict] = []
        self.chunks: list[dict] = []
        self.frames: list[Image.Image] = []
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def on(self) -> bool:
        return self.dir is not None

    def add_chunk(self, res: dict, measured14: np.ndarray, filtered: np.ndarray, rep: dict) -> None:
        if not self.on:
            return
        self.chunks.append({
            "action_phys": res["action_phys"],
            "action_norm": res["action_norm"],
            "action_filtered": filtered,
            "proprio_norm": res["proprio_norm"],
            "measured14": measured14,
            "infer_s": res["infer_s"],
            "pre_s": res["pre_s"],
            **{k: rep[k] for k in ("first_step_jump", "delta_clamped", "limit_clamped")},
        })
        if self.save_video:
            self.frames.append(ObsPipeline.frame_to_pil(res["frame"]))

    def add_step(self, **kw) -> None:
        if self.on:
            self.steps.append(kw)

    def flush(self, ep: int, meta: dict) -> None:
        if not self.on or (not self.steps and not self.chunks):
            return
        tag = f"episode_{ep:03d}"
        np.savez_compressed(
            self.dir / f"{tag}.npz",
            **{f"step_{k}": np.asarray([s[k] for s in self.steps])
               for k in (self.steps[0].keys() if self.steps else [])},
            **{f"chunk_{k}": np.asarray([c[k] for c in self.chunks])
               for k in (self.chunks[0].keys() if self.chunks else [])},
        )
        (self.dir / f"{tag}_summary.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str)
        )
        if self.save_video and self.frames:
            save_mp4(self.frames, str(self.dir / f"{tag}_model_view.mp4"),
                     fps=max(int(round(self.replan_hz)), 1))
            print(f"    模型视角 mp4: {self.dir / f'{tag}_model_view.mp4'} "
                  f"({len(self.frames)} 帧 @ {max(int(round(self.replan_hz)), 1)} fps)")
        print(f"    记录已落盘: {self.dir / f'{tag}.npz'}")
        self.steps, self.chunks, self.frames = [], [], []


# ══════════════════════════════════════════════════════════ 起始位姿分布
def start_pose_stats(dataset_dir: Path, limit: int, cache: Path | None) -> dict | None:
    """统计训练数据每个 episode **第 0 帧**的关节/夹爪分布。

    用途:真机 reset 后的初始位姿如果落在这个分布之外,模型的第一帧就是 OOD,
    整条 rollout 都会被污染。openpi service 的默认初始位姿实测在左腕滚转与左夹爪上越界。
    """
    if cache and cache.is_file():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    files = sorted(glob.glob(str(dataset_dir / "data" / "chunk-*" / "episode_*.parquet")))[: max(limit, 1)]
    if not files:
        return None
    try:
        import pandas as pd
    except ImportError:
        print("    ⚠️ 没有 pandas,跳过起始位姿分布检查")
        return None
    joints, grips = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["observation.state.joint", "observation.state.gripper_position"])
        joints.append(np.asarray(df["observation.state.joint"].iloc[0], dtype=np.float32))
        grips.append(np.asarray(df["observation.state.gripper_position"].iloc[0], dtype=np.float32))
    j, g = np.stack(joints), np.stack(grips)
    out = {
        "n_episodes": len(files),
        "joint": {"mean": j.mean(0).tolist(), "std": j.std(0).tolist(),
                  "min": j.min(0).tolist(), "max": j.max(0).tolist()},
        "gripper": {"mean": g.mean(0).tolist(), "std": g.std(0).tolist(),
                    "min": g.min(0).tolist(), "max": g.max(0).tolist()},
    }
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=2))
    return out


# ══════════════════════════════════════════════════════════════════ preflight
def preflight_obs(policy: FastWAMPolicy, obs: dict, cfg: DictConfig, sp: dict | None) -> None:
    """拿到第一帧真实观测后的全部一致性检查。用错单位/改名不会报错,只会动作全乱,所以这里查死。"""
    print("\n>>> preflight(真实观测)")
    travel = float(cfg.units.gripper_travel_m)
    joint, grip = obs_to_physical(obs, travel)

    # 1) 相机
    for meta in policy.pipe.image_meta:
        src = policy.pipe.cam_alias.get(meta["key"], meta["key"])
        arr = obs["images"][src]
        print(f"    相机 {meta['key']:16s} <- service '{src}'  {arr.shape} {arr.dtype} 均值 {arr.mean():5.1f}")
        if tuple(arr.shape[:2]) != tuple(meta["raw_shape"][1:]):
            print(f"      ⚠️ 分辨率 {arr.shape[:2]} 与训练 raw_shape {meta['raw_shape'][1:]} 不同 —— "
                  "service 端 --crop-size 是否设成 480x640?")

    # 2) 夹爪单位(唯一"错了不报错"的地方)
    gs = policy.stats["state"]["gripper_position"]
    lo = gs["global_min"].numpy().min() - float(cfg.preflight.gripper_range_margin)
    hi = gs["global_max"].numpy().max() + float(cfg.preflight.gripper_range_margin)
    m = np.asarray(obs["gripper_position"], dtype=np.float32)
    print(f"    夹爪  {np.round(m, 5)} m  /{travel} ->  分数 {np.round(grip, 4)}   "
          f"训练区间 [{gs['global_min'].numpy().min():.3f}, {gs['global_max'].numpy().max():.3f}]")
    if np.any(grip < lo) or np.any(grip > hi):
        msg = (f"夹爪换算后 {np.round(grip, 4)} 落在训练区间 [{lo:.3f}, {hi:.3f}] 之外 —— "
               f"units.gripper_travel_m={travel} 可能不对")
        if bool(cfg.preflight.abort_on_gripper_out_of_range):
            raise SystemExit(f"!! {msg}\n   确认无误可设 preflight.abort_on_gripper_out_of_range=false")
        print(f"      ⚠️ {msg}")

    # 3) proprio z-score(normalizer 会 clamp 到 ±5,越界即信息丢失)
    js, gsm = policy.stats["state"]["joint"], gs
    z_j = (joint - js["global_mean"].numpy()) / (js["global_std"].numpy() + 1e-8)
    z_g = (grip - gsm["global_mean"].numpy()) / (gsm["global_std"].numpy() + 1e-8)
    z = np.concatenate([z_j, z_g])
    print(f"    proprio z-score  max|z| = {np.abs(z).max():.2f}  (>5 会被 clamp)")
    if np.abs(z).max() > 5.0:
        bad = np.where(np.abs(z) > 5.0)[0].tolist()
        print(f"      ⚠️ 维度 {bad} 的 |z|>5,归一化时会被截断 —— 当前状态严重偏离训练分布")

    # 4) 初始位姿是否在训练起始位姿分布内(用户已决定:只告警,不干预)
    if sp:
        sj = sp["joint"]
        zs = (joint - np.asarray(sj["mean"])) / (np.asarray(sj["std"]) + 1e-8)
        out_lo = joint < np.asarray(sj["min"])
        out_hi = joint > np.asarray(sj["max"])
        print(f"    对训练起始位姿({sp['n_episodes']} 个 episode)  max|z| = {np.abs(zs).max():.2f}")
        names = [f"L_j{i}" for i in range(6)] + [f"R_j{i}" for i in range(6)]
        for i in np.where(out_lo | out_hi)[0]:
            print(f"      ⚠️ {names[i]} = {joint[i]:+.3f} 在训练起始区间 "
                  f"[{sj['min'][i]:+.3f}, {sj['max'][i]:+.3f}] 之外 (z={zs[i]:+.2f})")
        gj = sp["gripper"]
        for i, nm in enumerate(["L_grip", "R_grip"]):
            if grip[i] < gj["min"][i] or grip[i] > gj["max"][i]:
                print(f"      ⚠️ {nm} = {grip[i]:.3f} 在训练起始区间 "
                      f"[{gj['min'][i]:.3f}, {gj['max'][i]:.3f}] 之外")
    print()


# ═══════════════════════════════════════════════════════════════ 控制环
class EpisodeRunner:
    def __init__(self, policy: FastWAMPolicy, client: RobotArmClient, cfg: DictConfig,
                 safety: SafetyFilter, rec: Recorder):
        self.policy, self.client, self.cfg = policy, client, cfg
        self.safety, self.rec = safety, rec
        self.travel = float(cfg.units.gripper_travel_m)
        self.pacer = Pacer(float(cfg.control.hz))
        self.replan = max(1, int(cfg.control.replan_steps))
        self.max_steps = int(cfg.control.max_steps)
        self.step_i = 0
        self.chunk_i = 0
        self.n_stall = 0   # async: chunk 用尽只能保持上一条指令的次数
        # 观测到的真实 elapsed（从发起到主循环取到结果，单位=控制步）。
        # 这是预测 d 的**首选依据** —— 比 policy.stat_infer_s 准，因为后者只含模型耗时
        # (infer_s)，不含预处理与工作线程等 GIL 的时间。实测二者差 13~36 ms，
        # 正好导致 d 被系统性低估 1 步、进而 overrun（0903 实测 elapsed=7 vs 预估 d=6）。
        self._elapsed_hist: list[int] = []
        self.last_cmd_m: np.ndarray | None = None

        # ── Training-time RTC 开关 ────────────────────────────────────────────
        # 只在 async 下生效:sync 每次都从静止重新规划、chunk 之间没有重叠,
        # 前缀条件无从谈起(论文的异步执行框架是它的前提)。
        rtc_cfg = cfg.get("rtc", None)
        want_rtc = bool(rtc_cfg is not None and rtc_cfg.get("enabled", False))
        is_async = str(cfg.control.mode) == "async"
        self.rtc_on = want_rtc and is_async
        if want_rtc and not is_async:
            print("!! rtc.enabled=true 但 control.mode=sync —— RTC 前缀条件已忽略。"
                  "sync 模式 chunk 之间没有重叠,用不上前缀;要用请设 control.mode=async")

        # checkpoint 与配置的交叉校验:模型必须真的用前缀条件训练过,否则它没见过
        # per-token timestep 这种输入,结果会明显变差(而且不会报错,只是静默变差)。
        model_rtc = getattr(getattr(policy, "model", None), "rtc", None)
        model_trained_with_rtc = bool(model_rtc is not None and model_rtc.enabled)
        if self.rtc_on and not model_trained_with_rtc:
            raise SystemExit(
                "rtc.enabled=true,但这个 checkpoint 对应的 model.rtc.enabled=false —— "
                "它没有用动作前缀条件训练过。请换成 RTC 训出来的权重"
                "(task=agilex_rtc_3cam_384_1e-4),或把 deploy 配置的 rtc.enabled 设为 false。"
            )
        if is_async and model_trained_with_rtc and not want_rtc:
            print("!! 这个 checkpoint 是带 RTC 训练的,但 deploy 的 rtc.enabled=false —— "
                  "你正在用 naive 拼接跑一个为前缀条件训练过的模型,接缝收益拿不到。")
        # ── compile x 线程模型的兼容性检查 ─────────────────────────────────────
        # CUDA Graph Trees 的状态是线程局部的(cudagraph_trees.py:277-292 + 325 的 assert)。
        # 本机实测(torch 2.7.1+cu128)的**精确规则**是「必须从头到尾同一个线程」,而不是
        # 「必须主线程」：
        #     ✅ 专属线程首次 compile+record、之后一直用它   -> 正常，且最快
        #     ❌ 一个线程先 record，再换另一个线程 record 新图 -> AssertionError
        # 本脚本正好落在 ❌ 上:`warmup_before_episodes` 用 "warmup" 池(第 730 行)，
        # 而 `run_async` 另建一个 "infer" 池(第 740 行) —— 两个不同线程。
        # 所以这个守卫**依然必要**,但结论不再是「async 与 CUDA Graphs 不兼容」：
        # 要 CUDA Graphs 就用 scripts/fastwam_server.py + fastwam_client.py，那边把
        # 编译/预热/每次推理全钉在同一个专属线程上(InferenceThread)，实测可用且快 2.4 倍。
        inf_cfg = cfg.inference
        if (
            is_async
            and bool(inf_cfg.get("compile_action_infer", False))
            and str(inf_cfg.get("compile_mode", "reduce-overhead")) in ("reduce-overhead", "max-autotune")
        ):
            raise SystemExit(
                f"control.mode=async + inference.compile_mode={inf_cfg.get('compile_mode')!r} "
                "在**本单进程脚本**里不可用：该 mode 启用 CUDA Graph Trees，而它的状态是线程"
                "局部的；本脚本的预热池与推理池是两个不同线程，会在第一次后台推理时抛 "
                "AssertionError（torch/_inductor/cudagraph_trees.py:325）——那时机器人已经在动了。\n"
                "  ✅ 想要 CUDA Graphs（RTC 推荐，实测快 2.4 倍）就用拆分部署：\n"
                "       python scripts/fastwam_server.py --config <本配置> "
                "--compile-mode reduce-overhead --warmup-rtc\n"
                "       python scripts/fastwam_client.py --server tcp://<gpu>:8900 "
                "--endpoint tcp://<robot>:9900 --mode async --rtc ...\n"
                "     它把编译/预热/每次推理全钉在同一个专属线程上（InferenceThread）。\n"
                "  或者留在本脚本：inference.compile_mode: default（放弃 CUDA Graphs），"
                "或 compile_action_infer: false，或 control.mode: sync。"
            )

        if self.rtc_on:
            r = cfg.rtc
            # 论文的时序约束 d <= H - s;这里 s 用 replan_steps 近似
            print(f"[rtc] 已启用 | delay_mode={r.delay_mode} "
                  f"fixed={r.fixed_delay} p{float(r.latency_percentile) * 100:.0f} "
                  f"window={r.latency_window} d∈[{r.min_delay},{r.max_delay}]")

    # ── 基本动作 ────────────────────────────────────────────────────────────
    def _measured14(self, obs: dict) -> np.ndarray:
        joint, grip = obs_to_physical(obs, self.travel)
        return to_action_layout(joint, grip)

    def _send(self, a14_phys: np.ndarray, obs_in: dict, chunk_step: int) -> tuple[dict, bool]:
        cmd_m = to_service_action(a14_phys, self.travel)
        self.pacer.wait()
        obs, done, _info = self.client.step(cmd_m)
        self.last_cmd_m = cmd_m
        self.step_i += 1
        self.rec.add_step(
            t=time.monotonic(), step=self.step_i, chunk=self.chunk_i, chunk_step=chunk_step,
            cmd_phys=a14_phys.astype(np.float32), cmd_service=cmd_m,
            measured=self._measured14(obs_in),
        )
        if self.max_steps > 0 and self.step_i >= self.max_steps:
            done = True
        return obs, done

    def _plan(self, obs: dict) -> tuple[np.ndarray, dict, dict]:
        joint, grip = obs_to_physical(obs, self.travel)
        res = self.policy.infer_chunk(joint, grip, obs["images"])
        measured = to_action_layout(joint, grip)
        filtered, rep = self.safety.filter_chunk(res["action_phys"], measured)
        self.chunk_i += 1
        self.rec.add_chunk(res, measured, filtered, rep)
        return filtered, res, rep

    # ── 同步:执行 -> 驻留 -> 重取观测 -> 阻塞推理 ────────────────────────────
    def run_sync(self, obs: dict) -> None:
        self._warmup_inference(obs, pool=None)   # 已预热过则内部直接返回
        settle = max(0, int(self.cfg.control.settle_steps))
        done = False
        while not done:
            # 驻留:用"保持指令"换一帧**已稳定**的观测(见模块 docstring)
            if self.last_cmd_m is not None:
                for _ in range(settle):
                    self.pacer.wait()
                    obs, done, _ = self.client.step(self.last_cmd_m)
                    if done:
                        return

            chunk, res, rep = self._plan(obs)
            # 推理阻塞是**设计如此**的冻结,不是错过控制周期。不 reset 的话
            # 下一步的 pacer 会把这 0.4s 记成一次"超时",节拍统计就没意义了。
            self.pacer.reset()
            n = min(self.replan, chunk.shape[0])
            for k in range(n):
                obs, done = self._send(chunk[k], obs, k)
                if done:
                    break
            if self.chunk_i % 5 == 1 or done:
                print(f"    [sync] chunk {self.chunk_i} step {self.step_i}  "
                      f"infer {res['infer_s'] * 1000:.0f} ms  首步跳变 "
                      f"{np.degrees(rep['first_step_jump']):.1f} 度  "
                      f"钳位 delta/limit {rep['delta_clamped']}/{rep['limit_clamped']}")

    # ── RTC:预估下一次推理的延迟 d ──────────────────────────────────────────
    def _predict_delay(self) -> int:
        """预估这次推理会花掉多少个控制步,作为前缀长度 d。

        必须在**发起推理时**就定 d(前缀要作为输入喂进去),而真实 elapsed 只有算完才知道,
        所以这里用最近若干次实测耗时的分位数外推。论文相对 inference-time RTC 的主要增益
        之一正是「d 是运行时输入,不必保守估计」——所以用实测而不是一个固定的保守常数。

        取偏大一点更安全:d 估小了,切入点会落在前缀之外,那几步就失去了连续性保护。
        """
        r = self.cfg.rtc
        w = int(r.latency_window)
        pct = float(r.latency_percentile)

        def _nearest_rank(xs):
            ordered = sorted(xs)
            i = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
            return ordered[i]

        if str(r.delay_mode) == "fixed":
            d = int(r.fixed_delay)
        elif len(self._elapsed_hist) >= 3:
            # 首选：观测到的真实 elapsed（已是控制步，自带预处理/GIL/控制环速率的影响）
            d = int(_nearest_rank(self._elapsed_hist[-w:]))
        elif self.policy.stat_infer_s:
            # 次选：只有模型耗时可用（前几个 chunk）。它偏小，所以额外 +1 步兜住
            # 预处理与 GIL 等待，避免开局就 overrun。
            d = math.ceil(_nearest_rank(self.policy.stat_infer_s[-w:]) * float(self.cfg.control.hz)) + 1
        else:
            d = int(r.fixed_delay)  # 冷启动，什么都还没测到
        return int(max(int(r.min_delay), min(int(r.max_delay), d)))

    def _warmup_inference(self, obs: dict, pool=None, n: int = 2) -> None:
        """在控制环**之外**先跑几次推理，把 torch.compile 的编译成本付掉。

        为什么必须这样：`compile_action_infer=true` 时第一次调用会触发 Inductor 编译，
        实测在真机上要 **35 秒**。如果这发生在控制环里，机器人会连续几百步拿不到新 chunk
        （只能保持上一条指令），等新 chunk 到达时它已经基于 30 秒前的观测，首步跳变可达
        35 度直接触发 abort —— 这正是 0903 那次 run 的失败原因。
        `benchmark` 丢弃前 2 次迭代所以从没暴露过。

        `pool` 非空时通过它跑，让**工作线程**成为编译发生的线程（async 下推理就在那儿跑）。
        """
        if not bool(self.cfg.inference.compile_action_infer):
            return
        # 幂等：cmd_run 已在 episode 计时之外预热过就不再重复
        if getattr(self.policy, "_warmed_up", False):
            return
        joint, grip = obs_to_physical(obs, self.travel)
        imgs = {k: v.copy() for k, v in obs["images"].items()}

        # ⚠️ **必须把两张图都预热**：RTC 下 delay=0 走 1-D timestep、delay>0 走 2-D，
        #    是两个不同的计算图，各要编译一次（各约 17 s）。只热 delay=0 的话，控制环里
        #    第一次带前缀发起时会在**实时环内**触发第二次编译 —— 表现为「发起 1 次 / 落地
        #    0 次」、chunk 走完后一路停顿（0903 实测）。
        #    delay 的**数值**不影响图（timestep 形状恒为 [1, H]），所以任意一个 d>0 即可。
        plans: list[tuple[str, object, int]] = [("delay=0 (1-D timestep)", None, 0)] * n
        if self.rtc_on:
            d_probe = max(1, int(self.cfg.rtc.fixed_delay))
            probe_prefix = np.zeros((d_probe, 14), dtype=np.float32)
            plans = plans + [(f"delay={d_probe} (2-D timestep)", probe_prefix, d_probe)] * 2

        print(f">>> 预热推理 {len(plans)} 次(在控制环之外付掉 torch.compile 的编译成本)...")
        for i, (label, pref, d) in enumerate(plans):
            t0 = time.perf_counter()
            args = (joint, grip, imgs) if pref is None else (joint, grip, imgs, pref, d)
            if pool is not None:
                pool.submit(self.policy.infer_chunk, *args).result()
            else:
                self.policy.infer_chunk(*args)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tag = "  <- 含编译" if dt_ms > 1000 else ""
            print(f"    {i + 1}/{len(plans)}  {label:26} {dt_ms:7.0f} ms{tag}")
        # 预热后自检：最后一次（前缀路径）若仍是秒级，说明还有图没热到 ——
        # 那会在控制环里再触发一次编译，重演「发起 N 次 / 落地 0 次」。
        if dt_ms > 1000:
            print(f"    ⚠️ 最后一次预热仍耗时 {dt_ms:.0f} ms —— 可能还有未预热的计算图。"
                  "上真机前请确认 [async 诊断] 里落地次数正常，否则考虑 compile_action_infer=false。")

        # 预热的耗时不该污染 _predict_delay 的实测窗口
        self.policy.stat_infer_s.clear()
        self.policy._warmed_up = True
        print("    预热完成，实测延迟窗口已重置")

    def warmup_before_episodes(self, obs: dict) -> None:
        """在 episode 计时之外做编译预热。

        async 下用一个工作线程跑，与真实执行语境一致。注意 `mode="default"` 的编译缓存是
        **全局的**（Dynamo/Inductor 按 code object + guards 索引），所以换线程也命中，
        这里用工作线程主要是为了贴合真实路径；只有 cudagraph 类 mode 才真正依赖线程身份，
        而那种组合在 async 下已被 EpisodeRunner 的启动守卫拒绝。"""
        if str(self.cfg.control.mode) == "async":
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warmup")
            try:
                self._warmup_inference(obs, pool=pool)
            finally:
                pool.shutdown(wait=True)
        else:
            self._warmup_inference(obs, pool=None)

    # ── 异步:后台推理,按实际经过步数对齐拼接 ────────────────────────────────
    def run_async(self, obs: dict) -> None:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        # ⚠️ 必须在控制环之前、且**通过 pool**预热：编译要发生在工作线程上
        self._warmup_inference(obs, pool=pool)
        chunk, res, _ = self._plan(obs)
        prev_norm = res["action_norm"]        # 归一化 chunk,RTC 的前缀从这里切
        idx = 0
        inflight = None
        fired_at = 0
        fired_delay = 0
        done = False
        n_prefix_overrun = 0
        # 诊断：区分「模型算了多久」(infer_s，不含预处理与 GIL 等待) 与
        # 「从发起到主循环取到结果隔了多久」(wall / elapsed 步)。两者差得多说明
        # 工作线程被 GIL 或 CPU 抢占，而不是模型慢。
        n_fire = 0
        fired_wall = 0.0
        land_log: list[tuple[int, int, float, float]] = []   # (fire_step, elapsed, wall_ms, infer_ms)
        try:
            while not done:
                # 没有在飞的推理就立刻发起一次(延迟决定了能用到 chunk 的哪一段)
                if inflight is None:
                    joint, grip = obs_to_physical(obs, self.travel)
                    imgs = {k: v.copy() for k, v in obs["images"].items()}
                    fired_at = self.step_i
                    fired_wall = time.perf_counter()
                    n_fire += 1
                    if self.rtc_on:
                        # 前缀 = 当前 chunk 对「新 chunk 前 d 步」那段时刻的预测。
                        # 新 chunk 的 t=0 对应发起推理的这一刻,也就是当前 chunk 的 idx,
                        # 所以前缀取 prev_norm[idx : idx+d]。
                        fired_delay = self._predict_delay()
                        prefix = prev_norm[idx: idx + fired_delay]
                        if prefix.shape[0] < fired_delay:
                            # 当前 chunk 剩不下 d 步了(推理慢/chunk 快用完),缩短 d
                            fired_delay = int(prefix.shape[0])
                        if fired_delay > 0:
                            inflight = pool.submit(self.policy.infer_chunk, joint, grip, imgs,
                                                   prefix.copy(), fired_delay)
                        else:
                            inflight = pool.submit(self.policy.infer_chunk, joint, grip, imgs)
                    else:
                        fired_delay = 0
                        inflight = pool.submit(self.policy.infer_chunk, joint, grip, imgs)
                    fired_measured = to_action_layout(joint, grip)

                if inflight is not None and inflight.done():
                    res = inflight.result()
                    inflight = None
                    elapsed = self.step_i - fired_at  # 该 chunk 的 t=0 已经过去了这么多步
                    land_wall_ms = (time.perf_counter() - fired_wall) * 1000.0
                    land_log.append((fired_at, elapsed, land_wall_ms, res["infer_s"] * 1000.0))
                    self._elapsed_hist.append(int(elapsed))
                    filtered, rep = self.safety.filter_chunk(res["action_phys"], fired_measured)
                    self.chunk_i += 1
                    self.rec.add_chunk(res, fired_measured, filtered, rep)
                    if elapsed >= filtered.shape[0]:
                        print(f"    ⚠️ [async] 推理耗时 {elapsed} 步 >= action_horizon "
                              f"{filtered.shape[0]} 步 —— 整个 chunk 已过期,只能用末步。"
                              "降 num_inference_steps 或改用 sync。")
                    chunk, idx = filtered, min(elapsed, filtered.shape[0] - 1)
                    prev_norm = res["action_norm"]
                    if self.rtc_on and elapsed > fired_delay:
                        # 实测延迟超过了发起时预估的 d:切入点落在前缀之外,这几步没有
                        # 前缀条件的保护(退化成 naive 拼接)。计数以便事后调 latency_percentile。
                        n_prefix_overrun += 1
                        if n_prefix_overrun in (1, 10, 100):
                            print(f"    ⚠️ [rtc] 实测 elapsed={elapsed} > 预估 d={fired_delay}"
                                  f"(第 {n_prefix_overrun} 次):切入点在前缀之外,接缝失去保护。"
                                  "调高 rtc.latency_percentile 或 rtc.min_delay")
                    if self.chunk_i % 5 == 1:
                        rtc_note = f"  d(预估)={fired_delay}" if self.rtc_on else ""
                        print(f"    [async] chunk {self.chunk_i} step {self.step_i}  "
                              f"infer {res['infer_s'] * 1000:.0f} ms  接入 chunk 第 {idx} 步"
                              f"{rtc_note}")

                if idx >= chunk.shape[0]:
                    # chunk 用尽而新的还没算完:保持上一条指令(而不是外推)
                    self.n_stall += 1
                    if self.n_stall in (1, 10, 100, 1000):
                        print(f"    ⚠️ [async] chunk 用尽,保持上一条指令(第 {self.n_stall} 次)。"
                              "推理耗时已接近/超过 action_horizon —— 降 num_inference_steps 或改用 sync")
                    hold = chunk[-1]
                    obs, done = self._send(hold, obs, chunk.shape[0] - 1)
                    continue

                obs, done = self._send(chunk[idx], obs, idx)
                idx += 1
        finally:
            pool.shutdown(wait=False)
            if self.rtc_on:
                print(f"    [rtc] 前缀 overrun {n_prefix_overrun} 次"
                      f"(elapsed > 预估 d,接缝失去保护的次数)")
            # 发起 vs 落地的账：n_fire 明显大于落地数说明推理没跟上；
            # wall 明显大于 infer 说明瓶颈在工作线程之外(预处理/GIL/CPU 抢占)。
            print(f"    [async 诊断] 发起 {n_fire} 次 / 落地 {len(land_log)} 次")
            for fire_step, el, wall_ms, inf_ms in land_log[:8]:
                print(f"      发起@step{fire_step:<4} elapsed={el:<3}步  "
                      f"wall={wall_ms:6.0f} ms  其中 infer={inf_ms:5.0f} ms  "
                      f"(差 {wall_ms - inf_ms:5.0f} ms = 预处理+GIL 等待)")
            if len(land_log) > 8:
                print(f"      ... 共 {len(land_log)} 条")


# ══════════════════════════════════════════════════════════════ 子命令
def build_policy(cfg: DictConfig) -> tuple[FastWAMPolicy, dict]:
    dcfg = task_data_cfg(cfg)
    if abs(float(cfg.control.hz) - 30.0) > 1e-6:
        print(f"!! control.hz={cfg.control.hz} 与数据集 fps=30 不一致 —— "
              "动作步长会与训练不符,除非你确知在做什么")
    return FastWAMPolicy(cfg, dcfg), dcfg


def cmd_selfcheck(cfg: DictConfig, args) -> int:
    """无硬件:用训练数据集的一个样本,验证部署侧流水线与训练**逐元素一致**。

    这一步能抓掉整条链路里最难查的几类错:相机顺序/改名、resize 顺序、通道翻转、
    proprio 双 key 归一化、反归一化路径。顺带量化 JPEG 传输对动作的影响。
    """
    policy, dcfg = build_policy(cfg)
    ds_dir = Path(cfg.preflight.dataset_dir or dcfg["dataset_dirs"][0])
    misc.register_work_dir(str(REPO_ROOT / ".cache" / "deploy_selfcheck"))

    print(f"\n>>> selfcheck:构建 val 数据集 {ds_dir}")
    ds = instantiate(
        policy.hcfg.data.val,
        dataset_dirs=[str(ds_dir)],
        pretrained_norm_stats=str(cfg.norm_stats),
    )
    idx = int(args.sample_idx) if args.sample_idx is not None else len(ds) // 2
    print(f"    共 {len(ds):,} 帧,取样本 idx={idx}")

    # (1) 训练路径:数据集自己产出的拼图与归一化 proprio
    sample = ds[idx]
    frame_train = sample["video"][:, 0].unsqueeze(0)  # [1,3,384,320] in [-1,1]
    proprio_train = sample["proprio"][0].reshape(1, -1)  # [1,14] 归一化

    # (2) 部署路径:拿同一样本的**原始**逐相机图与物理 state,过 ObsPipeline
    #     临时摘掉 processor 就能拿到 preprocess 之前的原始 sample(base_lerobot_dataset.py:269)
    base = ds.lerobot_dataset
    saved_proc, base.processor = base.processor, None
    try:
        raw = base[idx]
    finally:
        base.processor = saved_proc

    raw_imgs = {}
    for meta in dcfg["shape_meta"]["images"]:
        t = raw["images"][meta["key"]]  # [T,C,H,W] uint8
        raw_imgs[meta["key"]] = t[0].permute(1, 2, 0).numpy().astype(np.uint8)  # HWC RGB
    joint = raw["state"]["joint"][0].numpy().astype(np.float32)
    grip = raw["state"]["gripper_position"][0].numpy().astype(np.float32)

    # 部署侧的相机名就是训练名(selfcheck 不经过 service),临时用恒等映射
    saved_alias, policy.pipe.cam_alias = policy.pipe.cam_alias, {}
    try:
        frame_deploy = policy.pipe.compose_image(raw_imgs, cast=False)  # fp32,才比得出流水线差异
        proprio_deploy = policy.pipe.normalize_proprio(joint, grip)

        # JPEG 往返:量化传输压缩对动作的影响
        jpeg_imgs = {}
        for k, v in raw_imgs.items():
            import io as _io
            buf = _io.BytesIO()
            Image.fromarray(v).save(buf, format="JPEG", quality=95)
            buf.seek(0)
            with Image.open(buf) as im:
                jpeg_imgs[k] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        frame_jpeg = policy.pipe.compose_image(jpeg_imgs, cast=False)
    finally:
        policy.pipe.cam_alias = saved_alias

    ok = True
    d_img = (frame_deploy - frame_train.float()).abs()
    d_pro = (proprio_deploy - proprio_train).abs()
    print("\n>>> 逐元素比对(部署路径 vs 训练路径,均为 fp32)")
    print(f"    拼图    max|Δ| = {d_img.max():.3e}   mean|Δ| = {d_img.mean():.3e}   "
          f"(值域 [-1,1],期望**严格 0**)")
    print(f"    proprio max|Δ| = {d_pro.max():.3e}   mean|Δ| = {d_pro.mean():.3e}   (归一化,期望严格 0)")
    if d_img.max() > 1e-6:
        print("    ❌ 拼图不一致 —— 检查相机顺序/resize 顺序/通道");  ok = False
    if d_pro.max() > 1e-6:
        print("    ❌ proprio 不一致 —— 检查 shape_meta 顺序或 norm_stats");  ok = False

    # 结构性:相机顺序错了逐元素比对会大得离谱,但这里再单独打一眼三个区域,便于人眼确认
    for nm, f in (("部署", frame_deploy[0]), ("训练", frame_train[0])):
        print(f"    {nm}拼图三区均值: 上(顶部相机)={f[:, :256, :].mean():+.4f}  "
              f"下左(左腕)={f[:, 256:, :160].mean():+.4f}  下右(右腕)={f[:, 256:, 160:].mean():+.4f}")

    # bf16 量化与 JPEG 压缩:都是固有代价,不是 bug,但要知道量级
    d_bf16 = (frame_deploy.to(policy.dtype).float() - frame_deploy).abs()
    d_jpeg = (frame_jpeg - frame_train.float()).abs()
    print(f"    bf16 量化(训练/离线评测同样存在) max|Δ| = {d_bf16.max():.3e}")
    print(f"    JPEG q95 传输往返               max|Δ| = {d_jpeg.max():.3e}  "
          f"mean|Δ| = {d_jpeg.mean():.3e}")

    # (3) 反归一化闭环:对 GT action 做 backward 再 forward 应能还原
    gt_norm = sample["action"]
    gt_phys = denorm_action(policy.processor, gt_norm, proprio_train)
    print("\n>>> 反归一化闭环")
    print(f"    GT action 物理量  首步 {np.round(gt_phys[0].numpy(), 4)}")
    print(f"    夹爪维 (6,13)     {gt_phys[:, ACT_GRIP_IDX.tolist()].min():.4f} ~ "
          f"{gt_phys[:, ACT_GRIP_IDX.tolist()].max():.4f}  (应在 0~1 附近)")
    m_round = grip_m_to_frac(grip_frac_to_m(grip, cfg.units.gripper_travel_m), cfg.units.gripper_travel_m)
    print(f"    夹爪单位往返      {np.round(grip, 6)} -> "
          f"{np.round(grip_frac_to_m(grip, cfg.units.gripper_travel_m), 6)} m -> {np.round(m_round, 6)}")

    # (4) 端到端:两条路径的 chunk 差异
    print("\n>>> 端到端推理(两条路径各跑一次,seed 固定以便比较)")
    saved_seed = cfg.inference.seed
    cfg.inference.seed = 0
    policy.pipe.cam_alias = {}  # raw_imgs/jpeg_imgs 的键是训练名,selfcheck 不经过 service 改名
    try:
        r_dep = policy.infer_chunk(joint, grip, {**raw_imgs})
        r_jpg = policy.infer_chunk(joint, grip, {**jpeg_imgs})
    finally:
        policy.pipe.cam_alias = saved_alias
        cfg.inference.seed = saved_seed

    a1, a2 = r_dep["action_phys"], r_jpg["action_phys"]
    dj = np.abs(a1[:, ACT_JOINT_IDX] - a2[:, ACT_JOINT_IDX])
    print(f"    原图 vs JPEG:关节 max|Δ| = {np.degrees(dj.max()):.3f} 度, "
          f"mean|Δ| = {np.degrees(dj.mean()):.3f} 度")
    print(f"    单次推理耗时 {r_dep['infer_s'] * 1000:.0f} ms(预处理另计 {r_dep['pre_s'] * 1000:.0f} ms)")

    gt = gt_phys.numpy()
    e = np.abs(a1 - gt)[:, ACT_JOINT_IDX]  # [T,12] 关节误差(rad)
    per_step = np.degrees(e.mean(axis=1))
    marks = sorted({0, 4, 9, 15, 23, e.shape[0] - 1})
    print("    对 GT 的关节误差(单个样本,12 关节平均,单位度):")
    print("      步号  " + "  ".join(f"{k:>6d}" for k in marks))
    print("      误差  " + "  ".join(f"{per_step[k]:6.2f}" for k in marks))
    print(f"      全程平均 {per_step.mean():.2f} 度")
    print("    (离线评测在 284 个样本上的参考:首步约 1.4 度、末步约 8.6 度。"
          "单样本非单调很正常,别据此判断好坏)")

    print("\n" + ("✅ selfcheck 通过" if ok else "❌ selfcheck 有不一致项,先修再上真机"))
    return 0 if ok else 1


def _thread_candidates(n_all: int) -> list[int]:
    """要扫的线程数。含 1 是因为小张量上单线程有时真的最快（没有同步开销）。"""
    return sorted({x for x in (1, 2, 4, 8, 16, 32, n_all) if 1 <= x <= n_all})


def _sweep_threads(policy, joint, grip, imgs, n_rep: int, dt: float) -> int:
    """分别扫 torch_threads_pre 与 torch_threads_infer，打印最优组合。

    **可以分开扫**，因为两个阶段的耗时相互独立：`infer_chunk` 在预处理前后各调一次
    `torch.set_num_threads`，所以 `pre_s` 只受 `_nt_pre` 影响、`infer_s` 只受 `_nt_infer`
    影响。于是复杂度是 O(P+I) 而不是 O(P*I)，模型也只加载一次。

    为什么两个阶段要分别调（DEPLOY_DESIGN.md 实测，80 核 A800，10 步去噪）：
      预处理（3 相机 resize + 拼图，CPU 小张量）: 16 线程 58ms / 80 线程 290ms  <- 少的好
      infer_action（去噪循环的 CPU 侧算子）     : 16 线程 564ms / 80 线程 428ms  <- 多的好
    所以**不能**把两者绑在一起同增同减。
    """
    n_all = os.cpu_count() or 8
    cands = _thread_candidates(n_all)
    saved_pre, saved_infer = policy._nt_pre, policy._nt_infer
    print(f"\n>>> 线程扫描（整机 {n_all} 核，候选 {cands}，每组 {n_rep} 次有效测量）")

    def measure(n_pre: int, n_infer: int) -> tuple[float, float]:
        policy._nt_pre, policy._nt_infer = n_pre, n_infer
        pre_l, inf_l = [], []
        for i in range(n_rep + 2):          # 前 2 次含 warmup/首次分配，丢掉
            r = policy.infer_chunk(joint, grip, imgs)
            if i >= 2:
                pre_l.append(r["pre_s"]); inf_l.append(r["infer_s"])
        return float(np.mean(pre_l)), float(np.mean(inf_l))

    try:
        # ── 阶段一：扫 pre（infer 固定在原值）
        print(f"\n  [预处理] torch_threads_pre 扫描（infer 固定 {saved_infer}）")
        pre_results = {}
        for n in cands:
            pre_s, _ = measure(n, saved_infer)
            pre_results[n] = pre_s
            print(f"    pre={n:>3}  ->  预处理 {pre_s * 1000:7.1f} ms")
        best_pre = min(pre_results, key=pre_results.get)

        # ── 阶段二：扫 infer（pre 固定在刚找到的最优）
        print(f"\n  [推理] torch_threads_infer 扫描（pre 固定 {best_pre}）")
        inf_results = {}
        for n in cands:
            _, inf_s = measure(best_pre, n)
            inf_results[n] = inf_s
            print(f"    infer={n:>3}  ->  推理 {inf_s * 1000:7.1f} ms")
        best_infer = min(inf_results, key=inf_results.get)

        # ── 用最优组合实测一次（而不是把两段的最小值相加）
        print(f"\n  [复核] pre={best_pre} infer={best_infer}")
        pre_s, inf_s = measure(best_pre, best_infer)
        total = pre_s + inf_s
        ctl = total / dt
        # 当前配置值可能不在候选网格里（比如手填了 24），所以两边都用 .get
        base_pre = pre_results.get(saved_pre)
        base_inf = inf_results.get(saved_infer)
        base_total = (base_pre + base_inf) if (base_pre is not None and base_inf is not None) else None

        print(f"\n>>> 最优: torch_threads_pre={best_pre}  torch_threads_infer={best_infer}")
        print(f"    推理 {inf_s * 1000:.1f} ms + 预处理 {pre_s * 1000:.1f} ms "
              f"= {total * 1000:.1f} ms = {ctl:.1f} 个控制周期 (d)")
        if base_total:
            print(f"    对比当前配置 (pre={saved_pre} infer={saved_infer}): "
                  f"{base_total * 1000:.1f} ms -> {total * 1000:.1f} ms "
                  f"({(1 - total / base_total) * 100:+.0f}%)")
        else:
            print(f"    (当前配置 pre={saved_pre} infer={saved_infer} 不在候选网格内，跳过对比)")
        print(f"\n    写进 configs/deploy/*.yaml:")
        print(f"      inference:")
        print(f"        torch_threads_pre: {best_pre}")
        print(f"        torch_threads_infer: {best_infer}")
        if ctl > 11:
            print(f"\n    ⚠️ d={ctl:.1f} 仍 > 11（训练时 max_delay=12 的上限）—— 前缀会持续 overrun。"
                  "\n       下一步试 inference.compile_mode: default，或降 num_inference_steps。")
        elif ctl > 8:
            print(f"\n    d={ctl:.1f} 落在 9~11：可用，但建议把 rtc.latency_percentile 提到 0.95。")
        else:
            print(f"\n    ✅ d={ctl:.1f} <= 8，落在训练分布中央，可直接用。")
    finally:
        policy._nt_pre, policy._nt_infer = saved_pre, saved_infer
    return 0


def cmd_benchmark(cfg: DictConfig, args) -> int:
    """实测推理延迟,换算成控制步数,给出 replan_steps 建议。用合成观测,不需要硬件。"""
    policy, dcfg = build_policy(cfg)
    h, w = dcfg["video_size"]
    rng = np.random.default_rng(0)
    imgs = {}
    for meta in dcfg["shape_meta"]["images"]:
        rh, rw = meta["raw_shape"][1], meta["raw_shape"][2]
        imgs[meta["key"]] = rng.integers(0, 256, size=(rh, rw, 3), dtype=np.uint8)
    joint = policy.stats["state"]["joint"]["global_mean"].numpy().astype(np.float32)
    grip = policy.stats["state"]["gripper_position"]["global_mean"].numpy().astype(np.float32)
    saved_alias, policy.pipe.cam_alias = policy.pipe.cam_alias, {}

    if getattr(args, "sweep_threads", False):
        # 线程扫描用单一去噪步数（默认取 steps-list 第一个），否则组合数爆炸
        ns = int((args.steps_list or "5").split(",")[0])
        cfg.inference.num_inference_steps = ns
        print(f">>> 线程扫描 @ num_inference_steps={ns}, "
              f"compile_action_infer={bool(cfg.inference.compile_action_infer)}, "
              f"compile_mode={cfg.inference.get('compile_mode', 'reduce-overhead')}")
        try:
            return _sweep_threads(policy, joint, grip, imgs,
                                  n_rep=int(args.repeat), dt=1.0 / float(cfg.control.hz))
        finally:
            policy.pipe.cam_alias = saved_alias

    steps_list = [int(x) for x in (args.steps_list or "4,5,10").split(",")]
    n_rep = int(args.repeat)
    dt = 1.0 / float(cfg.control.hz)
    want = getattr(args, "compile", None)
    if want == "both":
        compile_flags = [False, True]
    elif want is not None:
        compile_flags = [want == "true"]
    else:
        compile_flags = [bool(cfg.inference.compile_action_infer)]

    want_mode = getattr(args, "compile_mode", None)
    if want_mode == "both":
        mode_list = ["default", "reduce-overhead"]
    elif want_mode is not None:
        mode_list = [want_mode]
    else:
        mode_list = [str(cfg.inference.get("compile_mode", "reduce-overhead"))]

    # compile=False 时 mode 无意义，去重避免重复跑
    combos = []
    for f in compile_flags:
        for m in (mode_list if f else mode_list[:1]):
            combos.append((f, m))
    # 全局预热：第一组会承担一次性初始化（CPU 线程池启动、首次内存分配、page cache），
    # 只靠每组丢弃 2 次盖不住 —— 表现为第一组的 pre_s 虚高（实测 20.2 vs 稳态 8.8 ms），
    # 而预处理耗时本该与去噪步数无关。这里先空跑几次把这些一次性开销吃掉。
    print(">>> 全局预热 3 次（吃掉一次性初始化，避免第一组虚高）...")
    for _ in range(3):
        policy.infer_chunk(joint, grip, imgs)

    rows = []
    try:
      for use_compile, cmode in combos:
        cfg.inference.compile_action_infer = bool(use_compile)
        cfg.inference.compile_mode = cmode
        label = f"compile={use_compile}" + (f" mode={cmode}" if use_compile else "")
        note = ""
        if use_compile and cmode in ("reduce-overhead", "max-autotune"):
            note = "   ⚠️ 该 mode 在 async(工作线程) 下会 AssertionError，此数仅主线程可得"
        print(f"\n>>> {label}{note}")
        for ns in steps_list:
            cfg.inference.num_inference_steps = ns
            inf_l, pre_l = [], []
            for i in range(n_rep + 2):  # 前 2 次含 CUDA warmup / 首次分配,丢掉
                r = policy.infer_chunk(joint, grip, imgs)
                if i >= 2:
                    inf_l.append(r["infer_s"])
                    pre_l.append(r["pre_s"])
            a, p = np.array(inf_l), np.array(pre_l)
            tot = a + p
            n_ctl = tot.mean() / dt
            rows.append({"num_inference_steps": ns, "compile": bool(use_compile),
                         "compile_mode": cmode if use_compile else "-",
                         "infer_s": a.mean(), "pre_s": p.mean(),
                         "mean_s": tot.mean(), "p90_s": float(np.percentile(tot, 90)),
                         "ctl_steps": n_ctl})
            print(f"    steps={ns:3d}  推理 {a.mean() * 1000:6.1f} ms + 预处理 {p.mean() * 1000:5.1f} ms "
                  f"= {tot.mean() * 1000:6.1f} ms (p90 {np.percentile(tot, 90) * 1000:.0f})  "
                  f"= {n_ctl:5.1f} 个控制周期 @{cfg.control.hz:g}Hz")
    finally:
        policy.pipe.cam_alias = saved_alias

    print("\n>>> 建议")
    rp = int(cfg.control.replan_steps)
    st = int(cfg.control.settle_steps)
    for r in rows:
        duty = rp * dt / (rp * dt + r["mean_s"] + st * dt)
        tag = f"compile={str(r['compile']):5} mode={r['compile_mode']:15}"
        d_est = int(np.ceil(r["ctl_steps"]))
        usable = "" if d_est <= 11 else "  ❌ d>11，超过训练 max_delay 上限"
        print(f"    steps={r['num_inference_steps']:3d} {tag}: d={d_est:>2}"
              f" | sync replan={rp} 占空比 {duty * 100:.0f}%(有效 {duty * float(cfg.control.hz):.1f} Hz)"
              f"{usable}")
    print("    ⚠️ 改 num_inference_steps 会改动作分布,上真机前先离线复测:")
    print(f"       python scripts/offline_eval.py --config configs/eval/final_A_newval71.yaml "
          f"--num-inference-steps {steps_list[0]} --out-dir /tmp/eval_ns{steps_list[0]}")
    return 0


def cmd_rtc_check(cfg: DictConfig, args) -> int:
    """无硬件:验证 RTC 前缀条件的部署链路。合成观测,不需要机器人。

    覆盖四件事:
      1. 时序对齐逻辑(纯数组,不碰模型)—— 前缀切片与切入点是否自洽
      2. `_predict_delay()` 的 fixed / measured / 上下限钳位
      3. `infer_chunk(prefix, delay)` 端到端:返回的 action_norm 前 d 步是否精确等于前缀
      4. 归一化空间 vs 物理量:确认前缀必须用 action_norm(两者确实不同)
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(f"{name} {detail}")

    # ── 1. 时序对齐(纯逻辑,与模型无关)────────────────────────────────────────
    print("[1] 时序对齐")
    H = 32
    # 造一个"上一个 chunk":第 k 步的值就是 k,便于追踪时刻
    prev = np.arange(H, dtype=np.float32)[:, None].repeat(14, axis=1)
    idx, d = 6, 5              # 当前正在执行第 6 步,预估延迟 5 步
    prefix = prev[idx: idx + d]
    check("前缀长度 == d", prefix.shape[0] == d, f"{prefix.shape[0]}")
    check("前缀起点 == 发起推理的那一刻(prev[idx])",
          float(prefix[0, 0]) == float(idx), f"{float(prefix[0, 0])} vs {idx}")
    check("前缀终点 == prev[idx+d-1]",
          float(prefix[-1, 0]) == float(idx + d - 1))
    # 结果在 elapsed 步后到达,按 run_async 的逻辑从第 elapsed 步切入
    for elapsed in (d, d - 2, d + 3):
        enter = min(elapsed, H - 1)
        protected = enter <= d          # 切入点落在前缀内 -> 有连续性保护
        note = "受保护" if protected else "超出前缀(退化成 naive 拼接)"
        print(f"       elapsed={elapsed} -> 切入第 {enter} 步,{note}")
    check("elapsed == d 时切入点正好接在前缀末尾之后", min(d, H - 1) == d)

    # ── 2. _predict_delay ────────────────────────────────────────────────────
    print("[2] _predict_delay")
    policy_stub = type("P", (), {"stat_infer_s": [], "model": None})()
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = cfg
    runner.policy = policy_stub
    # `__new__` 跳过了 __init__,所以实例属性一个都没有。`_predict_delay` 会读
    # `_elapsed_hist`(首选依据,见 INFER_LATENCY_DEBUG 坑 5),不补上这行会 AttributeError
    # —— 这个 stub 是在 _elapsed_hist 之前写的,加那个字段时漏了同步。
    runner._elapsed_hist = []

    cfg.rtc.delay_mode = "fixed"
    cfg.rtc.fixed_delay = 5
    check("fixed 模式返回 fixed_delay", runner._predict_delay() == 5, str(runner._predict_delay()))

    cfg.rtc.delay_mode = "measured"
    policy_stub.stat_infer_s = []
    check("measured 冷启动回落到 fixed_delay", runner._predict_delay() == 5)

    hz = float(cfg.control.hz)
    # ⚠️ 这几条断言校验的是**次选路径**(只有 infer_s、还没有 elapsed 样本)。该路径故意
    # `+1 步`兜底:infer_s 不含预处理与 GIL 等待,会系统性低估 1 步进而 overrun
    # (INFER_LATENCY_DEBUG 坑 5:infer_s p90 197ms -> d=6,而真实 wall 233ms -> d=7)。
    # 所以期望值都要含这个 +1 —— 原来的断言写在加兜底之前,一直是错的。
    policy_stub.stat_infer_s = [3.5 / hz] * 20          # 每次 3.5 个控制步
    got = runner._predict_delay()
    check("measured 由实测耗时换算(3.5 步 -> ceil=4,再 +1 兜底 = 5)", got == 5, str(got))

    policy_stub.stat_infer_s = [999.0]                   # 荒谬地慢
    check("上限钳位到 max_delay", runner._predict_delay() == int(cfg.rtc.max_delay),
          str(runner._predict_delay()))
    policy_stub.stat_infer_s = [1e-6]                     # 荒谬地快
    # ceil(1e-6*30)=1,+1 兜底 = 2;min_delay=1 时它不该被钳位,所以期望 2 而不是 1。
    want_fast = max(int(cfg.rtc.min_delay), 2)
    check("极快时取 ceil+1 且不低于 min_delay", runner._predict_delay() == want_fast,
          str(runner._predict_delay()))

    # 真正的首选路径:有 elapsed 样本时直接用它的分位数,**不加** +1
    # (elapsed 本身就是控制步,已含预处理/GIL/控制环速率的全部影响)。
    runner._elapsed_hist = [7, 6, 6, 6, 7, 6, 6, 6]
    got = runner._predict_delay()
    check("有 elapsed 样本时用其 p90(=7)且不再 +1", got == 7, str(got))
    runner._elapsed_hist = []

    # ── 3/4. 端到端 infer_chunk ──────────────────────────────────────────────
    print("[3] infer_chunk 前缀条件(加载真实模型)")
    policy, dcfg = build_policy(cfg)
    rng = np.random.default_rng(0)
    imgs = {}
    for meta in dcfg["shape_meta"]["images"]:
        rh, rw = meta["raw_shape"][1], meta["raw_shape"][2]
        imgs[meta["key"]] = rng.integers(0, 256, size=(rh, rw, 3), dtype=np.uint8)
    joint = policy.stats["state"]["joint"]["global_mean"].numpy().astype(np.float32)
    grip = policy.stats["state"]["gripper_position"]["global_mean"].numpy().astype(np.float32)
    saved_alias, policy.pipe.cam_alias = policy.pipe.cam_alias, {}
    try:
        base = policy.infer_chunk(joint, grip, imgs)
        check("无前缀调用返回 delay=0", base["delay"] == 0, str(base["delay"]))
        check("_rtc_kwargs 在无前缀时为空", policy._rtc_kwargs(None, 0) == {})
        check("_rtc_kwargs 在 delay=0 时为空", policy._rtc_kwargs(base["action_norm"], 0) == {})

        d = 5
        pref = base["action_norm"][6: 6 + d].copy()
        out = policy.infer_chunk(joint, grip, imgs, pref, d)
        err = float(np.abs(out["action_norm"][:d] - pref).max())
        check("返回的 action_norm 前 d 步精确等于前缀(clamp 生效)", err < 1e-5, f"max_err={err:.3e}")
        check("返回 delay=d", out["delay"] == d, str(out["delay"]))
        tail = float(np.abs(out["action_norm"][d:] - base["action_norm"][d:]).max())
        check("前缀确实改变了后缀(模型没忽略条件)", tail > 1e-4, f"max_diff={tail:.3e}")

        # 归一化 vs 物理量:必须用 action_norm 当前缀
        gap = float(np.abs(base["action_norm"] - base["action_phys"]).max())
        check("action_norm 与 action_phys 确实不同(前缀必须用 norm)", gap > 1e-3, f"{gap:.3e}")
    finally:
        policy.pipe.cam_alias = saved_alias

    # ── 4. 线程模型 x compile_mode（复现并验证 CUDA Graph Trees 的线程限制）───────
    print("[4] compile x 后台线程")
    if not torch.cuda.is_available():
        print("  跳过（无 CUDA）")
    else:
        saved_alias, policy.pipe.cam_alias = policy.pipe.cam_alias, {}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtccheck")
        try:
            for cmode, expect_ok in (("default", True), ("reduce-overhead", False)):
                cfg.inference.compile_action_infer = True
                cfg.inference.compile_mode = cmode
                try:
                    # 关键：在**工作线程**上跑，与 run_async 的时序一致
                    r = pool.submit(policy.infer_chunk, joint, grip, imgs).result()
                    ok, note = True, f"delay={r['delay']}"
                except AssertionError as e:
                    ok, note = False, f"AssertionError（CUDA Graph Trees 的 TLS 缺失）"
                except Exception as e:
                    ok, note = False, f"{type(e).__name__}: {str(e)[:50]}"
                verdict = "可用" if ok else "不可用"
                if ok == expect_ok:
                    print(f"  ok   compile_mode={cmode:16} 工作线程上{verdict}  {note}")
                else:
                    print(f"  FAIL compile_mode={cmode:16} 预期{'可用' if expect_ok else '不可用'}"
                          f"，实际{verdict}  {note}")
                    failures.append(f"compile_mode={cmode} 线程行为与预期不符")
            # default 模式下前缀 clamp 仍须精确
            cfg.inference.compile_mode = "default"
            d = 5
            base2 = pool.submit(policy.infer_chunk, joint, grip, imgs).result()
            pref2 = base2["action_norm"][6: 6 + d].copy()
            out2 = pool.submit(policy.infer_chunk, joint, grip, imgs, pref2, d).result()
            err2 = float(np.abs(out2["action_norm"][:d] - pref2).max())
            check("compile_mode=default 下前缀 clamp 仍精确（工作线程）", err2 < 1e-5,
                  f"max_err={err2:.3e}")
        finally:
            pool.shutdown(wait=False)
            policy.pipe.cam_alias = saved_alias

    print()
    if failures:
        print(f"{len(failures)} 项失败:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RTC 部署链路校验全部通过。")
    return 0


def cmd_run(cfg: DictConfig, args) -> int:
    policy, dcfg = build_policy(cfg)
    safety = SafetyFilter(policy.stats, cfg.safety)

    sp = None
    if str(cfg.preflight.start_pose_stats or "").lower() not in ("", "none", "null"):
        ds_dir = Path(cfg.preflight.dataset_dir or dcfg["dataset_dirs"][0])
        cache = (REPO_ROOT / "runs" / "_shared" / "start_pose_stats.json"
                 if str(cfg.preflight.start_pose_stats) == "auto"
                 else Path(str(cfg.preflight.start_pose_stats)))
        print(f">>> 起始位姿分布({ds_dir.name}, 最多 {cfg.preflight.start_pose_episodes} 个 episode)...")
        sp = start_pose_stats(ds_dir, int(cfg.preflight.start_pose_episodes), cache)

    mode = str(cfg.control.mode).lower()
    if mode not in ("sync", "async"):
        raise SystemExit(f"control.mode 只能是 sync / async,得到 {mode!r}")
    if mode == "async":
        # 这里不再写死延迟数字 —— 它随 num_inference_steps / compile_mode / 机器变化很大：
        #   5 步 无compile        377 ms -> d≈11      5 步 compile+default  159 ms -> d≈5
        #   10 步 compile+default 297 ms -> d≈9
        # 真实值请用 benchmark 子命令实测；运行时 _predict_delay 会按实测 p90 自动定 d。
        print(f"\n⚠️ async 模式:切入 chunk 的第几步 **等于推理延迟(控制步)**，与发起时机无关。"
              f"\n   实测延迟请跑:python scripts/deploy_real.py benchmark --config <本配置>"
              f"\n   当前:num_inference_steps={cfg.inference.num_inference_steps}, "
              f"compile={bool(cfg.inference.compile_action_infer)}, "
              f"mode={cfg.inference.get('compile_mode', 'reduce-overhead')}, "
              f"rtc.max_delay={cfg.rtc.max_delay if cfg.rtc.enabled else '-'}"
              f"\n   若日志出现「前缀 overrun」或「chunk 用尽」，说明延迟超出了 max_delay/action_horizon。\n")

    rec_dir = Path(cfg.record.dir) if cfg.record.dir else None
    replan_hz = float(cfg.control.hz) / max(1, int(cfg.control.replan_steps))
    print(f">>> 连接 {cfg.robot.endpoint}")
    client = RobotArmClient(str(cfg.robot.endpoint), float(cfg.robot.recv_timeout_s)).connect()
    print("    service 在线")

    n_ep = max(1, int(cfg.control.num_episodes))
    rc = 0
    try:
        for ep in range(1, n_ep + 1):
            rec = Recorder(rec_dir, bool(cfg.record.save_video), replan_hz)
            print(f"\n═══ Episode {ep}/{n_ep} ═══")
            print(">>> reset:service 走初始位姿(约 5 s)...")
            obs = client.reset()
            if ep == 1:
                preflight_obs(policy, obs, cfg, sp)
                if args.confirm:
                    ans = input(">>> 以上检查确认无误?手臂即将开始运动。输入 yes 继续: ").strip().lower()
                    if ans not in ("y", "yes"):
                        print("已取消。")
                        return 0

            runner = EpisodeRunner(policy, client, cfg, safety, rec)
            if ep == 1:
                # 编译预热放在 episode 计时**之外**、且只做一次：
                #   - 它是 per-policy 的一次性成本（首次约 17 s），不该算进 episode 时长
                #     （否则「有效 Hz」会被严重低估：实测 19.9 s 里 17 s 是编译）
                #   - 也不该在每个 episode 重复（compile 结果已缓存，但没必要再跑）
                runner.warmup_before_episodes(obs)
            t0 = time.perf_counter()
            try:
                if mode == "sync":
                    runner.run_sync(obs)
                else:
                    runner.run_async(obs)
            except KeyboardInterrupt:
                print("\n>>> 收到 Ctrl-C,停止下发指令(手臂保持当前位姿)")
                rc = 130
            except RuntimeError as exc:
                print(f"\n!! episode 中止: {exc}")
                rc = 1
            wall = time.perf_counter() - t0

            lat = np.array(policy.stat_infer_s[-max(runner.chunk_i, 1):])
            print(f"\n--- Episode {ep} 结束 ---")
            print(f"    {runner.step_i} 控制步 / {runner.chunk_i} 次规划 / {wall:.1f} s "
                  f"(有效 {runner.step_i / max(wall, 1e-6):.1f} Hz,目标 {cfg.control.hz:g} Hz)")
            print(f"    节拍: {runner.pacer.report()}")
            if lat.size:
                print(f"    推理: 均值 {lat.mean() * 1000:.0f} ms  p90 {np.percentile(lat, 90) * 1000:.0f} ms")
            if mode == "async":
                print(f"    async 停顿(chunk 用尽): {runner.n_stall} 次"
                      + ("  <- 非 0 说明推理跟不上,该降 num_inference_steps 或改 sync" if runner.n_stall else ""))
            print(f"    安全层: 首步跳变告警 {safety.n_jump_warn} 次, "
                  f"delta 钳位 {safety.n_delta_clamped} 次, 限位钳位 {safety.n_limit_clamped} 次")
            print(f"    TCP: {client.stat_calls} 次往返, 累计 {client.stat_rtt_s:.1f} s "
                  f"(均 {client.stat_rtt_s / max(client.stat_calls, 1) * 1000:.1f} ms), "
                  f"JPEG 解码累计 {client.stat_decode_s:.1f} s")
            rec.flush(ep, {
                "episode": ep, "mode": mode, "steps": runner.step_i, "chunks": runner.chunk_i,
                "wall_s": round(wall, 2), "control_hz": float(cfg.control.hz),
                "replan_steps": int(cfg.control.replan_steps),
                "async_stalls": runner.n_stall,
                "num_inference_steps": int(cfg.inference.num_inference_steps),
                "infer_s_mean": (float(lat.mean()) if lat.size else None),
                "gripper_travel_m": float(cfg.units.gripper_travel_m),
                "weight": policy.weight_info, "instruction": str(cfg.instruction),
                "safety": {"jump_warn": safety.n_jump_warn, "delta_clamped": safety.n_delta_clamped,
                           "limit_clamped": safety.n_limit_clamped},
                "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
            })
            if rc:
                break
    finally:
        client.close(send_stop=bool(cfg.robot.go_zero_on_exit))
        print(">>> 已断开" + ("(已发 stop,双臂归零)" if cfg.robot.go_zero_on_exit else "(手臂保持当前位姿)"))
    return rc


# ══════════════════════════════════════════════════════════════════ CLI
def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="FastWAM Agilex 真机部署推理(config 驱动,命令行可覆盖)",
        epilog="联调顺序:selfcheck(无硬件) -> benchmark(无硬件) -> run --max-steps 60 -> run",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="部署 YAML,见 configs/deploy/agilex_real.yaml")
    common.add_argument("--task")
    common.add_argument("--checkpoint")
    common.add_argument("--norm-stats")
    common.add_argument("--instruction", help="必须与训练 tasks.jsonl 逐字一致(T5 缓存按它的 sha256 找)")
    common.add_argument("--device")
    common.add_argument("--num-inference-steps", type=int)
    common.add_argument("--gripper-travel-m", type=float, help="夹爪满开度(米),默认 0.07")
    common.add_argument("--torch-threads-pre", type=int, help="预处理阶段 CPU 线程数(0=不限)。实测 16 最优")
    common.add_argument("--torch-threads-infer", type=int, help="推理阶段 CPU 线程数(0=不限)。实测不限最优")
    common.add_argument("--overrides", nargs="*", help="额外 hydra 覆盖")

    p_sc = sub.add_parser("selfcheck", parents=[common],
                          help="无硬件:验证部署侧预处理与训练逐元素一致")
    p_sc.add_argument("--sample-idx", type=int, help="用哪个样本,默认取中间那个")

    p_bm = sub.add_parser("benchmark", parents=[common], help="无硬件:实测推理延迟")
    sub.add_parser("rtc-check", parents=[common],
                   help="无硬件:验证 RTC 前缀条件的部署链路")
    p_bm.add_argument("--steps-list", default="4,5,10", help="要测的 num_inference_steps,逗号分隔")
    p_bm.add_argument("--repeat", type=int, default=5)
    p_bm.add_argument(
        "--sweep-threads", action="store_true",
        help="扫 torch_threads_pre / torch_threads_infer 找最优组合（模型只加载一次）。"
             "两个阶段分别扫，因为它们要的线程数相反 —— 当前默认值 16/0 是在 80 核 A800 上"
             "调出来的，换机器必须重扫",
    )
    p_bm.add_argument(
        "--compile-mode", choices=["default", "reduce-overhead", "both"], default=None,
        help="覆盖 inference.compile_mode。both = 两种都测。⚠️ benchmark 跑在**主线程**，"
             "所以 reduce-overhead 在这里不会崩、会给出一个 async 下拿不到的乐观数字；"
             "它的意义是「CUDA Graphs 值多少钱」= 把推理挪回主线程能换回多少",
    )
    p_bm.add_argument(
        "--compile", choices=["true", "false", "both"], default=None,
        help="覆盖 inference.compile_action_infer。both = 两种都跑并对比("
             "RoPE 用复数算子,Inductor 无法 codegen 会警告 'may be worse than eager',"
             "所以必须实测而不能假设编译一定更快)",
    )

    p_run = sub.add_parser("run", parents=[common], help="真机 rollout")
    p_run.add_argument("--endpoint", help="RobotArmService 地址,如 tcp://192.168.1.10:9900")
    p_run.add_argument("--mode", choices=["sync", "async"])
    p_run.add_argument("--replan-steps", type=int)
    p_run.add_argument("--hz", type=float)
    p_run.add_argument("--max-steps", type=int, help="每个 episode 步数上限,0=不限")
    p_run.add_argument("--num-episodes", type=int)
    p_run.add_argument("--record-dir")
    p_run.add_argument("--no-save-video", action="store_true",
                       help="不录模型视角 mp4。长时间连续跑必加 —— 拼图帧在内存里攒,"
                            "约 1.5 GB/小时(其余记录只有 40 MB/小时)")
    p_run.add_argument("--no-safety", action="store_true", help="关掉钳位(**不建议**,只用于对照)")
    p_run.add_argument("--confirm", action="store_true",
                       help="preflight 之后等人工确认再动手臂(第一次上真机建议加)")

    args = ap.parse_args()
    cfg = load_config(args)
    print(OmegaConf.to_yaml(cfg))

    if args.cmd == "selfcheck":
        return cmd_selfcheck(cfg, args)
    if args.cmd == "benchmark":
        return cmd_benchmark(cfg, args)
    if args.cmd == "rtc-check":
        return cmd_rtc_check(cfg, args)
    return cmd_run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
