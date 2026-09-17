"""FastWAM 推理策略(模型 + processor + 观测流水线)—— **需要 torch，只在 GPU 侧用**。

从 `scripts/deploy_real.py` 抽出来，好让单进程的 `deploy_real.py` 与 GPU 侧的
`scripts/fastwam_server.py` 共用同一份预处理/推理/反归一化实现。这很关键：
`selfcheck` 子命令保证的「部署侧预处理与训练逐元素一致」只对**这一份**代码成立，
复制第二份出去就等于放弃那个保证。

════════════════════════════════════════════════════════════════════════════
                    ⚠️ 线程亲和性 —— 本模块存在的主要理由
════════════════════════════════════════════════════════════════════════════
`torch.compile(mode="reduce-overhead")` 会启用 Inductor 的 CUDA Graph Trees，
它的状态存在 `threading.local()` 里、只在 import `torch._inductor.cudagraph_trees`
的那个线程上初始化(该模块第 277-292 行是顶层语句)。

本机实测(torch 2.7.1+cu128)确认的**真实规则**比原注释更精确：

    ✅ 专属线程首次 compile+record，之后一直用它       -> 正常，且最快
    ❌ 主线程先 record，再换另一个线程 record 新图     -> AssertionError(:325)
    ⚠️ 主线程先 record，工作线程 replay 同一张已录图   -> 不报错，但主线程后续劣化

即 **CUDA Graph Trees 不要求「主线程」，只要求「从头到尾同一个线程」**。

所以 RTC(必须异步执行)与 `reduce-overhead` 并不矛盾 —— 只要把**编译、预热、
每一次推理**全部钉在同一个专属线程上。`InferenceThread` 就是干这个的，它替代了
`deploy_real.py` 里那个过严的「async + cudagraph mode 一律拒绝」启动守卫。

⚠️ 因此绝对不要在别处直接调 `FastWAMPolicy.infer_chunk`——一律经由 `InferenceThread`。
   `assert_owner()` 会在被别的线程调用时立刻报错，而不是等到 CUDA Graphs 静默劣化。
"""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from omegaconf import OmegaConf
from PIL import Image

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


# ═════════════════════════════════════════════════════════════════ 反归一化
def denorm_action(processor, action_norm: torch.Tensor, proprio_norm: torch.Tensor) -> torch.Tensor:
    """归一化 action [T,D] -> 物理量 [T,D]。逻辑与 trainer.py:503-535 / offline_eval.py 一致。

    必须走 merger.backward -> normalizer.backward -> merger.forward 这一圈:normalizer 是按
    shape_meta 的子 key(joint / gripper_position)分别统计的,不能对拼接后的 14 维直接反归一化。

    ⚠️ `ConcatLeftAlign._crop` 断言 ndim==3(`action_state_merger.py:56-57`),
    **action 和 state 都要带 batch 维**,漏了 state 那一句会 AssertionError。

    部署时只有 t=0 一行 proprio,这里复制成 T 行 —— 反归一化是逐元素线性变换、
    且 `use_stepwise_action_norm=False`(全局统计量),所以 action 的结果与 state 的取值无关,
    传 state 纯粹是为了满足上面那两个 backward 的接口。
    """
    a = action_norm if action_norm.ndim == 3 else action_norm.unsqueeze(0)  # [1,T,D]
    n_t = int(a.shape[1])
    p = proprio_norm.reshape(1, -1, int(proprio_norm.shape[-1]))
    if p.shape[1] == 1:
        p = p.expand(1, n_t, -1)
    batch = {
        "action": a.detach().to("cpu", torch.float32),
        "state": p.detach().to("cpu", torch.float32).contiguous(),
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {m["key"]: batch["action"][m["key"]].squeeze(0) for m in processor.shape_meta["action"]},
        "state": {m["key"]: batch["state"][m["key"]].squeeze(0) for m in processor.shape_meta["state"]},
    }
    merged = processor.action_state_merger.forward(merged)
    out = merged["action"]
    if out.ndim != 2:
        raise ValueError(f"反归一化后应为 [T,D],得到 {tuple(out.shape)}")
    return out


def action_affine(processor) -> tuple[np.ndarray, np.ndarray]:
    """取 action 的归一化仿射系数 (scale, offset),满足 norm = phys*scale + offset。

    只有 `use_stepwise_action_norm=False`(本数据集正是如此)时这才是**与 proprio 无关的
    固定逐元素变换**，也正因如此 RTC 前缀可以走「绝对物理量」上线协议：
    client 不必持有 norm_stats，server 收到后再换算回归一化空间。
    实测往返误差 phys->norm->phys 1.2e-07 / norm->phys->norm 4.8e-07。

    `fastwam_server.py` 启动时会用它做一次往返自检，把这个前提钉死而不是假设。
    """
    metas = processor.shape_meta["action"]
    if len(metas) != 1:
        raise ValueError(f"本部署假设 action 只有一个 key,得到 {[m['key'] for m in metas]}")
    norm = processor.normalizer.normalizers["action"][metas[0]["key"]]
    if getattr(processor, "use_stepwise_action_norm", False):
        raise ValueError(
            "use_stepwise_action_norm=True 时 action 归一化按步变化,"
            "「绝对物理量前缀」协议不成立 —— 需要改回传归一化前缀。"
        )
    return (norm.scale.detach().cpu().numpy().astype(np.float32),
            norm.offset.detach().cpu().numpy().astype(np.float32))


# ══════════════════════════════════════════════════════════════ 观测流水线
class ObsPipeline:
    """把原始观测变成模型输入,**逐算子照抄训练路径**。

    图像(每相机 480x640 RGB uint8 起):
        ToTensor(/255) -> Resize[240,320]          <- processor.val_transforms,逐相机
        cam_high  -> 256x320                        \\
        双腕      -> 各 128x160,左右拼成 128x320    | robot_video_dataset.py:170-194
        竖拼 -> 384x320                             /
        ResizeSmallestSideAspectPreserving(no-op) -> CenterCrop(no-op) -> Normalize(0.5,0.5)
        => [1,3,384,320],值域 [-1,1]

    proprio:
        (12 关节弧度, 2 夹爪分数) -> action_state_transform -> normalizer.forward(z-score,
        **会 clamp 到 ±5**)-> merger.forward(拼成 14 维)=> [1,14]
    """

    def __init__(self, processor, dcfg: dict, cam_alias: dict, device: str, dtype: torch.dtype):
        if dcfg["concat_multi_camera"] != "robotwin":
            raise SystemExit(
                f"本部署只支持 concat_multi_camera='robotwin'(agilex 3 相机拼图),"
                f"当前 task 是 {dcfg['concat_multi_camera']!r}"
            )
        self.processor = processor
        self.image_meta = dcfg["shape_meta"]["images"]
        if len(self.image_meta) != 3:
            raise SystemExit(f"robotwin 拼图要求正好 3 个相机,配置里有 {len(self.image_meta)} 个")
        self.cam_alias = dict(OmegaConf.to_container(cam_alias, resolve=True)
                              if OmegaConf.is_config(cam_alias) else (cam_alias or {}))
        self.device, self.dtype = device, dtype

        tr = processor.val_transforms
        if isinstance(tr, dict):
            raise SystemExit("processor.val_transforms 是 per-key dict,本部署只支持统一的 list 形式")
        self.cam_transforms = list(tr)

        h, w = int(dcfg["video_size"][0]), int(dcfg["video_size"][1])
        self.resize_t = ResizeSmallestSideAspectPreserving(args={"img_w": w, "img_h": h})
        self.crop_t = CenterCrop(args={"img_w": w, "img_h": h})
        self.norm_t = Normalize(args={"mean": 0.5, "std": 0.5})
        self.out_hw = (h, w)

    # ── 图像 ────────────────────────────────────────────────────────────────
    def compose_image(self, images: dict[str, np.ndarray], cast: bool = True) -> torch.Tensor:
        """{相机名: HWC uint8 RGB} -> [1,3,384,320] in [-1,1]。

        cast=True(默认)会转到 model 的 device/dtype(bf16)—— 与训练、离线评测一致。
        cast=False 保留 fp32 CPU,只给 selfcheck 做**逐元素**比对用:bf16 在 1.0 附近的
        间隔是 2^-8,单看 max|Δ| 会有约 2e-3 的量化噪声,会盖住真正的流水线差异。
        """
        per_cam = []
        for meta in self.image_meta:  # 顺序即 cam_high -> cam_left_wrist -> cam_right_wrist
            train_key = meta["key"]
            src_key = self.cam_alias.get(train_key, train_key)
            if src_key not in images:
                raise KeyError(
                    f"观测里没有相机 '{src_key}'(训练名 '{train_key}');"
                    f"实际有 {sorted(images)};改名映射见 robot.cameras"
                )
            arr = np.asarray(images[src_key])
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"相机 '{src_key}' 应为 HWC uint8,得到 {arr.shape} {arr.dtype}")
            x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W] uint8
            for t in self.cam_transforms:  # ToTensor 断言 uint8,所以必须在这之前保持 uint8
                x = t(x)
            if list(x.shape[1:]) != list(meta["shape"]):
                raise ValueError(
                    f"相机 '{train_key}' 过完 val_transforms 后是 {list(x.shape[1:])},"
                    f"配置要求 {meta['shape']}"
                )
            per_cam.append(x)

        video = torch.stack(per_cam, dim=0)  # [3,1,C,240,320],与数据集的 [cam,T,C,H,W] 同形
        top = TVF.resize(video[0], size=[256, 320], interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        left = TVF.resize(video[1], size=[128, 160], interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        right = TVF.resize(video[2], size=[128, 160], interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        frame = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)  # [1,C,384,320]

        frame = self.norm_t(self.crop_t(self.resize_t(frame)))
        if tuple(frame.shape[-2:]) != self.out_hw:
            raise ValueError(f"拼图后应为 {self.out_hw},得到 {tuple(frame.shape[-2:])}")
        return frame.to(device=self.device, dtype=self.dtype) if cast else frame

    # ── proprio ─────────────────────────────────────────────────────────────
    def normalize_proprio(self, joint12: np.ndarray, grip_frac2: np.ndarray) -> torch.Tensor:
        batch = {
            "state": {
                "joint": torch.as_tensor(joint12, dtype=torch.float32).reshape(1, -1),
                "gripper_position": torch.as_tensor(grip_frac2, dtype=torch.float32).reshape(1, -1),
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        out = batch["state"]  # [1,14]
        if tuple(out.shape) != (1, 14):
            raise ValueError(f"proprio 归一化后应为 (1,14),得到 {tuple(out.shape)}")
        return out

    @staticmethod
    def frame_to_pil(frame: torch.Tensor) -> Image.Image:
        """[1,3,H,W] in [-1,1] -> PIL,用于存 mp4 / 人眼比对。"""
        x = frame[0].detach().float().cpu().clamp(-1, 1)
        x = ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return Image.fromarray(x.permute(1, 2, 0).numpy())


# ═══════════════════════════════════════════════════════════════════ 策略
class FastWAMPolicy:
    """模型 + processor + 文本条件,对外只暴露 `infer_chunk`。

    ⚠️ 不要直接调 `infer_chunk` —— 用 `InferenceThread` 包住它(见模块 docstring 的线程亲和性)。
    """

    def __init__(self, cfg, dcfg: dict):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.cfg, self.dcfg = cfg, dcfg
        self.device = str(cfg.inference.device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("!! CUDA 不可用,退到 CPU(会非常慢)")
            self.device = "cpu"
        self.dtype = torch.bfloat16

        n_all = os.cpu_count() or 8
        self._nt_pre = int(cfg.inference.get("torch_threads_pre") or n_all)
        self._nt_infer = int(cfg.inference.get("torch_threads_infer") or n_all)
        print(f">>> torch CPU 线程: 预处理 {self._nt_pre} / 推理 {self._nt_infer}(整机 {n_all} 核)")

        ov = [f"task={cfg.task}",
              f"model.skip_dit_load_from_pretrain={bool(cfg.inference.get('skip_base_dit_load', True))}"]
        ov += list(cfg.runtime.overrides or [])   # 用户的 overrides 在后,可覆盖上面两条
        configs_dir = str(cfg.get("configs_dir") or (Path(__file__).resolve().parents[3] / "configs"))
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
            hcfg = compose(config_name="train", overrides=ov)
        self.hcfg = hcfg

        print(f">>> 构建 processor(norm_stats={cfg.norm_stats})...")
        self.processor = instantiate(hcfg.data.val.processor).eval()
        self.stats = load_dataset_stats_from_json(str(cfg.norm_stats))
        self.processor.set_normalizer_from_stats(self.stats)

        self.pipe = ObsPipeline(self.processor, dcfg, cfg.robot.cameras, self.device, self.dtype)

        print(f">>> 构建模型(task={cfg.task}, dtype=bf16, device={self.device})...")
        t0 = time.perf_counter()
        self.model = instantiate(hcfg.model, model_dtype=self.dtype, device=self.device)
        print(f"    模型就绪,耗时 {time.perf_counter() - t0:.1f}s")

        self.weight_info = self._load_and_verify(Path(cfg.checkpoint))
        self.model.eval()

        # 探测 infer_action 支持哪些可选参数，避免部署脚本与 src/ 的模型代码版本不同步时
        # 直接 TypeError。dryrun_fastwam.py:202 用的是同一套做法。
        self._infer_params = set(inspect.signature(self.model.infer_action).parameters)

        self.context, self.context_mask = self._load_text_context()
        horizon = cfg.inference.action_horizon
        self.action_horizon = int(horizon) if horizon else dcfg["num_frames"] - 1
        print(f">>> action_horizon = {self.action_horizon} 步 = {self.action_horizon / 30.0:.3f} s @30Hz")

        self.stat_infer_s: list[float] = []
        self._warmed_up = False

    # ── 权重 ────────────────────────────────────────────────────────────────
    def _load_and_verify(self, path: Path) -> dict:
        """加载权重并确认**真的加载上了**。

        `load_state_dict(strict=False)`(fastwam.py:1214)键名不匹配会被静默忽略,
        跑出来是随机初始化却看着挺"正常"。本模型应为 1649/1649,不匹配直接退出。
        """
        if not path.is_file():
            raise SystemExit(f"checkpoint 不存在: {path}")
        print(f">>> 加载权重 {path}")
        t0 = time.perf_counter()
        payload = self.model.load_checkpoint(str(path))
        have = set(self.model.mot.state_dict().keys())
        got = set(payload["mot"].keys()) if "mot" in payload else set()
        missing, unexpected = sorted(have - got), sorted(got - have)
        info = {
            "path": str(path),
            "step": payload.get("step"),
            "mot_total": len(have),
            "mot_loaded": len(have & got),
        }
        del payload
        if missing:
            raise SystemExit(
                f"{path.name}: mot 有 {len(missing)}/{len(have)} 个权重没被加载 —— "
                f"模型结构与 checkpoint 不匹配(task 是否与训练时一致?)\n  例如: {missing[:5]}"
            )
        if unexpected:
            print(f"    ⚠️ checkpoint 里有 {len(unexpected)} 个用不上的键(通常无害)")
        print(f"    权重校验 {info['mot_loaded']}/{info['mot_total']} 键全部匹配,"
              f"耗时 {time.perf_counter() - t0:.1f}s")
        return info

    # ── 文本条件 ────────────────────────────────────────────────────────────
    def _load_text_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        """读 T5 embedding 缓存。**逐字复刻** robot_video_dataset._get_cached_text_context
        以及紧随其后的两行(`context[~mask]=0` + mask 全置 1),否则条件与训练不一致。

        走缓存而不是加载 text encoder:省掉 T5 的显存,且保证与训练用的是同一份 embedding。
        """
        prompt = DEFAULT_PROMPT.format(task=str(self.cfg.instruction))
        ctx_len = self.dcfg["context_len"]
        cache_dir = Path(self.dcfg["text_embedding_cache_dir"])
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        f = cache_dir / f"{h}.t5_len{ctx_len}.wan22ti2v5b.pt"
        if not f.is_file():
            raise SystemExit(
                f"text embedding 缓存缺失: {f}\n"
                f"  instruction = {self.cfg.instruction!r}\n"
                f"  完整 prompt  = {prompt!r}\n"
                "  这条 instruction 必须与训练数据 meta/tasks.jsonl 里的**逐字一致**;\n"
                "  确实要换新指令的话先跑:\n"
                f"    python scripts/precompute_text_embeds.py --dataset-dir <数据集> --out-dir {cache_dir}"
            )
        payload = torch.load(f, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        context[~mask] = 0.0  # 与 wan2.2 行为一致(robot_video_dataset.py:246)
        mask = torch.ones_like(mask)
        print(f">>> 文本条件命中缓存 {f.name}  context={tuple(context.shape)}")
        return context.to(self.device, self.dtype), mask.to(self.device)

    # ── 推理 ────────────────────────────────────────────────────────────────
    def infer_chunk(
        self,
        joint12: np.ndarray,
        grip_frac2: np.ndarray,
        images: dict[str, np.ndarray],
        action_prefix: np.ndarray | None = None,
        delay: int = 0,
    ) -> dict:
        """一次规划。返回物理量 chunk [T,14](action 排布,夹爪是 0~1 分数)及中间量。

        `action_prefix` / `delay` 是 training-time RTC 的动作前缀条件(arXiv:2512.05964):
        前 `delay` 步被硬性 clamp 成上一个 chunk 已承诺的值,模型只生成后缀。
        **这里的 `action_prefix` 必须已经是归一化空间的** —— 上线协议走绝对物理量,
        由 `fastwam_server.py` 在调用本方法之前换算(见 `action_affine`)。
        delay=0 且 action_prefix=None 时走与非 RTC 完全相同的无条件路径。
        """
        t_pre = time.perf_counter()
        torch.set_num_threads(self._nt_pre)  # 拼图是 CPU 小张量,线程多了反而慢 5 倍
        frame = self.pipe.compose_image(images)
        proprio = self.pipe.normalize_proprio(joint12, grip_frac2)
        pre_s = time.perf_counter() - t_pre

        inf = self.cfg.inference
        torch.set_num_threads(self._nt_infer)  # 去噪循环相反,线程多的快
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=None,  # 与 context 互斥,必须给 None
                input_image=frame,
                action_horizon=self.action_horizon,
                proprio=proprio.to(self.device, self.dtype),
                context=self.context,
                context_mask=self.context_mask,
                num_inference_steps=int(inf.num_inference_steps),
                sigma_shift=inf.sigma_shift,
                seed=(None if inf.seed is None else int(inf.seed)),
                rand_device="cpu",
                tiled=False,
                compile_action_infer=bool(inf.compile_action_infer),
                **self._compile_mode_kwargs(inf),
                **self._rtc_kwargs(action_prefix, delay),
            )
        infer_s = time.perf_counter() - t0
        self.stat_infer_s.append(infer_s)

        act_norm = pred["action"].detach().to("cpu", torch.float32)  # [T,14] 归一化
        act_phys = denorm_action(self.processor, act_norm, proprio)  # [T,14] 物理量
        return {
            "action_phys": act_phys.numpy().astype(np.float32),
            "action_norm": act_norm.numpy().astype(np.float32),
            "proprio_norm": proprio.detach().cpu().numpy().astype(np.float32),
            "frame": frame,
            "infer_s": infer_s,
            "pre_s": pre_s,
            "delay": int(delay),
        }

    def _compile_mode_kwargs(self, inf) -> dict:
        """只有模型支持 `compile_mode` 时才传。

        `compile_mode` 是后加的参数；旧版 fastwam.py 的 `infer_action` 没有它，
        无条件传会 TypeError。配置里显式要求了非默认值但模型不支持时必须报错而不是静默忽略。
        """
        want = str(inf.get("compile_mode", "reduce-overhead"))
        if "compile_mode" in self._infer_params:
            return {"compile_mode": want}
        if want != "reduce-overhead":
            raise SystemExit(
                f"配置要求 inference.compile_mode={want!r}，但当前 fastwam.py 的 "
                "infer_action() 不接受 compile_mode 参数 —— 说明 src/ 的代码版本比部署脚本旧。"
            )
        return {}

    def _rtc_kwargs(self, action_prefix, delay: int) -> dict:
        """只有真的要用前缀时才往 infer_action 传 RTC 参数,保证非 RTC 路径调用签名不变。"""
        if action_prefix is None or int(delay) <= 0:
            return {}
        prefix = torch.as_tensor(np.asarray(action_prefix), dtype=torch.float32)
        if prefix.ndim != 2:
            raise ValueError(f"`action_prefix` 必须是 [L,14],得到 {tuple(prefix.shape)}")
        return {"action_prefix": prefix, "delay": int(delay)}


# ══════════════════════════════════════════════ 专属推理线程(CUDA Graphs 的前提)
class InferenceThread:
    """把 compile / 预热 / 每一次推理全部钉在**同一个**线程上。

    这是让 RTC 也能用 `compile_mode=reduce-overhead` 的关键。本机实测的规则见模块
    docstring：CUDA Graph Trees 不要求主线程，只要求「从头到尾同一个线程」。

    用法(server 与单进程部署都走这条路)：

        inf = InferenceThread(build_policy_fn)   # 构建也在该线程上做
        inf.warmup(joint, grip, images, rtc=True)
        res = inf.submit(joint, grip, images).result()

    ⚠️ `build_policy_fn` 在该线程上执行 —— 让模型的首次 CUDA 上下文也落在这里。
    """

    def __init__(self, build_policy_fn):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fastwam-infer")
        self._owner_ident: Optional[int] = None
        self._closed = False

        def _build():
            self._owner_ident = threading.get_ident()
            p = build_policy_fn()
            p._owner_ident = self._owner_ident
            return p

        self.policy: FastWAMPolicy = self._pool.submit(_build).result()

    # ── 线程身份守卫 ────────────────────────────────────────────────────────
    def assert_owner(self) -> None:
        """确认当前就在专属线程上。CUDA Graphs 的正确性完全依赖这一点。"""
        cur = threading.get_ident()
        if cur != self._owner_ident:
            raise RuntimeError(
                f"推理必须始终在专属线程上执行(owner={self._owner_ident}, 当前={cur})。"
                "\n  CUDA Graph Trees 的状态是线程局部的:换线程要么直接 AssertionError,"
                "\n  要么静默丢失 CUDA Graphs 收益。一律经由 InferenceThread.submit()。"
            )

    def _run(self, *args, **kw) -> dict:
        self.assert_owner()
        return self.policy.infer_chunk(*args, **kw)

    def submit(self, joint12, grip2, images, action_prefix=None, delay: int = 0) -> "Future[dict]":
        """异步发起一次推理。返回 Future —— 控制环拿它 `.done()` 轮询。"""
        if self._closed:
            raise RuntimeError("InferenceThread 已关闭")
        return self._pool.submit(self._run, joint12, grip2, images, action_prefix, delay)

    def run_sync(self, joint12, grip2, images, action_prefix=None, delay: int = 0) -> dict:
        """同步跑一次(仍在专属线程上执行,只是调用方阻塞等它)。"""
        return self.submit(joint12, grip2, images, action_prefix, delay).result()

    def call(self, fn, *args, **kw):
        """在专属线程上执行任意可调用对象(benchmark / selfcheck 等要用)。"""
        return self._pool.submit(fn, *args, **kw).result()

    # ── 预热 ────────────────────────────────────────────────────────────────
    def warmup(self, joint12, grip2, images, rtc: bool = False,
               n: int = 2, probe_delay: int = 5) -> None:
        """在服务/控制环**之外**先跑几次,把 torch.compile 的编译成本付掉。

        为什么必须这样(INFER_LATENCY_DEBUG 坑 2/3/4，都是真机换来的)：
          * `torch.compile` 惰性编译，第一次调用才编，真机实测 **17~84 s**。
            若这发生在控制环里，机器人连续几百步拿不到新 chunk，等 chunk 到达时它
            基于 30 秒前的观测 —— 首步跳变可达 35 度直接触发安全层 abort。
          * **两条形状不同的路径 = 两张独立的图，各要编译一次**：
                delay == 0  -> timestep 形状 [1]      图 A(冷启动第一个 chunk)
                delay >  0  -> timestep 形状 [1,H]    图 B(前缀条件)
            只热图 A 的话，控制环里第一次带前缀发起时会在**实时环内**编译图 B(约 41 s)，
            表现为「发起 N 次 / 落地 0 次」。
          * `delay` 的**数值**不影响图(timestep 形状恒为 [1,H])，所以任意一个 d>0 即可 ——
            两次预热搞定，不是 12 次。
        """
        if not bool(self.policy.cfg.inference.compile_action_infer):
            return
        if self.policy._warmed_up:
            return

        imgs = {k: np.asarray(v).copy() for k, v in images.items()}
        plans: list[tuple[str, Any, int]] = [("delay=0 (1-D timestep)", None, 0)] * n
        if rtc:
            d = max(1, int(probe_delay))
            plans += [(f"delay={d} (2-D timestep)", np.zeros((d, 14), dtype=np.float32), d)] * 2

        print(f">>> 预热推理 {len(plans)} 次(在专属线程上付掉 torch.compile 的编译成本)...")
        dt_ms = 0.0
        for i, (label, pref, d) in enumerate(plans):
            t0 = time.perf_counter()
            self.run_sync(joint12, grip2, imgs, pref, d)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"    {i + 1}/{len(plans)}  {label:26} {dt_ms:7.0f} ms"
                  + ("  <- 含编译" if dt_ms > 1000 else ""))
        if dt_ms > 1000:
            print(f"    ⚠️ 最后一次预热仍耗时 {dt_ms:.0f} ms —— 可能还有未预热的计算图。"
                  "上真机前请确认落地次数正常，否则考虑 compile_action_infer=false。")

        # 预热的耗时不该污染延迟预估的实测窗口
        self.policy.stat_infer_s.clear()
        self.policy._warmed_up = True
        print("    预热完成，实测延迟窗口已重置")

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False)
