#!/usr/bin/env python3
"""Giga World Policy 0.5 真机部署推理服务 (GPU 服务器端).

运行在 GPU 服务器上, 加载微调好的 Giga-World-Policy 模型, 通过 websocket + msgpack
提供动作预测服务。工控机端只需要 websockets + msgpack, 不需要 torch/CUDA。

协议沿用此前已验证的 FastWAM websocket/msgpack 形式：
    Request:  {"ping": True}                                  -> {"ok": True, "info": {...}}
              {"reset": True}                                 -> {"ok": True}
              {"obs": {"top": <HWC uint8 RGB>,
                       "left_wrist": <HWC uint8 RGB>,
                       "right_wrist": <HWC uint8 RGB>},
               "state": <14 floats, 训练序>,
               "instruction": "<可选, 仅 --use-t5 时生效>"}
              -> {"ok": True, "action": <[T,14] float32 绝对关节弧度+0~1爪>,
                  "action_horizon": T, "infer_s": float}

状态布局 (与 action 对齐):
    state[0:6]=左臂关节(rad), state[6]=左爪(0~1),
    state[7:13]=右臂关节(rad), state[13]=右爪(0~1)

动作布局 (输出序, 与 robot_client 一致):
    action[0:6]=左臂关节(rad), action[6]=左爪(0~1),
    action[7:13]=右臂关节(rad), action[13]=右爪(0~1)

后端:
    --backend python (默认): 加载 Giga 微调 checkpoint + 固定 T5 token (省 ~10.5GB 显存,
                             与训练/评估同款路径, 精度最高)
    --backend cpp:           仅为以后重新生成并验证 aligned GGUF 保留。旧 GGUF 属于 legacy
                             布局；当前 cpp 也不支持 RTC 的 action_prefix + delay 条件推理。

用法:
    python robot_server.py --self-test
    python robot_server.py --host 0.0.0.0 --port 8000            # Python 后端 (固定 token)
    # RTC 条件推理当前使用 Python；不要把旧 GGUF 传给本目录代码。
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/mnt/data/zzd/giga-world-policy")
DATA_ROOT = MODEL_ROOT / "geek_data"
for _p in (PROJECT_ROOT,
           PROJECT_ROOT / "third_party" / "giga-datasets",
           PROJECT_ROOT / "third_party" / "giga-models",
           PROJECT_ROOT / "third_party" / "giga-train"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robot_server")

# ------------------------------------------------------------------ 默认路径
DEFAULT_BASE_MODEL = str(MODEL_ROOT / "Wan2.2-TI2V-5B-Diffusers")
DEFAULT_CKPT = (
    "/mnt/data/models/zzd/checkpoint/"
    "giga_470_aligned_l6g_r6g_bs128_gpu8_b16g1_100k/models/"
    "checkpoint_epoch_13_step_100000/transformer_ema"
)
DEFAULT_NORM_STATS = str(DATA_ROOT / "norm_stats_aligned_l6g_r6g.json")
DEFAULT_TOKEN_FILE = str(PROJECT_ROOT / "prompt_tokens" / "fixed_task_token.pt")
DEFAULT_GGUF = (
    "/mnt/data/zzd/giga-world-policy/weights/"
    "giga_470_aligned_l6g_r6g_step100000.gguf"
)
DEFAULT_WAM_LIB = "/home/zzd/project/wam.cpp/build-cuda/libwam_c_api.so"
DEFAULT_TOKENIZER = str(MODEL_ROOT / "Wan2.2-TI2V-5B-Diffusers" / "tokenizer")

# 训练/评估同款配置
DST = (320, 384)          # (W, H) T-shape 复合图
ACTION_CHUNK = 48         # 动作预测时域 (48 帧 @30Hz = 1.6s)
NUM_FRAMES = 5
NUM_INFERENCE_STEPS = 10
ACTION_DIM = 16           # 模型动作维度 (14 有效 + 2 padding)
CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
WIRE_TARGET_SIZES = {"top": (DST[0], DST[1] // 2),
                     "left_wrist": (DST[0] // 2, DST[1] // 2),
                     "right_wrist": (DST[0] // 2, DST[1] // 2)}
# 数据集任务完整原文 (meta/tasks.parquet), 固定 token 与其逐位一致
INSTRUCTION = ("Alternately use the left arm and the right arm to pick up the goods "
               "from the nearby box and place them in the distant box, until the "
               "nearby box is empty.")

STATE_LAYOUT_DOC = (
    "state[0:6]=left joints(rad), state[6]=left gripper(0~1), "
    "state[7:13]=right joints(rad), state[13]=right gripper(0~1)  [6+1+6+1]"
)
ACTION_LAYOUT_DOC = (
    "action[0:6]=left joints(rad), action[6]=left gripper(0~1), "
    "action[7:13]=right joints(rad), action[13]=right gripper(0~1)  [6+1+6+1]"
)


# ================================================================ 图像预处理
def process_images(input_images, dst_width, dst_height):
    """训练同款: 保比例 resize (BILINEAR) + center crop."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F

    height = input_images.height
    width = input_images.width
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
    return input_images


def compose_tshape(images: Dict[str, np.ndarray]) -> "Image.Image":
    """3 路相机 -> 320x384 T-shape 复合图, 与训练 get_ref_image_3views 完全一致.

    top 在上 (320x192), 左/右腕在下并排 (各 160x192).
    """
    from PIL import Image

    img_front = Image.fromarray(images["top"], mode="RGB")
    img_left = Image.fromarray(images["left_wrist"], mode="RGB")
    img_right = Image.fromarray(images["right_wrist"], mode="RGB")
    dst_width, dst_height = DST
    top_h = dst_height // 2
    bottom_h = dst_height - top_h
    left_w = dst_width // 2
    right_w = dst_width - left_w
    # 原服务端统一缩图（原分辨率客户端仍兼容）：
    # cam_high = process_images(img_front, dst_width=dst_width, dst_height=top_h)
    # cam_left = process_images(img_left, dst_width=left_w, dst_height=bottom_h)
    # cam_right = process_images(img_right, dst_width=right_w, dst_height=bottom_h)
    # 客户端按完全相同的 PIL 算法先缩图，减少上行；已达标的图不重复处理。
    def fit(img, w, h):
        return img if img.size == (w, h) else process_images(img, w, h)

    cam_high = fit(img_front, dst_width, top_h)
    cam_left = fit(img_left, left_w, bottom_h)
    cam_right = fit(img_right, right_w, bottom_h)
    out = Image.new("RGB", (dst_width, dst_height))
    out.paste(cam_high, (0, 0))
    out.paste(cam_left, (0, top_h))
    out.paste(cam_right, (left_w, top_h))
    return out


# ================================================================ 归一化
def load_norm(path: str) -> Dict[str, Any]:
    with open(path) as f:
        s = json.load(f)["norm_stats"]
    smin = np.asarray(s["observation.state"]["q01"][:14], dtype=np.float32)
    smax = np.asarray(s["observation.state"]["q99"][:14], dtype=np.float32)
    dmin = np.asarray(s["action"]["q01"][:14], dtype=np.float32)
    dmax = np.asarray(s["action"]["q99"][:14], dtype=np.float32)
    return {
        "smin": smin, "smax": smax, "dmin": dmin, "dmax": dmax,
        "srange": np.maximum(smax - smin, 1e-8),
        "drange": np.maximum(dmax - dmin, 1e-8),
        # 关节维训练 target 是 delta (action - state), 爪维是绝对开合
        "delta_mask": np.array([True] * 6 + [False] + [True] * 6 + [False]),
    }


# ================================================================ 后端
class PythonBackend:
    """Giga WAPipeline 后端: 固定 token (默认) 或完整 T5. 与评估同款路径."""

    name = "python"

    def __init__(self, ckpt: str, base_model: str, norm_stats: str, token_file: str,
                 use_t5: bool, device: str, instruction: str,
                 torch_compile_action_stack: bool = False,
                 torch_compile_mode: str = "reduce-overhead",
                 compile_enabled: bool = False, compile_prefix: bool = False,
                 compile_scope: str = "action-blocks"):
        import torch
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from world_action_model.models import CasualWorldActionTransformer_MoT

        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "inference_openloop", str(PROJECT_ROOT / "scripts" / "inference_openloop.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        WAPipeline = _mod.WAPipeline

        self.torch = torch
        self.norm = load_norm(norm_stats)
        self.use_t5 = use_t5
        self.checkpoint = str(ckpt)
        self.norm_stats = str(norm_stats)

        logger.info("Loading Wan VAE ...")
        vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16)
        logger.info("Loading transformer (Giga MoT) from %s ...", ckpt)
        t = CasualWorldActionTransformer_MoT.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
        t.eval()
        t._enable_action_only_prefix_cache = True
        # 原来仅手工编译 action stack，漏了条件缓存和 pipeline mark_step 开关：
        # t._compiled_forward_action_stack_with_prefix_cache = torch.compile(
        #     t.forward_action_stack_with_prefix_cache, mode=torch_compile_mode,
        #     fullgraph=False, dynamic=False)
        # 现在在 pipeline 建立后统一调用 compile_policy_action_blocks。
        sched = FlowMatchEulerDiscreteScheduler(shift=5.0)
        if use_t5:
            pipe = WAPipeline.from_pretrained(base_model, vae=vae, transformer=t,
                                              scheduler=sched, torch_dtype=torch.bfloat16)
        else:
            pipe = WAPipeline.from_pretrained(base_model, vae=vae, transformer=t,
                                              scheduler=sched, text_encoder=None,
                                              tokenizer=None, torch_dtype=torch.bfloat16)
        pipe.to(device=device, dtype=torch.bfloat16)
        self.pipe = pipe
        self.device = device
        self.pipe.set_progress_bar_config(disable=True)
        self.compile_mode = None
        self.compile_prefix = False
        self.compile_scope = None
        if compile_enabled or torch_compile_action_stack or compile_prefix:
            scope = "action-stack" if torch_compile_action_stack else compile_scope
            names = _mod.compile_policy_action_blocks(
                pipe, mode=torch_compile_mode, scope=scope, compile_prefix=compile_prefix)
            self.compile_mode = torch_compile_mode
            self.compile_prefix = bool(compile_prefix)
            self.compile_scope = scope
            logger.info("torch.compile(mode=%s, scope=%s, prefix=%s): %s",
                        self.compile_mode, scope, self.compile_prefix, ", ".join(names))
        self.last_prefix_error = None

        if not use_t5:
            emb = torch.load(token_file, map_location="cpu")
            if isinstance(emb, dict):
                emb = emb.get("t5_embedding", next(iter(emb.values())))
            self.prompt_embeds = torch.zeros(1, 64, 4096, device=device, dtype=torch.bfloat16)
            self.prompt_embeds[0, :emb.shape[0]] = emb.bfloat16().to(device)
            logger.info("Fixed task token loaded (%d tokens)", emb.shape[0])
        self.instruction = instruction

    def infer(self, images: Dict[str, np.ndarray], state14: np.ndarray,
              action_prefix: Optional[np.ndarray] = None, delay: int = 0,
              seed: Optional[int] = None) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        torch.cuda.reset_peak_memory_stats(self.device)
        # 原入口没有 mark_step，prefix 图可能复用上一轮生命周期；不能只在 benchmark 中补。
        if self.compile_mode:
            torch.compiler.cudagraph_mark_step_begin()

        ref = compose_tshape(images)  # PIL 320x384, 与训练同款
        n = self.norm
        ns = ((torch.from_numpy(state14).float() - torch.tensor(n["smin"])) / torch.tensor(n["srange"])) * 2 - 1
        ns = F.pad(ns, (0, ACTION_DIM - 14), value=0.0).unsqueeze(0).to(self.device)
        self.pipe.transformer.reset_action_only_prefix_cache()

        delay = int(delay or 0)
        norm_prefix = None
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        if delay > 0:
            if action_prefix is None:
                raise ValueError("action_prefix is required when delay > 0")
            prefix_abs = np.asarray(action_prefix, dtype=np.float32).reshape(-1, 14)
            if prefix_abs.shape[0] < delay:
                raise ValueError(f"action_prefix has {prefix_abs.shape[0]} steps, delay={delay}")
            if delay >= ACTION_CHUNK:
                raise ValueError(f"delay={delay} must be smaller than action chunk {ACTION_CHUNK}")
            prefix_delta = prefix_abs[:delay] - state14.reshape(1, 14) * n["delta_mask"]
            prefix_norm = ((prefix_delta - n["dmin"]) / n["drange"]) * 2 - 1
            prefix_t = torch.from_numpy(prefix_norm).float()
            prefix_t = F.pad(prefix_t, (0, ACTION_DIM - 14), value=0.0).unsqueeze(0)
            norm_prefix = prefix_t.to(self.device, dtype=torch.bfloat16)

        kwargs = dict(height=DST[1], width=DST[0], action_chunk=ACTION_CHUNK, state=ns,
                      num_frames=NUM_FRAMES, guidance_scale=0.0,
                      num_inference_steps=NUM_INFERENCE_STEPS, image=ref,
                      return_dict=False, action_dim=ACTION_DIM,
                      action_prefix=norm_prefix, delay=delay)
        if seed is not None:  # 仅预热/自测固定噪声；实际部署仍逐次随机抽样。
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        if self.use_t5:
            kwargs["prompt"] = self.instruction
        else:
            kwargs["prompt_embeds"] = self.prompt_embeds

        with torch.no_grad():
            out = self.pipe(**kwargs)
        # 原 torch.cuda.synchronize() 隐含 cuda:0，多卡自测应同步实际推理设备。
        torch.cuda.synchronize(self.device)

        self.last_prefix_error = None
        if seed is not None and delay > 0:
            # 比较模型归一化空间的硬前缀，不把 BF16 量化/反归一化误差误判为接缝错误。
            expected = norm_prefix.to(dtype=out.dtype)
            self.last_prefix_error = float((out[:, :delay] - expected).abs().max().item())
            if self.last_prefix_error != 0.0:
                raise RuntimeError(f"RTC normalized prefix changed: error={self.last_prefix_error}")

        # 反归一化: 关节维 delta + state = 绝对; 爪维模型直接输出绝对开合
        pred = out[..., :14].float().cpu().numpy()[0]
        pred = ((pred + 1) / 2) * n["drange"] + n["dmin"]
        pred_abs = pred + state14 * n["delta_mask"]
        if not np.all(np.isfinite(pred_abs)):
            logger.error("non-finite action produced: %s", pred_abs)
            raise RuntimeError("model produced non-finite actions")
        return pred_abs.astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name, "use_t5": self.use_t5,
                "checkpoint": self.checkpoint, "norm_stats": self.norm_stats,
                "compile_mode": self.compile_mode, "compile_prefix": self.compile_prefix,
                "compile_scope": self.compile_scope, "num_inference_steps": NUM_INFERENCE_STEPS,
                "wire_target_sizes": WIRE_TARGET_SIZES}

    def warmup(self, n: int = 3, delay: int = 8, repeat: int = 5,
               output: Optional[str] = None) -> None:
        """合成观测，先付清编译成本；必须与在线 infer 在同一专用线程调用。"""
        rng = np.random.default_rng(0)
        images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in CAMERA_KEYS}
        # 原 _self_test 的均值把夹爪放在末尾；这里用 aligned [L6,Lg,R6,Rg]。
        state = np.array([-0.2187, 0.1383, -0.4933, -0.2940, 0.8513, 0.2763, 0.5086,
                          0.3524, 0.1806, -0.4801, 0.0619, 0.7892, -0.0104, 0.5402],
                         dtype=np.float32)
        results = {"state": state, "seed": np.array(12345), "delay": np.array(delay)}
        # 两条路径连续预热后再连续计时，避免交替触发 graph record 被误当成稳态。
        for d in (0, delay):
            prefix = np.repeat(state[None, :], d, axis=0) if d else None
            samples = []
            for i in range(n + repeat):
                start = time.perf_counter()
                chunk = self.infer(images, state, action_prefix=prefix, delay=d, seed=12345)
                elapsed = (time.perf_counter() - start) * 1000
                if chunk.shape != (ACTION_CHUNK, 14) or not np.isfinite(chunk).all():
                    raise RuntimeError(f"invalid warmup action chunk: {chunk.shape}")
                if i < n:
                    logger.info("warmup delay=%d %d/%d: %.1f ms (not steady)", d, i + 1, n, elapsed)
                else:
                    samples.append(elapsed)
            median, p95 = np.percentile(samples, [50, 95])
            logger.info("STEADY delay=%d n=%d median=%.1f ms p95=%.1f ms max=%.1f ms prefix_err=%s",
                        d, repeat, median, p95, max(samples), self.last_prefix_error)
            if p95 > 2 * median:
                logger.warning("Timing has outliers; increase WARMUP and check GPU load before using this budget")
            results[f"action_d{d}"] = chunk
            results[f"latency_ms_d{d}"] = np.asarray(samples)
        if output:
            np.savez(output, **results)
            logger.info("fixed-seed self-test samples saved to %s", output)

    def close(self):
        pass


class CppBackend:
    """wam.cpp 后端: external_embedding 固定 token. 输入原始 state, 输出绝对动作."""

    name = "cpp"

    def __init__(self, gguf: str, wam_lib: str, token_file: str, use_t5: bool,
                 tokenizer: str, device: int, instruction: str):
        import torch
        import wam
        self.torch = torch

        if use_t5:
            self.pipe = wam.Pipeline.load(gguf, library=wam_lib,
                runtime_config=wam.RuntimeConfig(backend="cuda", precision="bf16",
                                                 device=device),
                session_config=wam.SessionConfig(random_seed=0), tokenizer=tokenizer)
        else:
            self.pipe = wam.Pipeline.load(gguf, library=wam_lib,
                runtime_config=wam.RuntimeConfig(backend="cuda", precision="bf16",
                                                 device=device,
                                                 language_mode="external_embedding"),
                session_config=wam.SessionConfig(random_seed=0))
            emb = torch.load(token_file, map_location="cpu")
            if isinstance(emb, dict):
                emb = emb.get("t5_embedding", next(iter(emb.values())))
            emb_bf16 = emb.bfloat16()
            # wam.cpp 需要 uint16 (bf16 位模式)
            self.embedding = np.zeros((64, 4096), dtype=np.uint16)
            self.embedding[:emb_bf16.shape[0]] = emb_bf16.view(torch.uint16).numpy()
            self.emb_mask = np.zeros(64, dtype=np.int32)
            self.emb_mask[:emb_bf16.shape[0]] = 1
        self.use_t5 = use_t5
        self.instruction = instruction
        self.roles = ("camera_high", "camera_left_wrist", "camera_right_wrist")
        self.gguf = str(gguf)

    def infer(self, images: Dict[str, np.ndarray], state14: np.ndarray,
              action_prefix: Optional[np.ndarray] = None, delay: int = 0) -> np.ndarray:
        imgs = [{"name": r, "data": images[k]} for r, k in zip(self.roles, CAMERA_KEYS)]
        if self.use_t5:
            r = self.pipe.predict(imgs, state14, instruction=self.instruction)
        else:
            r = self.pipe.predict(imgs, state14, embedding=self.embedding,
                                  embedding_attention_mask=self.emb_mask)
        pred = np.asarray(r.action, dtype=np.float32)[:, :14]
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("wam.cpp produced non-finite actions")
        return pred

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name, "use_t5": self.use_t5, "gguf": self.gguf}

    def close(self):
        self.pipe.close()


# ================================================================ 协议层
class ActionServer:
    """websocket + msgpack 请求处理 (协议与 FastWAM 部署一致)."""

    def __init__(self, backend, default_instruction: str):
        self.backend = backend
        self.instruction = default_instruction
        self._infer_count = 0

    def info(self) -> Dict[str, Any]:
        return {
            "state_layout": STATE_LAYOUT_DOC,
            "action_layout": ACTION_LAYOUT_DOC,
            "camera_keys": list(CAMERA_KEYS),
            "action_horizon": ACTION_CHUNK,
            "control_hz_from_training_fps": 30,
            "instruction": self.instruction,
            "infer_count": self._infer_count,
            **self.backend.info(),
        }

    def infer(self, images: Dict[str, np.ndarray], state: np.ndarray,
              instruction: Optional[str] = None,
              action_prefix: Optional[np.ndarray] = None, delay: int = 0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14:
            raise ValueError(f"state must be 14D ({STATE_LAYOUT_DOC}), got {state.shape[0]}D")
        missing = [k for k in CAMERA_KEYS if k not in images]
        if missing:
            raise ValueError(f"missing camera(s): {missing}; required {list(CAMERA_KEYS)}")
        # 相机帧必须同一时刻采样; 校验尺寸非空
        for k in CAMERA_KEYS:
            img = images[k]
            if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 100 or img.shape[1] < 100:
                raise ValueError(f"camera {k} has unexpected shape {img.shape} (expect HWC RGB)")

        chunk = self.backend.infer(images, state, action_prefix=action_prefix, delay=delay)
        if chunk.shape[1] != 14:
            raise RuntimeError(f"backend returned {chunk.shape[1]}D actions, expected 14D")
        dt = time.perf_counter() - t0
        self._infer_count += 1
        logger.info("infer #%d -> chunk %s in %.3fs delay=%d", self._infer_count, chunk.shape, dt, int(delay or 0))
        return {"action": chunk, "action_horizon": int(chunk.shape[0]), "infer_s": dt}


def _self_test(server: ActionServer) -> int:
    logger.info("=== SELF TEST (synthetic input, no robot, no network) ===")
    rng = np.random.default_rng(0)
    images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in CAMERA_KEYS}
    # 数据集均值起始姿态 (弧度 + 0~1 爪), 避免全零外推
    state = np.array([-0.2187, 0.1383, -0.4933, -0.2940, 0.8513, 0.2763,
                      0.3524, 0.1806, -0.4801, 0.0619, 0.7892, -0.0104,
                      0.5086, 0.5402], dtype=np.float32)
    out = server.infer(images, state)
    chunk = out["action"]
    logger.info("action chunk shape: %s dtype=%s", chunk.shape, chunk.dtype)
    logger.info("first action row: %s", np.round(chunk[0], 4).tolist())
    logger.info("left  joints min/max: %.4f / %.4f", chunk[:, 0:6].min(), chunk[:, 0:6].max())
    logger.info("right joints min/max: %.4f / %.4f", chunk[:, 7:13].min(), chunk[:, 7:13].max())
    logger.info("left  gripper min/max: %.4f / %.4f", chunk[:, 6].min(), chunk[:, 6].max())
    logger.info("right gripper min/max: %.4f / %.4f", chunk[:, 13].min(), chunk[:, 13].max())
    logger.info("infer time: %.3fs", out["infer_s"])
    logger.info("=== SELF TEST PASSED ===")
    return 0


async def _serve(server: ActionServer, host: str, port: int, infer_pool):
    import msgpack
    import msgpack_numpy
    import websockets
    msgpack_numpy.patch()

    async def handler(websocket):
        peer = getattr(websocket, "remote_address", "?")
        logger.info("client connected: %s", peer)
        try:
            async for raw in websocket:
                try:
                    req = msgpack.unpackb(raw, raw=False)
                    if req.get("ping"):
                        resp = {"ok": True, "info": server.info()}
                    elif req.get("reset"):
                        logger.info("reset requested")
                        resp = {"ok": True}
                    elif "obs" in req:
                        # 原 asyncio.to_thread(server.infer, ...) 使用默认线程池。
                        # CUDA Graph 首次录制、预热和在线请求必须走同一专用线程。
                        out = await asyncio.get_running_loop().run_in_executor(
                            infer_pool, functools.partial(server.infer,
                            images=req["obs"],
                            state=np.asarray(req["state"], dtype=np.float32),
                            instruction=req.get("instruction"),
                            action_prefix=req.get("action_prefix"),
                            delay=int(req.get("delay", 0) or 0)))
                        resp = {"ok": True, **out}
                    else:
                        resp = {"ok": False, "error": f"unknown request keys: {list(req)}"}
                except Exception as exc:  # 坏请求不杀连接
                    logger.error("request failed: %s\n%s", exc, traceback.format_exc())
                    resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                await websocket.send(msgpack.packb(resp, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    logger.info("serving on ws://%s:%d", host, port)
    async with websockets.serve(handler, host, port, ping_interval=None, max_size=None):
        await asyncio.Future()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="python", choices=["python", "cpp"],
                   help="推理后端 (默认 python)")
    p.add_argument("--use-t5", action="store_true",
                   help="在线加载完整 T5 编码指令 (默认固定 token, 省 ~10.5GB 显存)")
    p.add_argument("--torch-compile-action-stack", action="store_true",
                   help="Python 后端: torch.compile action stack，首次推理会有较长编译/预热")
    p.add_argument("--torch-compile-mode", default="reduce-overhead",
                   help="Python 后端 torch.compile mode (默认 reduce-overhead)")
    p.add_argument("--compile", action="store_true", help="编译动作去噪（旧 action-stack 参数仍可用）")
    p.add_argument("--compile-prefix", action="store_true", help="同时编译视觉/状态条件 KV cache")
    p.add_argument("--compile-scope", choices=["action-blocks", "action-stack"], default="action-blocks")
    p.add_argument("--torch-threads", type=int, default=max(2, (os.cpu_count() or 8) // 4))
    p.add_argument("--warmup", type=int, default=3, help="每条路径的预热次数（不计入稳态）")
    p.add_argument("--warmup-rtc-delay", type=int, default=8, help="应与客户端 RTC_PREFIX 对齐")
    p.add_argument("--benchmark-repeat", type=int, default=5, help="每条路径的稳态采样数")
    p.add_argument("--self-test-output", help="可选 .npz 文件，保存固定种子动作与延迟供 A/B 比较")
    p.add_argument("--device", default="cuda:0", help="Python 后端 CUDA 设备 (如 cuda:4)")
    p.add_argument("--device-id", type=int, default=0, help="cpp 后端 CUDA 设备号")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Giga 微调 checkpoint")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Wan2.2 diffusers 基模型")
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS, help="norm_stats.json")
    p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE, help="固定 T5 token .pt")
    p.add_argument("--gguf", default=DEFAULT_GGUF, help="cpp 后端 GGUF 权重")
    p.add_argument("--wam-lib", default=DEFAULT_WAM_LIB, help="cpp 后端 wam.cpp 动态库")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="cpp T5 模式 tokenizer")
    p.add_argument("--instruction", default=None,
                   help="任务指令文本 (默认取数据集任务文本)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--self-test", action="store_true",
                   help="加载模型, 合成输入推理一次, 打印形状后退出")
    args = p.parse_args()
    if args.warmup < 1 or args.benchmark_repeat < 1 or not 1 <= args.warmup_rtc_delay < ACTION_CHUNK:
        p.error("warmup/repeat must be positive and RTC delay must be in [1,47]")
    if args.torch_threads < 1:
        p.error("torch-threads must be positive")

    instruction = args.instruction or INSTRUCTION
    if args.backend == "python":
        os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
        import torch
        torch.set_num_threads(args.torch_threads)
        logger.info("CPU inference threads=%d; reserving other cores for ROS/client", args.torch_threads)
        backend = PythonBackend(args.checkpoint, args.base_model, args.norm_stats,
                                args.token_file, args.use_t5, args.device, instruction,
                                args.torch_compile_action_stack,
                                args.torch_compile_mode, args.compile,
                                args.compile_prefix, args.compile_scope)
    else:
        backend = CppBackend(args.gguf, args.wam_lib, args.token_file, args.use_t5,
                             args.tokenizer, args.device_id, instruction)
    server = ActionServer(backend, instruction)

    # 原 self-test 只推一次；原 _serve 使用默认线程池，预热不能覆盖在线线程的图。
    # if args.self_test: return _self_test(server)
    # asyncio.run(_serve(server, args.host, args.port))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gwp-infer") as infer_pool:
        try:
            if backend.name == "python":
                infer_pool.submit(backend.warmup, args.warmup, args.warmup_rtc_delay,
                                  args.benchmark_repeat, args.self_test_output).result()
                if args.self_test:
                    logger.info("=== SELF TEST PASSED (delay=0 and RTC; no robot) ===")
                    return 0
            elif args.self_test:
                return infer_pool.submit(_self_test, server).result()
            asyncio.run(_serve(server, args.host, args.port, infer_pool))
        finally:
            infer_pool.submit(backend.close).result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
