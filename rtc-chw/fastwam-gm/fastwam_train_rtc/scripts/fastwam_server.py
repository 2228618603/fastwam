"""FastWAM 推理服务(GPU 侧)—— 把推理钉在一个专属线程上，让 RTC 也能用 CUDA Graphs。

    ┌──────────────────────────┐   TCP + msgpack   ┌───────────────────────────┐
    │ fastwam_server.py  :8900 │ ◄───────────────► │ fastwam_client.py         │
    │ · 模型 / 预处理 / 反归一化 │  {"cmd":"infer"}  │ · 控制环 + 安全层 + RTC 调度│
    │ · 专属推理线程(CUDA Graph)│  {"action":[T,14]}│ · 无 torch                │
    └──────────────────────────┘                   └──────────┬────────────────┘
                                                              │ TCP + msgpack
                                                   ┌──────────▼────────────────┐
                                                   │ openpi RobotArmService    │
                                                   │ :9900  ROS2 + Piper SDK   │
                                                   └───────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                本服务存在的唯一理由：让 RTC 用上 CUDA Graphs
═══════════════════════════════════════════════════════════════════════════════
`deploy_real.py` 的 `run_async` 把推理丢进 ThreadPoolExecutor，而 CUDA Graph Trees 的状态
是**线程局部**的，于是它只能退到 `compile_mode=default`，白扔掉 2.4~3.2 倍加速
（那条路径只有 32 个 action token，瓶颈在 kernel launch 而非算力，CUDA Graphs 正是对症的）。

本机实测(torch 2.7.1+cu128)确认的真实规则：

    ✅ 专属线程首次 compile+record，之后一直用它       正常，最快(实测 0.188 ms/iter)
    ❌ 主线程先 record，再换另一个线程 record 新图     AssertionError(cudagraph_trees.py:325)
    ⚠️ 主线程先 record，工作线程 replay 同一张已录图   不报错，但主线程后续 0.355->0.797 ms

即 **CUDA Graph Trees 不要求「主线程」，只要求「从头到尾同一个线程」**。所以进程一拆，
推理独占一个线程（`InferenceThread`），`reduce-overhead` 就能用了 —— 这才是拆进程的收益，
不是"跨机器"。附带好处：server 重启不碰机器人。

⚠️ 注意别照抄 GWP 的 `deploy/robot_server.py`：它在**主线程** warmup、再交给 pool serve，
   正好落在上表第三行(不报错但劣化)；一旦控制环里出现需要新录的图(RTC 的 delay=0 与
   delay>0 是两张不同形状的图)就是硬 AssertionError。本服务的 warmup 走 InferenceThread。

线协议(与 openpi RobotArmService 同一套编解码，复用 deploy_robot_client.py)
--------------------------------------------------------------------------
帧格式:4 字节大端无符号长度前缀 + msgpack-numpy 载荷。**零新依赖** ——
不用 websockets(fastwam env 里没装)，也不用第二套 msgpack-numpy 实现。

    client -> server                                    server -> client
    {"cmd":"ping"}                                      {"info": {...}}
    {"cmd":"infer", "state":(14,)f32,                   {"action":[T,14]f32, "infer_s":f,
                    "images":{...JPEG bytes or HWC},     "pre_s":f, "delay":int}
                    "action_prefix":(d,14)f32 可选,
                    "delay":int 可选}
    任何异常                                             {"error": str}

`state` 是**训练序**(见 layout.STATE_LAYOUT_DOC)，`action` 是**动作序**(6+1+6+1)，两者
排布不同，别混 —— 这是 deploy_real.py 契约 1。

`images` 直接转发 openpi service 给的 **JPEG 字节流**(q95，已中心裁到 480x640)。
故意不学 GWP 在客户端预缩图：
  * openpi 给的本来就是 JPEG(三路约 180 KB)，GWP 那个 2.64 MB->0.35 MB 的问题不存在；
  * `selfcheck` 保证的「预处理与训练逐元素一致」只对 server 侧这一份代码成立，
    客户端插一次 resize 就破坏它(GWP 自己也警告 cv2 与 PIL BILINEAR 最大差 96)；
  * 顺带把 JPEG 解码也从控制环挪到了 server，控制环更轻。

RTC 前缀走**绝对物理量**(弧度 / 夹爪 0~1)
----------------------------------------
客户端不必持有 `dataset_stats.json` —— 也就不可能用错一份统计量。server 收到后换算回
归一化空间。这个换算成立的前提是 `use_stepwise_action_norm=False`(本数据集正是如此)，
此时 norm<->phys 是与 proprio 无关的固定逐元素仿射。启动时会做一次往返自检把这个前提钉死
（实测 phys->norm->phys 1.2e-07、norm->phys->norm 4.8e-07）。

用法
----
    source env.sh

    # 0) 无 GPU 也能跑通链路：假后端(随机但形状/单位/协议全真)
    python scripts/fastwam_server.py --mock --port 8900

    # 1) 自检：加载模型、专属线程上推一次、打印形状与限位
    python scripts/fastwam_server.py --config configs/deploy/agilex_real_rtc.yaml --self-test

    # 2) 正式起服务(RTC + CUDA Graphs)
    python scripts/fastwam_server.py --config configs/deploy/agilex_real_rtc.yaml \
        --port 8900 --compile-mode reduce-overhead --warmup-rtc

    # 3) 实测延迟(在专属线程上，所以 reduce-overhead 的数字是**真能拿到**的)
    python scripts/fastwam_server.py --config configs/deploy/agilex_real_rtc.yaml \
        --benchmark --steps-list 5,10 --compile-mode both
"""

from __future__ import annotations

import argparse
import contextlib
import io
import socket
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 deploy_robot_client 可导入

from deploy_robot_client import Packer, _recv, _send, unpackb  # noqa: E402  复用同一套编解码

from fastwam.deploy.layout import (  # noqa: E402
    ACT_GRIP_IDX,
    ACT_JOINT_IDX,
    ACTION_LAYOUT_DOC,
    STATE_LAYOUT_DOC,
    split_state14,
)

DEFAULT_PORT = 8900


# ══════════════════════════════════════════════════════════════════ 图像解码
def decode_images(images: dict) -> dict[str, np.ndarray]:
    """{相机名: JPEG 字节 或 HWC uint8} -> {相机名: HWC uint8 RGB}。

    ⚠️ **用 PIL 解码时不要再翻通道**。openpi service 那边做的是
    `cv2.imencode(".jpg", rgb[:, :, ::-1])`：cv2 把入参当 BGR 处理，而入参正好是真实图像的
    BGR，所以落盘的 JPEG 颜色本身是对的。于是 `PIL.Image.open()` 直接给出 **RGB**。
    翻错了模型看到的是蓝红互换的世界，而且**不报任何错**，只是动作全乱。
    """
    from PIL import Image

    out: dict[str, np.ndarray] = {}
    for name, raw in images.items():
        if raw is None:
            raise ValueError(f"相机 '{name}' 没有数据(service 端 ROS2 topic 没上来?)")
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
            out[name] = np.ascontiguousarray(arr)  # 已是 HWC RGB(fake service / 直传)
            continue
        buf = arr.tobytes() if isinstance(arr, np.ndarray) else bytes(raw)
        with Image.open(io.BytesIO(buf)) as im:
            # np.array(copy) 而非 asarray:PIL 的 buffer 只读，torch.from_numpy 会告警
            out[name] = np.array(im.convert("RGB"), dtype=np.uint8)
    return out


# ══════════════════════════════════════════════════════════════ 显存 preflight
# 最小部署集实测约 13.4 GB(mot 12.0 + VAE 1.4),加上推理激活值峰值约 15.1 GB。
MODEL_FOOTPRINT_GB = 15.5


def preflight_gpu(device: str) -> None:
    """加载模型**之前**先看一眼显存,不够就说清是谁占着,而不是等 hydra 抛 OOM。

    为什么值得单独做:模型加载到一半才 OOM 时,报错埋在
    `hydra.errors.InstantiationException -> torch.OutOfMemoryError` 里,真正的信息
    ("另一个进程占了 13.57 GiB")在第 12 行的一句话中间,很容易被当成"显卡不够大"。
    实际上 4090 的 24 GB 对本模型是够的(实测峰值 15.1 GB)—— 最常见的原因是
    **上一次跑崩的 server 没退干净**,那个进程还占着整整一份模型。
    """
    if not device.startswith("cuda"):
        return
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return
        idx = int(device.split(":")[1]) if ":" in device else 0
        free_b, total_b = _t.cuda.mem_get_info(idx)
    except Exception:
        return          # 探测本身不该阻断启动
    free, total = free_b / 1024**3, total_b / 1024**3
    print(f">>> 显存 preflight: {device} 空闲 {free:.1f} / {total:.1f} GiB "
          f"(本模型约需 {MODEL_FOOTPRINT_GB} GiB)")
    if free >= MODEL_FOOTPRINT_GB:
        return

    others = _nvidia_smi_procs(idx)
    msg = [f"{device} 只剩 {free:.1f} GiB 空闲,但本模型约需 {MODEL_FOOTPRINT_GB} GiB。"]
    if others:
        msg.append("该卡上的进程:")
        msg += [f"    PID {p:<8} {m:>7.1f} GiB  {c}" for p, m, c in others]
        big = [p for p, m, c in others if m > 8 and "fastwam" in c.lower()]
        if big:
            msg.append("")
            msg.append(f"  ⚠️ PID {big} 看起来是**上一次没退干净的 FastWAM 进程**"
                       "(占着整整一份模型)。先确认它不是你正在用的,再:")
            msg.append(f"       kill {' '.join(str(p) for p in big)}")
    else:
        msg.append("  (拿不到进程列表,手动看: nvidia-smi)")
    msg.append("")
    msg.append("  其他办法:换一张卡 --device cuda:1;或先跑 --mock 验证链路(不占显存)。")
    raise SystemExit("!! " + "\n".join(msg))


def _nvidia_smi_procs(idx: int) -> list[tuple[int, float, str]]:
    """-> [(pid, 显存 GiB, 命令行简写)]。拿不到就返回空列表。"""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            capture_output=True, text=True, timeout=5, check=True).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        try:
            pid_s, mem_s = (x.strip() for x in line.split(","))
            pid, mem = int(pid_s), float(mem_s) / 1024.0
        except ValueError:
            continue
        cmd = ""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()[:70]
        except OSError:
            cmd = "(进程已不在本机 /proc 中)"
        rows.append((pid, mem, cmd))
    return rows


# ══════════════════════════════════════════════════════════════════ 真后端
class TorchBackend:
    """真模型后端。构建/预热/推理**全部**在 InferenceThread 的专属线程上。"""

    name = "torch"

    def __init__(self, cfg, compile_mode: str | None, warmup: int, warmup_rtc: bool):
        from fastwam.deploy.policy import FastWAMPolicy, InferenceThread, action_affine

        if compile_mode:
            cfg.inference.compile_mode = compile_mode
            cfg.inference.compile_action_infer = True
        self.cfg = cfg
        self.rtc_capable = bool(cfg.get("rtc", {}).get("enabled", False))

        preflight_gpu(str(cfg.inference.device))

        dcfg = task_data_cfg(cfg)
        self.dcfg = dcfg
        # ★ 模型也在专属线程上构建，让首次 CUDA 上下文与之后的 CUDA Graphs 同线程
        self.inf = InferenceThread(lambda: FastWAMPolicy(cfg, dcfg))
        self.policy = self.inf.policy

        # RTC 前缀的线上协议是绝对物理量 -> 这里换算回归一化空间需要仿射系数
        self._scale, self._offset = self.inf.call(action_affine, self.policy.processor)
        self._verify_affine()

        stats = self.policy.stats
        self.action_min = stats["action"]["default"]["global_min"].numpy().astype(np.float32)
        self.action_max = stats["action"]["default"]["global_max"].numpy().astype(np.float32)

        # 模型是否真的用前缀条件训练过 —— client 据此硬拒错误组合。
        model_rtc = getattr(self.policy.model, "rtc", None)
        self.rtc_trained = bool(model_rtc is not None and getattr(model_rtc, "enabled", False))

        if warmup > 0:
            j, g, imgs = self._synthetic_obs()
            with self._identity_cam_alias():      # 合成观测用训练名;退出后恢复真实改名映射
                self.inf.warmup(j, g, imgs, rtc=self.rtc_trained and warmup_rtc, n=int(warmup),
                                probe_delay=int(cfg.get("rtc", {}).get("fixed_delay", 5) or 5))
        # 预热不该有任何持久副作用:确认改名映射还在(它错了不报错,只是所有请求都失败)
        self._assert_cam_alias_intact()

    def _verify_affine(self) -> None:
        """把「norm<->phys 是固定仿射」这个协议前提钉死，而不是假设。"""
        rng = np.random.default_rng(0)
        phys = (self.policy.stats["action"]["default"]["global_mean"].numpy()
                + rng.normal(0, 0.5, (16, 14)).astype(np.float32)
                * self.policy.stats["action"]["default"]["global_std"].numpy())
        norm = phys * self._scale + self._offset
        back = (norm - self._offset) / self._scale
        err = float(np.abs(back - phys).max())
        if err > 1e-4:
            raise SystemExit(
                f"action 归一化往返误差 {err:.3e} 太大 —— 「RTC 前缀走绝对物理量」的协议前提不成立。"
                "\n  检查 use_stepwise_action_norm 是否为 False。"
            )
        print(f">>> 归一化仿射自检通过(phys->norm->phys max|Δ| = {err:.2e})")

    def _synthetic_obs(self):
        """合成观测(预热/自检用)。键用**训练名**,所以调用方要临时走恒等映射。

        ⚠️ 不要在这里改 `self.policy.pipe.cam_alias` —— 那是**进程级共享状态**,
        改了之后所有真实请求都会丢掉 `cam_high <- cam_top` 的改名,报
        "观测里没有相机 'cam_high'"。用 `_identity_cam_alias()` 上下文管理器,
        它保证恢复。
        """
        rng = np.random.default_rng(0)
        imgs = {}
        for meta in self.dcfg["shape_meta"]["images"]:
            rh, rw = meta["raw_shape"][1], meta["raw_shape"][2]
            imgs[meta["key"]] = rng.integers(0, 256, size=(rh, rw, 3), dtype=np.uint8)
        j = self.policy.stats["state"]["joint"]["global_mean"].numpy().astype(np.float32)
        g = self.policy.stats["state"]["gripper_position"]["global_mean"].numpy().astype(np.float32)
        return j, g, imgs

    @contextlib.contextmanager
    def _identity_cam_alias(self):
        """临时把相机改名映射置空(合成观测的键已经是训练名),退出时**必定**恢复。"""
        saved = self.policy.pipe.cam_alias
        self.policy.pipe.cam_alias = {}
        try:
            yield
        finally:
            self.policy.pipe.cam_alias = saved

    def _assert_cam_alias_intact(self) -> None:
        """确认相机改名映射与配置一致。

        这条断言是真机换来的:`_synthetic_obs` 曾经直接把 `cam_alias` 置空而不恢复,
        于是预热之后**每一个真实请求**都报 `观测里没有相机 'cam_high'` —— 而 preflight
        全绿、self-test 也过(它们用的都是训练名的合成图),问题只在接上真机那一刻暴露。
        """
        want = dict(OmegaConf.to_container(self.cfg.robot.cameras, resolve=True))
        got = dict(self.policy.pipe.cam_alias)
        if got != want:
            raise SystemExit(
                f"内部错误:相机改名映射被破坏。\n  期望 {want}\n  实际 {got}\n"
                "  (改名错了不会在合成输入上报错,只有接上真机才会 —— 所以这里查死)"
            )
        print(f">>> 相机改名映射 {want}")

    def phys_prefix_to_norm(self, prefix_phys: np.ndarray) -> np.ndarray:
        """绝对物理动作前缀 -> 归一化空间(模型要的)。"""
        return np.asarray(prefix_phys, dtype=np.float32) * self._scale + self._offset

    def infer(self, images: dict, state14: np.ndarray,
              action_prefix: np.ndarray | None = None, delay: int = 0) -> dict:
        joint, grip = split_state14(state14)
        imgs = decode_images(images)
        pfx = None if action_prefix is None or delay <= 0 else self.phys_prefix_to_norm(action_prefix)
        res = self.inf.run_sync(joint, grip, imgs, pfx, int(delay))
        return {
            "action": res["action_phys"],
            "infer_s": res["infer_s"],
            "pre_s": res["pre_s"],
            "delay": res["delay"],
        }

    def info(self) -> dict:
        cfg = self.cfg
        return {
            "backend": self.name,
            "action_horizon": int(self.policy.action_horizon),
            "num_inference_steps": int(cfg.inference.num_inference_steps),
            "compile_action_infer": bool(cfg.inference.compile_action_infer),
            "compile_mode": (str(cfg.inference.get("compile_mode"))
                             if cfg.inference.compile_action_infer else None),
            "torch_threads_pre": int(self.policy._nt_pre),
            "torch_threads_infer": int(self.policy._nt_infer),
            "rtc_trained": self.rtc_trained,
            "rtc_max_delay": int(cfg.get("rtc", {}).get("max_delay", 11) or 11),
            "checkpoint": str(cfg.checkpoint),
            "weight": self.policy.weight_info,
            "instruction": str(cfg.instruction),
            "gripper_travel_m": float(cfg.units.gripper_travel_m),
            "action_limits": {"min": self.action_min, "max": self.action_max},
            "cameras": dict(cfg.robot.cameras),
        }

    def close(self):
        self.inf.close()


# ══════════════════════════════════════════════════════════════════ 假后端
class MockBackend:
    """无 GPU 也能验证整条链路：形状、单位、排布、协议全真，只有动作是编的。

    专门为「开发机 GPU 被占满」和「客户端联调」准备。动作从 state 出发做一条平滑的小幅
    正弦，**保证不触发安全层**(步长远小于 max_delta)，这样能验证控制环与钳位逻辑本身，
    而不是一直在看 abort。
    """

    name = "mock"

    def __init__(self, action_horizon: int = 32, travel_m: float = 0.07,
                 infer_s: float = 0.13, rtc_trained: bool = True):
        self.h = int(action_horizon)
        self.travel_m = float(travel_m)
        self.fake_infer_s = float(infer_s)
        self.rtc_trained = bool(rtc_trained)
        # 与训练 action 的 global_min/max 同量级(只为让 client 的限位逻辑有真值可用)
        self.action_min = np.full(14, -3.0, dtype=np.float32)
        self.action_max = np.full(14, 3.0, dtype=np.float32)
        self.action_min[ACT_GRIP_IDX] = 0.0
        self.action_max[ACT_GRIP_IDX] = 1.0
        self._k = 0

    def infer(self, images: dict, state14: np.ndarray,
              action_prefix: np.ndarray | None = None, delay: int = 0) -> dict:
        t0 = time.perf_counter()
        imgs = decode_images(images)          # 真解码，才能验证 JPEG 路径与相机名
        for name, a in imgs.items():
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError(f"相机 '{name}' 形状异常 {a.shape}")
        joint, grip = split_state14(state14)

        chunk = np.zeros((self.h, 14), dtype=np.float32)
        for t in range(self.h):
            j = joint + 0.01 * np.sin((self._k + t) * 0.05 + np.arange(12) * 0.3)
            g = np.clip(grip + 0.01 * np.sin((self._k + t) * 0.04), 0.0, 1.0)
            chunk[t, ACT_JOINT_IDX] = j
            chunk[t, ACT_GRIP_IDX] = g
        self._k += 1

        # RTC:把前缀原样 clamp 回去，**并让后缀从前缀末端接着走**。
        # 只 clamp 不接续的话，chunk 在下标 d 处会出现一个纯属假后端造成的跳变
        # (实测 0.0105 rad vs 别处 0.0005)，客户端算出来的 seam_ratio 就没意义了 ——
        # 而 seam 连续性正是 RTC 要验证的东西。真模型是**条件生成**后缀，天然连续。
        d = int(delay)
        if action_prefix is not None and d > 0:
            pfx = np.asarray(action_prefix, dtype=np.float32).reshape(-1, 14)[:d]
            shift = pfx[-1] - chunk[d - 1]      # 前缀末端 与 本来第 d-1 步 的偏移
            chunk[:d] = pfx
            chunk[d:] += shift                  # 后缀整体平移,保持它自身的平滑步长
            np.clip(chunk[:, ACT_GRIP_IDX], 0.0, 1.0, out=chunk[:, ACT_GRIP_IDX])
        time.sleep(max(0.0, self.fake_infer_s - (time.perf_counter() - t0)))
        return {"action": chunk, "infer_s": time.perf_counter() - t0, "pre_s": 0.0, "delay": d}

    def info(self) -> dict:
        return {
            "backend": self.name,
            "action_horizon": self.h,
            "num_inference_steps": 0,
            "compile_action_infer": False,
            "compile_mode": None,
            "rtc_trained": self.rtc_trained,
            "rtc_max_delay": 11,
            "checkpoint": "(mock)",
            "instruction": "(mock)",
            "gripper_travel_m": self.travel_m,
            "action_limits": {"min": self.action_min, "max": self.action_max},
            "mock_infer_s": self.fake_infer_s,
        }

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════ 协议层
class ActionService:
    """请求校验 + 分发。校验放这里而不是后端里，两个后端共享同一套拒绝规则。"""

    def __init__(self, backend):
        self.backend = backend
        self.n_infer = 0
        self._lock = threading.Lock()   # 一次只服务一个推理请求(GPU 与专属线程都只有一份)

    def info(self) -> dict:
        return {
            "state_layout": STATE_LAYOUT_DOC,
            "action_layout": ACTION_LAYOUT_DOC,
            "control_hz_from_training_fps": 30,
            "infer_count": self.n_infer,
            "wire_images": "JPEG bytes(openpi 原样转发) 或 HWC uint8 RGB",
            "rtc_prefix_space": "absolute physical (rad / gripper 0~1)",
            **self.backend.info(),
        }

    def infer(self, images: dict, state, action_prefix=None, delay: int = 0) -> dict:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14:
            raise ValueError(f"state 必须是 14 维({STATE_LAYOUT_DOC}),得到 {state.shape[0]}")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"state 含 NaN/Inf: {state.tolist()}")
        if not images:
            raise ValueError("请求里没有 images")

        d = int(delay or 0)
        pfx = None
        if action_prefix is not None and d > 0:
            pfx = np.asarray(action_prefix, dtype=np.float32).reshape(-1, 14)
            if pfx.shape[0] < d:
                raise ValueError(f"action_prefix 有 {pfx.shape[0]} 步 < delay={d}")
            if not np.all(np.isfinite(pfx)):
                raise ValueError("action_prefix 含 NaN/Inf")
            h = int(self.backend.info()["action_horizon"])
            if d >= h:
                raise ValueError(f"delay={d} 必须 < action_horizon={h}")
            pfx = pfx[:d]

        with self._lock:
            out = self.backend.infer(images, state, action_prefix=pfx, delay=d)
        chunk = np.asarray(out["action"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise RuntimeError(f"后端返回 {chunk.shape},期望 [T,14]")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("后端产出了 NaN/Inf 动作")
        self.n_infer += 1
        if self.n_infer <= 3 or self.n_infer % 20 == 0:
            print(f"    infer #{self.n_infer} -> {chunk.shape} "
                  f"模型 {out['infer_s'] * 1000:.0f} ms(预处理 {out.get('pre_s', 0) * 1000:.0f} ms)"
                  + (f"  RTC d={d}" if d > 0 else ""))
        return {"action": chunk, "action_horizon": int(chunk.shape[0]),
                "infer_s": float(out["infer_s"]), "pre_s": float(out.get("pre_s", 0.0)),
                "delay": d}


# ══════════════════════════════════════════════════════════════════ TCP 服务
def serve_forever(service: ActionService, host: str, port: int) -> None:
    packer = Packer()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            peer = self.client_address
            print(f">>> 客户端接入 {peer}")
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    try:
                        raw = _recv(self.request)
                    except (EOFError, ConnectionResetError):
                        break
                    try:
                        req = unpackb(raw)
                        cmd = req.get("cmd")
                        if cmd == "ping":
                            reply = {"info": service.info()}
                        elif cmd == "infer":
                            reply = service.infer(
                                images=req.get("images") or {},
                                state=req["state"],
                                action_prefix=req.get("action_prefix"),
                                delay=int(req.get("delay", 0) or 0),
                            )
                        else:
                            reply = {"error": f"unknown cmd: {cmd!r}(支持 ping / infer)"}
                    except Exception as exc:   # 坏请求不杀连接
                        print(f"!! 请求失败: {exc}\n{traceback.format_exc()}")
                        reply = {"error": f"{type(exc).__name__}: {exc}"}
                    _send(self.request, packer.pack(reply))
            finally:
                print(f">>> 客户端断开 {peer}")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as srv:
        print(f"\n>>> FastWAM 推理服务已就绪: tcp://{host}:{port}")
        print(f"    后端={service.backend.name}  "
              f"action_horizon={service.info()['action_horizon']}  "
              f"compile_mode={service.info().get('compile_mode')}")
        print("    客户端: python scripts/fastwam_client.py "
              f"--server tcp://<本机IP>:{port} --endpoint tcp://<机器人IP>:9900\n")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n>>> 收到 Ctrl-C,关闭服务")


# ══════════════════════════════════════════════════════════════════ 子命令
def task_data_cfg(cfg) -> dict:
    """只取几个数据/时间尺度字段,避免过早 compose 整个 hydra。"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    configs_dir = str(cfg.get("configs_dir") or (REPO_ROOT / "configs"))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
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


def load_cfg(args):
    """读部署 YAML(与 deploy_real.py 同一份配置文件),命令行可覆盖。"""
    from omegaconf import OmegaConf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from deploy_real import DEFAULTS, _merge  # 复用同一份默认值,避免两处漂移

    cfg = dict(DEFAULTS)
    if args.config:
        raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if not isinstance(raw, dict):
            raise SystemExit(f"{args.config} 顶层必须是 mapping")
        cfg = _merge(cfg, raw)
    over = {"inference": {}, "runtime": {}}
    if args.task:
        cfg["task"] = args.task
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.norm_stats:
        cfg["norm_stats"] = args.norm_stats
    if args.device:
        over["inference"]["device"] = args.device
    if args.num_inference_steps:
        over["inference"]["num_inference_steps"] = int(args.num_inference_steps)
    if args.torch_threads_pre is not None:
        over["inference"]["torch_threads_pre"] = int(args.torch_threads_pre)
    if args.torch_threads_infer is not None:
        over["inference"]["torch_threads_infer"] = int(args.torch_threads_infer)
    if args.overrides:
        over["runtime"]["overrides"] = list(args.overrides)
    cfg = _merge(cfg, {k: v for k, v in over.items() if v})
    if not cfg.get("checkpoint"):
        raise SystemExit("必须给 --checkpoint 或在 --config 里写 checkpoint")
    return OmegaConf.create(cfg)


def cmd_self_test(service: ActionService) -> int:
    """加载后端 -> 合成观测推一次 -> 打印形状/量程/限位。不联网、不碰机器人。"""
    print("\n=== SELF TEST(合成输入,无机器人,无网络)===")
    info = service.info()
    print(f"  后端         {info['backend']}   compile_mode={info.get('compile_mode')}")
    print(f"  action_horizon {info['action_horizon']}  steps={info.get('num_inference_steps')}")
    print(f"  rtc_trained  {info.get('rtc_trained')}")
    rng = np.random.default_rng(0)
    # ★ 用 server **真实上报**的相机名(即 service 端的名字)造图,而不是硬编码一张改名表 ——
    #   这样 self-test 走的就是真机那条改名路径。硬编码的话配置一改就静默失配,
    #   而这个错误只在接上真机时才暴露(真机实测踩过:报"观测里没有相机 'cam_high'")。
    cams = info.get("cameras")
    src_keys = sorted(dict(cams).values()) if cams else ["cam_top", "cam_left_wrist", "cam_right_wrist"]
    print(f"  相机映射     {dict(cams) if cams else '(mock,无改名)'}  -> 造图用 {src_keys}")
    images = {k: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for k in src_keys}
    # 数据集均值起始姿态,避免全零外推
    state = np.zeros(14, dtype=np.float32)
    state[12:14] = 0.5
    out = service.infer(images, state)
    c = out["action"]
    print(f"  chunk        {c.shape} dtype={c.dtype}  infer {out['infer_s'] * 1000:.0f} ms")
    print(f"  首步         {np.round(c[0], 4).tolist()}")
    print(f"  左关节 min/max {c[:, 0:6].min():+.4f} / {c[:, 0:6].max():+.4f}")
    print(f"  右关节 min/max {c[:, 7:13].min():+.4f} / {c[:, 7:13].max():+.4f}")
    print(f"  夹爪  min/max {c[:, ACT_GRIP_IDX].min():.4f} / {c[:, ACT_GRIP_IDX].max():.4f}  (应在 0~1)")

    if info.get("rtc_trained"):
        d = 5
        pfx = c[6:6 + d].copy()
        out2 = service.infer(images, state, action_prefix=pfx, delay=d)
        err = float(np.abs(out2["action"][:d] - pfx).max())
        # 前缀在归一化空间被 clamp，反归一化回来应当还原成传入的绝对量。
        # 1e-3 rad = 0.06 度，覆盖 bf16 往返(2^-8 相对间隔)后仍足以抓住"前缀没生效"。
        tag = "OK" if err < 1e-3 else "FAIL"
        print(f"  RTC 前缀 d={d}: 返回的前 d 步 vs 传入前缀 max|Δ| = {err:.3e}  [{tag}]")
        if err >= 1e-3:
            print("     ❌ 前缀没被 clamp 住 —— 检查 phys<->norm 换算或 checkpoint 是否 RTC 训练")
            return 1
    print("=== SELF TEST PASSED ===")
    return 0


def cmd_benchmark(cfg, args) -> int:
    """实测延迟并换算成控制步 d。**在专属线程上跑**，所以 reduce-overhead 的数字是真能拿到的。

    这是与 `deploy_real.py benchmark` 的关键差别：那个跑在主线程，给出的 cudagraph 数字在
    旧的 async 架构下拿不到(INFER_LATENCY_DEBUG §4.3)。本服务把推理钉在专属线程上，
    所以这里测到的就是部署时的真实值。
    """
    from fastwam.deploy.policy import FastWAMPolicy, InferenceThread

    dt = 1.0 / float(cfg.control.hz)
    steps_list = [int(x) for x in (args.steps_list or "5,10").split(",")]
    modes = (["default", "reduce-overhead"] if args.compile_mode == "both"
             else [args.compile_mode or "reduce-overhead"])
    n_rep = int(args.repeat)

    dcfg = task_data_cfg(cfg)
    inf = InferenceThread(lambda: FastWAMPolicy(cfg, dcfg))
    policy = inf.policy
    # benchmark 全程用训练名的合成图,且跑完就退出进程 —— 这里置空是安全的
    # (与 TorchBackend 不同:那边之后要服务真实请求,必须用 _identity_cam_alias 恢复)。
    policy.pipe.cam_alias = {}
    rng = np.random.default_rng(0)
    imgs = {m["key"]: rng.integers(0, 256, (m["raw_shape"][1], m["raw_shape"][2], 3), dtype=np.uint8)
            for m in dcfg["shape_meta"]["images"]}
    joint = policy.stats["state"]["joint"]["global_mean"].numpy().astype(np.float32)
    grip = policy.stats["state"]["gripper_position"]["global_mean"].numpy().astype(np.float32)

    print(">>> 全局预热 3 次(吃掉一次性初始化,避免第一组虚高)...")
    for _ in range(3):
        inf.run_sync(joint, grip, imgs)

    rows = []
    try:
        for mode in modes:
            cfg.inference.compile_action_infer = True
            cfg.inference.compile_mode = mode
            print(f"\n>>> compile_mode={mode}(专属线程,与部署时同一条路径)")
            for ns in steps_list:
                cfg.inference.num_inference_steps = ns
                # 每个 (mode, steps) 组合都要重新付一次编译；丢掉前 2 次
                lat, pre = [], []
                for i in range(n_rep + 2):
                    r = inf.run_sync(joint, grip, imgs)
                    if i >= 2:
                        lat.append(r["infer_s"])
                        pre.append(r["pre_s"])
                a, p = np.array(lat), np.array(pre)
                tot = a + p
                d = int(np.ceil(tot.mean() / dt))
                rows.append((mode, ns, a.mean(), p.mean(), tot.mean(), d))
                print(f"    steps={ns:3d}  推理 {a.mean() * 1000:6.1f} ms + 预处理 {p.mean() * 1000:5.1f} ms"
                      f" = {tot.mean() * 1000:6.1f} ms  ->  d={d:2d} 个控制步 @{cfg.control.hz:g}Hz")
    finally:
        inf.close()

    print("\n>>> 结论(d = RTC 前缀长度 = 切入 chunk 的第几步)")
    for mode, ns, a, p, tot, d in rows:
        verdict = ("✅ 落在训练分布中央" if d <= 8 else
                   "可用,建议 rtc.latency_percentile 提到 0.95" if d <= 11 else
                   "❌ d>11 超过训练 max_delay 上限,前缀会持续 overrun")
        print(f"    mode={mode:16} steps={ns:3d}  d={d:2d}  {verdict}")
    print("\n    ⚠️ 改 num_inference_steps 会改动作分布,上真机前先离线复测:")
    print(f"       python scripts/offline_eval.py --config configs/eval/final_A_newval71.yaml \\")
    print(f"           --num-inference-steps {steps_list[0]} --out-dir /tmp/eval_ns{steps_list[0]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FastWAM 推理服务(GPU 侧,推理钉在专属线程 -> RTC 可用 CUDA Graphs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="联调顺序: --mock(无 GPU) -> --self-test -> --benchmark -> 正式起服务",
    )
    ap.add_argument("--config", help="部署 YAML,如 configs/deploy/agilex_real_rtc.yaml")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--task")
    ap.add_argument("--checkpoint")
    ap.add_argument("--norm-stats")
    ap.add_argument("--device")
    ap.add_argument("--num-inference-steps", type=int)
    ap.add_argument("--overrides", nargs="*", help="额外 hydra 覆盖")
    ap.add_argument(
        "--compile-mode", choices=["default", "reduce-overhead", "max-autotune", "both"], default=None,
        help="覆盖 inference.compile_mode 并开启 compile。**RTC 推荐 reduce-overhead** —— "
             "本服务把推理钉在专属线程上,所以 CUDA Graph Trees 可用(实测比 default 快 2.4 倍)。"
             "both 只对 --benchmark 有意义。")
    ap.add_argument("--torch-threads-pre", type=int,
                    help="预处理阶段 CPU 线程数(0=不限)。80 核 A800 实测 16 最优,4090/20 核实测 8")
    ap.add_argument("--torch-threads-infer", type=int,
                    help="推理阶段 CPU 线程数(0=不限)。**与 client 同机时必须限制** —— "
                         "不限的话推理占满全部核,30Hz 控制环抢不到 CPU(实测节拍超时 25%%->0%%,"
                         "有效频率 23.7->27.5 Hz)。20 核机器给 4~6。")
    ap.add_argument("--warmup", type=int, default=2,
                    help="启动时先跑几次推理付掉编译成本(真机实测首次可达 35s)。0=不预热")
    ap.add_argument("--warmup-rtc", action="store_true",
                    help="额外预热带前缀的路径。**开 RTC 时必加** —— delay=0 与 delay>0 是"
                         "两张不同形状的图,各要编译一次;只热前者会让第一次带前缀的请求"
                         "在实时环里触发 41s 编译(INFER_LATENCY_DEBUG 坑 4)")
    ap.add_argument("--mock", action="store_true",
                    help="假后端:不加载模型、不需要 GPU,但形状/单位/排布/协议全真。"
                         "用来在无 GPU 机器上联调客户端与控制环")
    ap.add_argument("--mock-infer-s", type=float, default=0.13,
                    help="假后端模拟的推理耗时(秒)。0.13 ≈ 真机 reduce-overhead 实测值")
    ap.add_argument("--self-test", action="store_true", help="加载后端、推一次、打印形状后退出")
    ap.add_argument("--benchmark", action="store_true", help="实测延迟并换算 d,不起服务")
    ap.add_argument("--steps-list", default="5,10", help="--benchmark 要测的去噪步数")
    ap.add_argument("--repeat", type=int, default=5, help="--benchmark 每组有效测量次数")
    args = ap.parse_args()

    if args.mock:
        if args.benchmark:
            raise SystemExit("--benchmark 需要真模型,不能配 --mock")
        backend = MockBackend(travel_m=0.07, infer_s=float(args.mock_infer_s))
        print(">>> ⚠️ MOCK 后端:动作是编的,**只能用于链路联调,绝不能上真机做任务**")
    else:
        cfg = load_cfg(args)
        if args.benchmark:
            return cmd_benchmark(cfg, args)
        backend = TorchBackend(cfg, compile_mode=(None if args.compile_mode == "both" else args.compile_mode),
                               warmup=int(args.warmup), warmup_rtc=bool(args.warmup_rtc))

    service = ActionService(backend)
    try:
        if args.self_test:
            return cmd_self_test(service)
        serve_forever(service, args.host, int(args.port))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
