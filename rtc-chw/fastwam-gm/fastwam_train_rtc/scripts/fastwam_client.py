#!/usr/bin/env python
"""FastWAM 真机部署控制环 —— 无 torch,只管取观测 / 调推理服务 / 下发指令 / 安全层。

    fastwam_server.py :8900  <──TCP──  **本脚本**  ──TCP──>  RobotArmService :9900
    (模型 + CUDA Graphs)                (控制环 + 安全层)      (ROS2 + Piper SDK)

为什么拆出来:`deploy_real.py` 把推理和控制环放在一个进程里,推理只能跑在
ThreadPoolExecutor 的工作线程上,于是 RTC 被迫放弃 `compile_mode=reduce-overhead`
(CUDA Graph Trees 的状态是线程局部的)。拆开后 server 把推理钉在专属线程上,
CUDA Graphs 就能用了 —— 实测比 default 快 2.4 倍。详见 fastwam_server.py 的 docstring。

本脚本刻意保留 openpi 的 `RobotArmService` 做硬件层(不学 GWP 让客户端直接驱动
Piper SDK + ROS2):那一层已真机验证过,重写只为省一跳 localhost 不值得,
而且会把 GWP 的夹爪比例问题(0.105 vs 硬件上限 0.07)带进来。

═══════════════════════════════════════════════════════════════════════════════
                         安全模型 —— 先读这一节
═══════════════════════════════════════════════════════════════════════════════
默认不动手臂。逐级 opt-in,没有一步跳到闭环自主的开关:

    --mode dry       (默认)只推理并打印动作,**什么都不发给手臂**。
                     用它核对单位、量程、方向。会调 reset 取观测(那是真实运动),
                     除非服务端支持只读 {"cmd":"obs"}。
    --mode sync      算完再动:执行 replan_steps 步 -> 驻留 -> 重取观测 -> 阻塞推理。
                     动作一顿一顿,但只用 chunk 前若干步(误差最小)。
    --mode async     后台推理、连续 30Hz。配 --rtc 才有接缝保护。

会动手臂的模式都还要 `--confirm-safety`(确认急停在手边、工作区清空、包络内无人)。
`--mode async` 额外要求 `--i-am-watching` 与 `--max-steps` 或 `--max-duration`。

始终生效的防护(都在 SafetyFilter 里,见 src/fastwam/deploy/control.py):
  * 首步跳变    chunk[0] 与实测状态的关节差:超 warn 告警,超 abort 中止
  * 逐步 delta  相邻指令的步长钳位(默认 0.10 rad/33.3ms,训练数据 p99.9)
  * 绝对限位    训练 action 的 global_min/max ± 余量(限位由 server 下发)
  * 节拍监控    超时次数与最大超时(定位控制环是否被抢 CPU)

时间尺度(fps=30 -> 1 步 = 33.3 ms,action_horizon = 32 步 = 1.067 s)
--------------------------------------------------------------------
async 的关键恒等式(INFER_LATENCY_DEBUG §1,当时理解错过):

    切入 chunk 的第几步  ==  推理延迟(控制步)

它与「什么时候发起推理」无关。等一会儿再发起不会让你从第 0 步切入,只会让上一个 chunk
往更深处走(误差更大)。所以唯一出路是把延迟压下去 —— 这正是本架构的目的。

用法
----
    # 0) 无硬件 + 无 GPU:假 server + 假 service,验证整条控制链
    python scripts/fastwam_server.py --mock --port 8900 &
    python scripts/deploy_fake_service.py --port 19912 &
    python scripts/fastwam_client.py --server tcp://127.0.0.1:8900 \
        --endpoint tcp://127.0.0.1:19912 --mode async --rtc \
        --confirm-safety --i-am-watching --max-steps 120

    # 1) 真机:先 dry 核对单位与方向
    python scripts/fastwam_client.py --server tcp://<gpu>:8900 \
        --endpoint tcp://<robot>:9900 --mode dry

    # 2) 真机:小步试
    python scripts/fastwam_client.py --server tcp://<gpu>:8900 \
        --endpoint tcp://<robot>:9900 --mode sync --confirm-safety --max-steps 60

    # 3) 真机:RTC 异步闭环
    python scripts/fastwam_client.py --server tcp://<gpu>:8900 \
        --endpoint tcp://<robot>:9900 --mode async --rtc \
        --confirm-safety --i-am-watching --max-duration 60 --record-dir runs/deploy/$(date +%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deploy_robot_client import Packer, RobotArmClient, _recv, _send, unpackb  # noqa: E402

from fastwam.deploy.control import DelayEstimator, Pacer, SafetyFilter  # noqa: E402
from fastwam.deploy.layout import (  # noqa: E402
    ACT_GRIP_IDX,
    ACT_JOINT_IDX,
    GRIPPER_TRAVEL_M,
    describe_action,
    obs_to_physical,
    to_action_layout,
    to_service_action,
    to_state_layout,
)


# ══════════════════════════════════════════════════════════════ 推理服务客户端
class PolicyClient:
    """`fastwam_server.py` 的同步客户端。协议与 RobotArmService 同一套编解码。

    图像**原样转发** openpi 给的 JPEG 字节流 —— 不解码、不 resize。理由见
    fastwam_server.py 的 docstring:预处理与训练的逐元素一致性只对 server 侧那一份
    代码成立,客户端插一步就破坏它;顺带把 JPEG 解码开销也从控制环挪走了。
    """

    def __init__(self, endpoint: str, recv_timeout_s: float = 120.0):
        self.endpoint = endpoint
        self._packer = Packer()
        host, port = endpoint.replace("tcp://", "").rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(float(recv_timeout_s))   # 首次请求可能含编译,别设太小
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.connect((host, int(port)))
        except OSError as exc:
            sock.close()
            raise ConnectionError(
                f"连不上推理服务 {endpoint}: {exc}\n"
                f"  GPU 机上先起: python scripts/fastwam_server.py --config <部署yaml> --port {port}"
            ) from exc
        self._sock = sock
        self.last_infer_s = 0.0
        self.last_timing: dict[str, float] = {}
        self.stat_calls = 0
        self.info: dict[str, Any] = self._request({"cmd": "ping"})["info"]

    def _request(self, msg: dict) -> dict:
        # 逐段计时:主循环看到的 elapsed 与 server 自报的 infer_s 差多少,决定了瓶颈在
        # 「模型慢」还是「网络/序列化慢」—— 两者的处置完全不同(INFER_LATENCY_DEBUG 坑 5)。
        t0 = time.perf_counter()
        blob = self._packer.pack(msg)
        t1 = time.perf_counter()
        _send(self._sock, blob)
        t2 = time.perf_counter()
        raw = _recv(self._sock)
        t3 = time.perf_counter()
        reply = unpackb(raw)
        self.stat_calls += 1
        self.last_timing = {
            "pack_ms": (t1 - t0) * 1000.0,
            "send_ms": (t2 - t1) * 1000.0,
            "recv_ms": (t3 - t2) * 1000.0,      # 含服务端处理时间
            "unpack_ms": (time.perf_counter() - t3) * 1000.0,
            "payload_mb": len(blob) / 1024 / 1024,
        }
        if "error" in reply:
            raise RuntimeError(f"推理服务报错: {reply['error']}")
        return reply

    def infer(self, images: dict, state14: np.ndarray,
              action_prefix: Optional[np.ndarray] = None, delay: int = 0) -> np.ndarray:
        """-> [T,14] 绝对物理动作(关节弧度 + 夹爪 0~1),action 排布。

        `action_prefix` 也是 **[d,14] 绝对物理量**(已承诺在接下来 d 步执行的指令)。
        走绝对量而不是归一化量是刻意的:客户端不持有 norm_stats,也就不可能把参考系搞错。
        """
        msg: dict[str, Any] = {"cmd": "infer", "images": images,
                               "state": np.ascontiguousarray(np.asarray(state14, dtype=np.float32))}
        if action_prefix is not None and int(delay) > 0:
            msg["action_prefix"] = np.ascontiguousarray(
                np.asarray(action_prefix, dtype=np.float32).reshape(-1, 14))
            msg["delay"] = int(delay)
        reply = self._request(msg)
        self.last_infer_s = float(reply.get("infer_s", 0.0) or 0.0)
        return np.asarray(reply["action"], dtype=np.float32)

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════ 落盘
class Recorder:
    """每步指令 + 每次规划的完整 chunk,便于事后离线复盘。

    不存拼图 mp4 —— 客户端不解 JPEG(拼图在 server 侧),这也让内存占用从
    1.5 GB/小时降到约 40 MB/小时。要看模型视角就在 server 侧另存。
    """

    def __init__(self, out_dir: Optional[Path]):
        self.dir = Path(out_dir) if out_dir else None
        self.steps: list[dict] = []
        self.chunks: list[dict] = []
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def on(self) -> bool:
        return self.dir is not None

    def add_chunk(self, chunk_raw, filtered, measured14, rep, infer_s, delay) -> None:
        if self.on:
            self.chunks.append({
                "action_raw": np.asarray(chunk_raw, dtype=np.float32),
                "action_filtered": np.asarray(filtered, dtype=np.float32),
                "measured14": np.asarray(measured14, dtype=np.float32),
                "infer_s": float(infer_s), "delay": int(delay),
                **{k: rep[k] for k in ("first_step_jump", "delta_clamped", "limit_clamped")},
            })

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
            json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        print(f"    记录已落盘: {self.dir / f'{tag}.npz'}")
        self.steps, self.chunks = [], []


# ═══════════════════════════════════════════════════════════════ preflight
def preflight(policy: PolicyClient, obs: dict, args, travel_m: float) -> None:
    """拿到第一帧真实观测后的一致性检查。用错单位/改名不报错,只会动作全乱,所以这里查死。"""
    print("\n>>> preflight(真实观测)")
    joint, grip = obs_to_physical(obs, travel_m)
    info = policy.info

    print(f"    server       后端={info.get('backend')}  "
          f"compile={info.get('compile_mode')}  steps={info.get('num_inference_steps')}  "
          f"horizon={info.get('action_horizon')}")
    print(f"    checkpoint   {info.get('checkpoint')}")
    if info.get("backend") == "mock":
        print("    ⚠️ server 是 MOCK 后端 —— 动作是编的,只能验证链路,不能做任务")

    # 1) 相机:名字与分辨率
    for name, v in obs["images"].items():
        arr = np.asarray(v)
        shape = f"{arr.shape} {arr.dtype}"
        kind = "JPEG 字节" if arr.ndim == 1 else "HWC"
        print(f"    相机 {name:18} {kind:9} {shape}")

    # 2) 夹爪单位(唯一"错了不报错"的地方)
    m = np.asarray(obs["gripper_position"], dtype=np.float32)
    print(f"    夹爪  {np.round(m, 5)} m  /{travel_m} ->  分数 {np.round(grip, 4)}")
    if np.any(grip < -0.15) or np.any(grip > 1.15):
        msg = (f"夹爪换算后 {np.round(grip, 4)} 明显落在 0~1 之外 —— "
               f"--gripper-travel-m={travel_m} 可能不对")
        if not args.allow_gripper_out_of_range:
            raise SystemExit(f"!! {msg}\n   确认无误可加 --allow-gripper-out-of-range")
        print(f"      ⚠️ {msg}")

    # 3) 起始位姿:与 server 下发的 action 限位比对(粗筛 OOD)
    lim = info.get("action_limits")
    if lim:
        a14 = to_action_layout(joint, grip)
        lo = np.asarray(lim["min"], dtype=np.float32)
        hi = np.asarray(lim["max"], dtype=np.float32)
        names = ([f"L_j{i}" for i in range(6)] + ["L_grip"]
                 + [f"R_j{i}" for i in range(6)] + ["R_grip"])
        n_out = 0
        for i in range(14):
            if a14[i] < lo[i] - 1e-6 or a14[i] > hi[i] + 1e-6:
                print(f"      ⚠️ {names[i]} = {a14[i]:+.3f} 在训练 action 区间 "
                      f"[{lo[i]:+.3f}, {hi[i]:+.3f}] 之外")
                n_out += 1
        print(f"    起始位姿     14 维中有 {n_out} 维落在训练 action 区间外"
              + ("(模型会在没见过的状态上外推)" if n_out else " —— 全部在分布内"))
    print()


# ═══════════════════════════════════════════════════════════════ 控制环
class Runner:
    def __init__(self, policy: PolicyClient, robot: RobotArmClient, args, rec: Recorder):
        self.policy, self.robot, self.args, self.rec = policy, robot, args, rec
        self.travel = float(args.gripper_travel_m)
        self.pacer = Pacer(float(args.hz))
        self.dry = args.mode == "dry"
        self.step_i = 0
        self.chunk_i = 0
        self.n_stall = 0
        self.n_overrun = 0
        self.last_cmd_m: Optional[np.ndarray] = None
        self.deadline = (time.monotonic() + float(args.max_duration)) if args.max_duration else None

        info = policy.info
        self.horizon = int(info.get("action_horizon", 32))
        sc = {
            "max_delta_joint": args.max_delta_joint,
            "max_delta_gripper": args.max_delta_gripper,
            "limit_margin_joint": args.limit_margin_joint,
            "limit_margin_gripper": args.limit_margin_gripper,
            "warn_first_step_jump": args.warn_first_step_jump,
            "abort_first_step_jump": args.abort_first_step_jump,
            "enabled": not args.no_safety,
        }
        self.safety = SafetyFilter.from_info(info, sc)

        # ── RTC ────────────────────────────────────────────────────────────
        self.rtc = bool(args.rtc) and args.mode == "async"
        if args.rtc and args.mode != "async":
            print("!! --rtc 只在 --mode async 下有意义(sync 每次从静止重新规划,chunk 之间没有"
                  "重叠,前缀无从谈起)—— 已忽略")
        if self.rtc:
            trained = info.get("rtc_trained")
            if trained is False:
                raise SystemExit(
                    "[拒绝] --rtc 需要 RTC 训练过的 checkpoint,但 server 上报 rtc_trained=False\n"
                    f"        checkpoint = {info.get('checkpoint')}\n"
                    "        拿基线 checkpoint 配 RTC 前缀会掉精度(离线实测关节误差 1.52->2.02 度、"
                    "MSE 近翻倍)。换 RTC 权重,或去掉 --rtc 跑 naive 异步。")
            if trained is None:
                print("!! server 没上报 rtc_trained —— 无法校验这份权重是不是 RTC 训练的,请自行确认")
            mx = int(info.get("rtc_max_delay", 11) or 11)
            if args.rtc_max_delay > mx:
                print(f"!! --rtc-max-delay={args.rtc_max_delay} 超过 server 上报的训练支撑上界 {mx}"
                      " —— 超出的 d 模型没见过")
            if info.get("compile_mode") is None:
                print("!! ⚠️ **server 没开 torch.compile** —— 30Hz 下延迟会远超训练覆盖的 d<=11,"
                      "RTC 前缀形同虚设(overrun 会接近 100%)。\n"
                      "   重启 server 加: --compile-mode reduce-overhead --warmup-rtc")
            elif info.get("compile_mode") == "default":
                print("!! server 用的是 compile_mode=default —— 本架构把推理钉在专属线程上,"
                      "reduce-overhead(CUDA Graphs)是可用的且快 2.4 倍,建议换过去")
        self.delay_est = DelayEstimator(
            mode=args.rtc_delay_mode, fixed_delay=args.rtc_fixed_delay,
            percentile=args.rtc_latency_percentile, window=args.rtc_latency_window,
            min_delay=args.rtc_min_delay, max_delay=args.rtc_max_delay, hz=float(args.hz))

    # ── 基本动作 ────────────────────────────────────────────────────────────
    def _measured14(self, obs: dict) -> np.ndarray:
        j, g = obs_to_physical(obs, self.travel)
        return to_action_layout(j, g)

    def _state14(self, obs: dict) -> np.ndarray:
        j, g = obs_to_physical(obs, self.travel)
        return to_state_layout(j, g)

    def _budget_done(self) -> bool:
        if self.args.max_steps and self.step_i >= int(self.args.max_steps):
            print(f"[{self.args.mode}] 到达 --max-steps={self.args.max_steps},停止")
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            print(f"[{self.args.mode}] 到达 --max-duration={self.args.max_duration}s,停止")
            return True
        return False

    def _send(self, a14: np.ndarray, obs_in: dict, chunk_step: int) -> tuple[dict, bool]:
        cmd_m = to_service_action(a14, self.travel)
        self.pacer.wait()
        obs, done, _info = self.robot.step(cmd_m)
        self.last_cmd_m = cmd_m
        self.step_i += 1
        self.rec.add_step(t=time.monotonic(), step=self.step_i, chunk=self.chunk_i,
                          chunk_step=chunk_step, cmd_phys=a14.astype(np.float32),
                          cmd_service=cmd_m, measured=self._measured14(obs_in))
        return obs, done

    def _plan(self, obs: dict, prefix=None, delay: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
        """取观测 -> 推理 -> 安全层。返回 (钳位后 chunk, 原始 chunk, 统计)。"""
        raw = self.policy.infer(obs["images"], self._state14(obs), prefix, delay)
        measured = self._measured14(obs)
        filtered, rep = self.safety.filter_chunk(raw, measured)
        self.chunk_i += 1
        self.rec.add_chunk(raw, filtered, measured, rep, self.policy.last_infer_s, delay)
        return filtered, raw, rep

    # ── dry:只推理并打印 ───────────────────────────────────────────────────
    def run_dry(self, obs: dict) -> None:
        print("\n" + "=" * 78)
        print("MODE: dry —— 只推理并打印,**不会给手臂发任何指令**")
        print("=" * 78)
        # dry 下 --max-steps 的含义是「推理几次」(不是控制步,因为不下发指令)。
        # --run-forever 时一直推到 Ctrl-C,便于长时间盯着输出看单位/方向是否稳定。
        forever = bool(getattr(self.args, "run_forever", False)) and not self.args.max_steps
        n = 0 if forever else int(self.args.max_steps or 3)
        i = 0
        while forever or i < n:
            i += 1
            t0 = time.perf_counter()
            raw = self.policy.infer(obs["images"], self._state14(obs))
            wall = time.perf_counter() - t0
            measured = self._measured14(obs)
            jump = float(np.abs(raw[0][ACT_JOINT_IDX] - measured[ACT_JOINT_IDX]).max())
            tm = self.policy.last_timing
            print(f"\n--- 推理 #{i}: chunk {raw.shape}  wall {wall * 1000:.0f} ms "
                  f"(server 自报模型 {self.policy.last_infer_s * 1000:.0f} ms) ---")
            print(f"  d(按当前 wall 换算) = {int(np.ceil(wall * float(self.args.hz)))} 个控制步")
            print(f"  上行 {tm['payload_mb']:.2f} MB  pack {tm['pack_ms']:.0f} / send {tm['send_ms']:.0f}"
                  f" / recv(含服务端) {tm['recv_ms']:.0f} / unpack {tm['unpack_ms']:.0f} ms")
            print(f"  实测状态 {describe_action(measured)}")
            print(f"  首步跳变 {jump:.4f} rad ({np.degrees(jump):.1f} 度)"
                  + ("  <- 会触发 abort!" if jump > self.args.abort_first_step_jump else ""))
            for k in range(min(3, raw.shape[0])):
                print(f"  [{k}] {describe_action(raw[k])}")
            _f, rep = self.safety.filter_chunk(raw, measured)
            print(f"  安全层会钳位: delta {rep['delta_clamped']} / limit {rep['limit_clamped']}")
            if forever or i < n:
                time.sleep(0.5)
        print("\n>>> dry 结束。单位与方向确认无误后,用 --mode sync --confirm-safety --max-steps 60 小步试。")

    # ── sync:执行 -> 驻留 -> 重取观测 -> 阻塞推理 ────────────────────────────
    def run_sync(self, obs: dict) -> None:
        settle = max(0, int(self.args.settle_steps))
        replan = max(1, int(self.args.replan_steps))
        done = False
        while not done and not self._budget_done():
            # 驻留:用"保持指令"换一帧**已稳定**的观测。
            # `step` 返回的 obs 是"刚下发、手臂还没走到"时刻采的(与训练语义一致),
            # 但推理阻塞的那 0.1~0.4 s 里手臂还在继续走向上一个目标,所以规划前要重发
            # 保持指令再取一次新 obs,否则拿的是过期状态。
            if self.last_cmd_m is not None:
                for _ in range(settle):
                    self.pacer.wait()
                    obs, done, _ = self.robot.step(self.last_cmd_m)
                    if done:
                        return

            chunk, _raw, rep = self._plan(obs)
            self.pacer.reset()   # 推理阻塞是设计如此的冻结,不是错过控制周期
            for k in range(min(replan, chunk.shape[0])):
                obs, done = self._send(chunk[k], obs, k)
                if done or self._budget_done():
                    done = True
                    break
            if self.chunk_i % 5 == 1 or done:
                print(f"    [sync] chunk {self.chunk_i} step {self.step_i}  "
                      f"模型 {self.policy.last_infer_s * 1000:.0f} ms  首步跳变 "
                      f"{np.degrees(rep['first_step_jump']):.1f} 度  "
                      f"钳位 delta/limit {rep['delta_clamped']}/{rep['limit_clamped']}")

    # ── async:后台推理,按实际经过步数对齐拼接 ────────────────────────────────
    def run_async(self, obs: dict) -> None:
        from concurrent.futures import ThreadPoolExecutor

        # 客户端这个线程只做「等 server 回包」(网络 IO),不碰 CUDA ——
        # CUDA Graphs 的线程亲和性由 server 侧的 InferenceThread 保证,与这里无关。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="net")
        chunk, raw, _ = self._plan(obs)     # 第一个 chunk 同步拿(冷启动没有可用前缀)
        prev_raw = raw
        idx = 0
        inflight = None
        fired_at = fired_delay = 0
        fired_measured = self._measured14(obs)
        done = False
        n_fire = 0
        fired_wall = time.perf_counter()
        land_log: list[tuple[int, int, int, float, float]] = []

        try:
            while not done and not self._budget_done():
                # 没有在飞的推理就立刻发起(延迟决定了能用到 chunk 的哪一段)
                if inflight is None:
                    imgs = dict(obs["images"])
                    state = self._state14(obs)
                    fired_at, fired_wall, n_fire = self.step_i, time.perf_counter(), n_fire + 1
                    fired_measured = self._measured14(obs)
                    if self.rtc:
                        fired_delay = self.delay_est.predict()
                        # 前缀 = 当前 chunk 对「新 chunk 前 d 步」那段时刻的预测。新 chunk 的
                        # t=0 对应发起推理的这一刻,也就是当前 chunk 的 idx ——
                        # 所以切 [idx : idx+d],不是 [0:d](INFER_LATENCY_DEBUG 教训 2)。
                        pfx = chunk[idx: idx + fired_delay]
                        if pfx.shape[0] < fired_delay:
                            fired_delay = int(pfx.shape[0])   # 当前 chunk 剩不下 d 步了
                        inflight = (pool.submit(self.policy.infer, imgs, state, pfx.copy(), fired_delay)
                                    if fired_delay > 0 else
                                    pool.submit(self.policy.infer, imgs, state))
                    else:
                        fired_delay = 0
                        inflight = pool.submit(self.policy.infer, imgs, state)

                # 算完了就切入
                if inflight is not None and inflight.done():
                    try:
                        new_raw = inflight.result()
                    except Exception as exc:
                        print(f"!! 推理失败: {type(exc).__name__}: {exc}")
                        break
                    inflight = None
                    el = self.step_i - fired_at      # 该 chunk 的 t=0 已经过去了这么多步
                    wall_ms = (time.perf_counter() - fired_wall) * 1000.0
                    infer_ms = self.policy.last_infer_s * 1000.0
                    self.delay_est.observe(el)
                    self.delay_est.observe_infer_s(self.policy.last_infer_s)
                    land_log.append((fired_at, el, fired_delay, wall_ms, infer_ms))

                    filtered, rep = self.safety.filter_chunk(new_raw, fired_measured)
                    self.chunk_i += 1
                    self.rec.add_chunk(new_raw, filtered, fired_measured, rep,
                                       self.policy.last_infer_s, fired_delay)
                    if el >= filtered.shape[0]:
                        print(f"    ⚠️ [async] 推理耗时 {el} 步 >= action_horizon {filtered.shape[0]}"
                              " —— 整条 chunk 已过期,只能用末步。降 steps 或改 sync。")
                    chunk, prev_raw = filtered, new_raw
                    idx = min(el, filtered.shape[0] - 1)
                    if self.rtc and el > fired_delay:
                        # 实测延迟超过发起时预估的 d:切入点落在前缀之外,这几步退化成
                        # naive 拼接。计数以便事后调 --rtc-latency-percentile。
                        self.n_overrun += 1
                        if self.n_overrun in (1, 10, 100):
                            print(f"    ⚠️ [rtc] 实测 elapsed={el} > 预估 d={fired_delay}"
                                  f"(第 {self.n_overrun} 次):切入点在前缀之外,接缝失去保护。"
                                  "调高 --rtc-latency-percentile 或 --rtc-min-delay")
                    if len(land_log) % 5 == 1:
                        tm = self.policy.last_timing
                        print(f"    [async] 落地 {len(land_log)}  step {self.step_i}  "
                              f"elapsed={el} 步 ({wall_ms:.0f} ms)  接入 chunk 第 {idx} 步"
                              + (f"  d(预估)={fired_delay}" if self.rtc else ""))
                        print(f"             其中 模型 {infer_ms:.0f} ms  "
                              f"网络+序列化 {wall_ms - infer_ms:.0f} ms  "
                              f"(上行 {tm['payload_mb']:.2f} MB)")

                if idx >= chunk.shape[0]:
                    # chunk 用尽而新的还没算完:保持上一条指令(而不是外推)
                    self.n_stall += 1
                    if self.n_stall in (1, 10, 100, 1000):
                        print(f"    ⚠️ [async] chunk 用尽,保持上一条指令(第 {self.n_stall} 次)。"
                              "推理耗时已接近/超过 action_horizon —— 降 steps 或改 sync")
                    obs, done = self._send(chunk[-1], obs, chunk.shape[0] - 1)
                    continue

                obs, done = self._send(chunk[idx], obs, idx)
                idx += 1
        finally:
            pool.shutdown(wait=False)
            print(f"\n    [async 诊断] 发起 {n_fire} 次 / 落地 {len(land_log)} 次 / "
                  f"chunk 用尽 {self.n_stall} 次")
            for fs, el, d, wall, inf_ms in land_log[:8]:
                print(f"      发起@step{fs:<4} elapsed={el:<3}步 d(预估)={d:<3} "
                      f"wall={wall:6.0f} ms  其中模型={inf_ms:5.0f} ms  "
                      f"(差 {wall - inf_ms:5.0f} ms = 网络+序列化)")
            if len(land_log) > 8:
                print(f"      ... 共 {len(land_log)} 条")
            print(f"    {self.delay_est.report()}")
            if self.rtc:
                print(f"    [rtc] 前缀 overrun {self.n_overrun} 次(elapsed > 预估 d,接缝失去保护)"
                      + ("  <- 应接近 0,否则调高 --rtc-latency-percentile" if self.n_overrun else ""))


# ══════════════════════════════════════════════════════════════════ CLI
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="tcp://127.0.0.1:8900", help="fastwam_server.py 地址")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:9900", help="RobotArmService 地址")
    ap.add_argument("--mode", default="dry", choices=["dry", "sync", "async"],
                   help="dry(默认,不动手臂) | sync(算完再动) | async(连续 30Hz,配 --rtc)")

    ap.add_argument("--confirm-safety", action="store_true",
                   help="任何运动都需要。确认急停在手边、工作区清空、包络内无人")
    ap.add_argument("--i-am-watching", action="store_true", help="--mode async 需要:有人全程目视监督")
    ap.add_argument("--confirm", action="store_true", help="preflight 之后等人工输入 yes 再动手臂")

    ap.add_argument("--rtc", action="store_true",
                   help="启用 RTC 动作前缀条件(只对 --mode async 有效)。"
                        "**server 的 checkpoint 必须是 rtc.enabled=true 训练出来的**,"
                        "否则本脚本直接拒绝")
    ap.add_argument("--rtc-delay-mode", default="measured", choices=["measured", "fixed"])
    ap.add_argument("--rtc-fixed-delay", type=int, default=6, help="冷启动 / fixed 模式用的 d")
    ap.add_argument("--rtc-min-delay", type=int, default=1)
    ap.add_argument("--rtc-max-delay", type=int, default=11,
                   help="必须 <= 训练 rtc.max_delay - 1(训练用 12 -> 这里 11)")
    ap.add_argument("--rtc-latency-window", type=int, default=20)
    ap.add_argument("--rtc-latency-percentile", type=float, default=0.9,
                   help="取偏大更安全:d 估小了切入点会落到前缀之外,那几步失去连续性保护")

    ap.add_argument("--hz", type=float, default=30.0, help="控制频率,必须等于数据集 fps=30")
    ap.add_argument("--replan-steps", type=int, default=10, help="sync:每个 chunk 执行多少步")
    ap.add_argument("--settle-steps", type=int, default=3, help="sync:规划前驻留几步以取到已稳定的观测")
    ap.add_argument("--max-steps", type=int, default=0, help="步数上限,0=不限(dry 下默认 3 次推理)")
    ap.add_argument("--max-duration", type=float, default=None, help="墙钟上限(秒)")
    ap.add_argument("--run-forever", action="store_true",
                   help="不设步数/时长上限,一直跑到 Ctrl-C。**手臂会持续运动**,"
                        "只在有人全程盯着时用。仅 --mode async/sync 需要;它与 "
                        "--max-steps / --max-duration 互斥(给了那两个就不必给这个)")
    ap.add_argument("--num-episodes", type=int, default=1)

    ap.add_argument("--gripper-travel-m", type=float, default=GRIPPER_TRAVEL_M,
                   help="夹爪满开度(米)。**别用 GWP 的 0.105** —— 硬件行程上限 70000 raw = 0.07 m,"
                        "0.105 会让 0.667 以上的输出全部塌缩成全开")
    ap.add_argument("--no-safety", action="store_true", help="关掉钳位(**不建议**,只用于对照)")
    ap.add_argument("--max-delta-joint", type=float, default=0.10)
    ap.add_argument("--max-delta-gripper", type=float, default=0.20)
    ap.add_argument("--limit-margin-joint", type=float, default=0.10)
    ap.add_argument("--limit-margin-gripper", type=float, default=0.05)
    ap.add_argument("--warn-first-step-jump", type=float, default=0.15)
    ap.add_argument("--abort-first-step-jump", type=float, default=0.60)
    ap.add_argument("--allow-gripper-out-of-range", action="store_true",
                   help="夹爪换算后明显越界时只告警不退出")

    ap.add_argument("--record-dir", default=None, help="落盘目录(建议每次 rollout 都存)")
    ap.add_argument("--go-zero-on-exit", action="store_true",
                   help="退出时让双臂归零。**会扫过工作区**,默认不开")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    moves = args.mode in ("sync", "async")
    if moves and not args.confirm_safety:
        print(f"[拒绝] --mode {args.mode} 会移动机器人,但没给 --confirm-safety。\n"
              "        确认:急停在手边、工作区清空、机械臂包络内无人,然后加 --confirm-safety。")
        return 2
    if args.mode == "async":
        if not args.i_am_watching:
            print("[拒绝] --mode async 需要 --i-am-watching(有人全程目视监督)。")
            return 2
        if not args.max_steps and not args.max_duration and not args.run_forever:
            print("[拒绝] --mode async 需要一个显式的停止界限:\n"
                  "         --max-duration <秒>   跑固定时长(推荐,如 600)\n"
                  "         --max-steps <步数>    跑固定步数(30 步 = 1 秒)\n"
                  "         --run-forever         一直跑到 Ctrl-C(**手臂会持续运动**)\n"
                  "       要求显式界限是刻意的:异步闭环会连续下发指令,忘了设界限"
                  "就等于让机器人无人值守地一直动。")
            return 2
        if args.run_forever:
            print("!! --run-forever:**不设步数/时长上限**,手臂会一直动到你按 Ctrl-C。"
                  "\n   确认急停在手边、有人全程盯着。Ctrl-C 会停止下发指令并让手臂保持当前位姿。")
        if not args.rtc:
            print("!! --mode async 未加 --rtc:这是 naive 异步 —— 不停机,但接缝处没有任何"
                  "条件保护,会有可见抖动。")
    if args.rtc_min_delay > args.rtc_max_delay:
        print("[拒绝] --rtc-min-delay 不能大于 --rtc-max-delay。")
        return 2
    if abs(float(args.hz) - 30.0) > 1e-6:
        print(f"!! --hz={args.hz} 与数据集 fps=30 不一致 —— 动作步长会与训练不符")

    print(f">>> 连接推理服务 {args.server}")
    policy = PolicyClient(args.server)
    print(f"    server 就绪: {json.dumps({k: v for k, v in policy.info.items() if k != 'action_limits'}, ensure_ascii=False, default=str)[:400]}")
    gt = policy.info.get("gripper_travel_m")
    if gt and abs(float(gt) - float(args.gripper_travel_m)) > 1e-9:
        print(f"!! 夹爪满开度不一致:server={gt} m,client={args.gripper_travel_m} m —— "
              "两边必须相同,否则夹爪指令会系统性偏差")

    print(f">>> 连接机器人 {args.endpoint}")
    robot = RobotArmClient(args.endpoint, recv_timeout_s=30.0, decode_images=False).connect()
    print("    RobotArmService 在线(图像保持 JPEG 不解码,由 server 侧解)")

    rc = 0
    try:
        for ep in range(1, max(1, int(args.num_episodes)) + 1):
            rec = Recorder(Path(args.record_dir) if args.record_dir else None)
            print(f"\n═══ Episode {ep}/{args.num_episodes} ═══")
            if args.mode == "dry":
                obs = robot.try_obs()
                if obs is None:
                    print(">>> 服务端不支持只读 obs;dry 模式下退到 reset(**手臂会走到初始位姿**)")
                    if not args.confirm_safety:
                        print("[拒绝] 那是真实运动,请加 --confirm-safety,或在 service 端加 "
                              "{'cmd':'obs'} 只读命令(见 deploy_robot_client.try_obs)")
                        return 2
                    obs = robot.reset()
            else:
                print(">>> reset:service 走初始位姿(约 5 s)...")
                obs = robot.reset()

            if ep == 1:
                preflight(policy, obs, args, float(args.gripper_travel_m))
                if args.confirm and args.mode != "dry":
                    if input(">>> 以上检查无误?手臂即将运动。输入 yes 继续: ").strip().lower() not in ("y", "yes"):
                        print("已取消。")
                        return 0

            runner = Runner(policy, robot, args, rec)
            t0 = time.perf_counter()
            try:
                if args.mode == "dry":
                    runner.run_dry(obs)
                elif args.mode == "sync":
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

            print(f"\n--- Episode {ep} 结束 ---")
            print(f"    {runner.step_i} 控制步 / {runner.chunk_i} 次规划 / {wall:.1f} s "
                  f"(有效 {runner.step_i / max(wall, 1e-6):.1f} Hz,目标 {args.hz:g} Hz)")
            print(f"    节拍: {runner.pacer.report()}")
            print(f"    安全层: {runner.safety.report()}")
            print(f"    推理服务: {policy.stat_calls} 次调用")
            print(f"    机器人 TCP: {robot.stat_calls} 次往返, 累计 {robot.stat_rtt_s:.1f} s "
                  f"(均 {robot.stat_rtt_s / max(robot.stat_calls, 1) * 1000:.1f} ms)")
            rec.flush(ep, {
                "episode": ep, "mode": args.mode, "rtc": bool(runner.rtc),
                "steps": runner.step_i, "chunks": runner.chunk_i, "wall_s": round(wall, 2),
                "hz": float(args.hz), "async_stalls": runner.n_stall,
                "rtc_overrun": runner.n_overrun,
                "elapsed_hist": runner.delay_est.elapsed_hist,
                "gripper_travel_m": float(args.gripper_travel_m),
                "server_info": {k: v for k, v in policy.info.items() if k != "action_limits"},
                "safety": {"jump_warn": runner.safety.n_jump_warn,
                           "delta_clamped": runner.safety.n_delta_clamped,
                           "limit_clamped": runner.safety.n_limit_clamped},
                "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
            })
            if rc:
                break
    finally:
        robot.close(send_stop=bool(args.go_zero_on_exit))
        policy.close()
        print(">>> 已断开" + ("(已发 stop,双臂归零)" if args.go_zero_on_exit else "(手臂保持当前位姿)"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
