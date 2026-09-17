"""控制环组件:安全层 / 节拍器 / 延迟预估 —— **纯 numpy，不 import torch**。

从 `scripts/deploy_real.py` 抽出来，好让 `scripts/fastwam_client.py`(控制环侧，无 torch)
和既有的单进程 `deploy_real.py` 用同一份实现。

与 `deploy_real.py` 里原版的唯一差异:`SafetyFilter` 改成吃 **numpy 的 lo/hi 数组**
而不是 torch 的 stats dict —— 否则 client 为了几个限位数字就得 import torch。
限位由 server 在 `ping` 的 info 里下发(见 `fastwam_server.py` 的 `action_limits`)。
"""

from __future__ import annotations

import math
import time

import numpy as np

from .layout import ACT_GRIP_IDX, ACT_JOINT_IDX


# ══════════════════════════════════════════════════════════════════ 安全层
class SafetyFilter:
    """在反归一化后的 14 维绝对指令上依次做:首步跳变检查 -> 逐步 delta 钳位 -> 绝对限位。

    delta 钳位是**累积**的(第 k 步以第 k-1 步的钳位结果为基准),所以如果模型想跑得比
    max_delta 快,执行轨迹会落后于预测轨迹且在本 chunk 内追不回来 —— 这没关系,
    每次重规划都从**实测状态**出发,误差不累积。

    参数(默认值来自训练数据实测的步间动作差分,60 episode / 129,762 转移):
        12 关节 p99 = 0.059 rad, p99.9 = 0.097, max = 0.815(离群)
        2 夹爪  p99 = 0.124,     p99.9 = 0.198
    取 p99.9:放过示教里 99.9% 的真实速度,挡住 46 度的单步跳变。
    """

    def __init__(
        self,
        action_min: np.ndarray,
        action_max: np.ndarray,
        max_delta_joint: float = 0.10,
        max_delta_gripper: float = 0.20,
        limit_margin_joint: float = 0.10,
        limit_margin_gripper: float = 0.05,
        warn_first_step_jump: float = 0.15,
        abort_first_step_jump: float = 0.60,
        enabled: bool = True,
    ):
        amin = np.asarray(action_min, dtype=np.float32).reshape(-1)
        amax = np.asarray(action_max, dtype=np.float32).reshape(-1)
        if amin.shape != (14,) or amax.shape != (14,):
            raise ValueError(f"action_min/max 必须是 14 维,得到 {amin.shape}/{amax.shape}")
        margin = np.zeros(14, dtype=np.float32)
        margin[ACT_JOINT_IDX] = float(limit_margin_joint)
        margin[ACT_GRIP_IDX] = float(limit_margin_gripper)
        self.lo, self.hi = amin - margin, amax + margin

        self.maxd = np.zeros(14, dtype=np.float32)
        self.maxd[ACT_JOINT_IDX] = float(max_delta_joint)
        self.maxd[ACT_GRIP_IDX] = float(max_delta_gripper)

        self.warn_jump = float(warn_first_step_jump)
        self.abort_jump = float(abort_first_step_jump)
        self.enabled = bool(enabled)
        self.n_jump_warn = 0
        self.n_delta_clamped = 0
        self.n_limit_clamped = 0

    @classmethod
    def from_info(cls, info: dict, cfg: dict | None = None) -> "SafetyFilter":
        """用 server `ping` 下发的 info 构造(client 侧用法)。

        限位来自训练 action 的 global_min/max —— 由 server 从 dataset_stats.json 读出并下发,
        这样 client 不需要持有 norm_stats,也就不可能用错一份。
        """
        lim = info.get("action_limits")
        if not lim:
            raise ValueError(
                "server 的 info 里没有 action_limits —— 它是旧版 server 或 mock 后端? "
                "安全层的绝对限位必须来自训练统计量,不能瞎猜。"
            )
        c = dict(cfg or {})
        return cls(action_min=lim["min"], action_max=lim["max"], **c)

    def filter_chunk(self, chunk: np.ndarray, measured14: np.ndarray) -> tuple[np.ndarray, dict]:
        """chunk [T,14] 物理量 + 实测 14 维 -> (钳位后的 chunk, 统计)。

        Raises:
            RuntimeError: 首步跳变超过 abort_first_step_jump(通常意味着单位/映射/权重出错)。
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"chunk 必须是 [T,14],得到 {chunk.shape}")
        measured14 = np.asarray(measured14, dtype=np.float32).reshape(-1)

        jump = float(np.abs(chunk[0][ACT_JOINT_IDX] - measured14[ACT_JOINT_IDX]).max())
        if jump > self.abort_jump:
            raise RuntimeError(
                f"chunk 首步与实测状态的关节差 {jump:.3f} rad ({np.degrees(jump):.1f} 度) "
                f"超过 abort_first_step_jump={self.abort_jump} —— 中止。"
                "\n  常见原因:初始位姿严重 OOD、夹爪单位换算错、相机改名映射错、权重不对。"
            )
        if jump > self.warn_jump:
            self.n_jump_warn += 1
            print(f"    ⚠️ chunk 首步跳变 {jump:.3f} rad ({np.degrees(jump):.1f} 度),已由 delta 钳位吸收")

        if not self.enabled:
            return chunk.copy(), {"first_step_jump": jump, "delta_clamped": 0, "limit_clamped": 0}

        out = np.empty_like(chunk)
        prev = measured14.copy()
        n_d = n_l = 0
        for k in range(chunk.shape[0]):
            raw_step = chunk[k] - prev
            step = np.clip(raw_step, -self.maxd, self.maxd)
            n_d += int(np.count_nonzero(np.abs(raw_step) > self.maxd + 1e-9))
            cand = prev + step
            clipped = np.clip(cand, self.lo, self.hi)
            n_l += int(np.count_nonzero(np.abs(clipped - cand) > 1e-9))
            out[k] = clipped
            prev = out[k]
        self.n_delta_clamped += n_d
        self.n_limit_clamped += n_l
        return out, {"first_step_jump": jump, "delta_clamped": n_d, "limit_clamped": n_l}

    def report(self) -> str:
        return (f"首步跳变告警 {self.n_jump_warn} 次, delta 钳位 {self.n_delta_clamped} 次, "
                f"限位钳位 {self.n_limit_clamped} 次")


# ══════════════════════════════════════════════════════════════════ 节拍器
class Pacer:
    """把每个控制周期补齐到 dt。在周期**开头**睡 —— 这样覆盖的是上一轮的全部开销
    (TCP 往返 + JPEG 解码 + 下发),而不只是其中一段。同 openpi AgilexEnv.step 的做法。
    """

    def __init__(self, hz: float):
        self.dt = 1.0 / float(hz) if hz and hz > 0 else 0.0
        self.t = None
        self.n = 0
        self.n_overrun = 0
        self.max_overrun_s = 0.0

    def wait(self) -> None:
        if self.dt > 0 and self.t is not None:
            slack = self.dt - (time.monotonic() - self.t)
            if slack > 0:
                time.sleep(slack)
            else:
                self.n_overrun += 1
                self.max_overrun_s = max(self.max_overrun_s, -slack)
        self.t = time.monotonic()
        self.n += 1

    def reset(self) -> None:
        """推理阻塞是**设计如此**的冻结,不是错过控制周期 —— 不 reset 会把它记成一次超时。"""
        self.t = None

    def report(self) -> str:
        if self.n == 0:
            return "无控制步"
        return (f"{self.n} 步,超时 {self.n_overrun} 次 ({self.n_overrun / self.n * 100:.1f}%),"
                f"最大超时 {self.max_overrun_s * 1000:.1f} ms")


# ══════════════════════════════════════════════════════════ RTC 延迟预估
def nearest_rank(values, pct: float) -> float:
    """分位数(nearest-rank)。样本少时比线性插值更保守。"""
    xs = sorted(values)
    if not xs:
        raise ValueError("nearest_rank 收到空序列")
    i = min(len(xs) - 1, max(0, math.ceil(float(pct) * len(xs)) - 1))
    return xs[i]


class DelayEstimator:
    """预估「这次推理会花掉多少个控制步」,作为 RTC 的前缀长度 d。

    必须在**发起推理时**就定 d(前缀要作为输入喂进去),而真实 elapsed 只有算完才知道,
    所以用最近若干次实测的分位数外推。论文相对 inference-time RTC 的主要增益之一正是
    「d 是运行时输入,不必保守估计」—— 所以用实测而不是一个固定的保守常数。

    取偏大一点更安全:d 估小了,切入点会落在前缀之外,那几步就失去了连续性保护。

    ⚠️ 首选依据是**观测到的 elapsed**(控制步),不是模型自报的 infer_s ——
    后者不含网络往返、预处理与 GIL 等待,会系统性偏小 1 步进而 overrun
    (INFER_LATENCY_DEBUG 坑 5:infer_s p90 197ms -> d=6,而真实 wall 233ms -> d=7)。
    """

    def __init__(self, mode: str = "measured", fixed_delay: int = 6,
                 percentile: float = 0.9, window: int = 20,
                 min_delay: int = 1, max_delay: int = 11, hz: float = 30.0):
        if mode not in ("measured", "fixed"):
            raise ValueError(f"delay_mode 只能是 measured / fixed,得到 {mode!r}")
        self.mode = mode
        self.fixed_delay = int(fixed_delay)
        self.percentile = float(percentile)
        self.window = int(window)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.hz = float(hz)
        self.elapsed_hist: list[int] = []
        self._infer_s_hist: list[float] = []

    def observe(self, elapsed_steps: int) -> None:
        """落地时记一次真实 elapsed(控制步)。"""
        self.elapsed_hist.append(int(elapsed_steps))

    def observe_infer_s(self, infer_s: float) -> None:
        """次选依据:server 自报的模型耗时,只在 elapsed 样本不足时用。"""
        if infer_s and infer_s > 0:
            self._infer_s_hist.append(float(infer_s))

    def predict(self) -> int:
        if self.mode == "fixed":
            d = self.fixed_delay
        elif len(self.elapsed_hist) >= 3:
            d = int(nearest_rank(self.elapsed_hist[-self.window:], self.percentile))
        elif self._infer_s_hist:
            # 只有模型耗时可用(前几个 chunk)。它偏小,所以额外 +1 步兜住网络/预处理/GIL。
            d = math.ceil(nearest_rank(self._infer_s_hist[-self.window:], self.percentile) * self.hz) + 1
        else:
            d = self.fixed_delay  # 冷启动,什么都还没测到
        return int(max(self.min_delay, min(self.max_delay, d)))

    def report(self) -> str:
        if not self.elapsed_hist:
            return "无 elapsed 样本"
        h = self.elapsed_hist
        return (f"实测 elapsed(控制步): 中位 {nearest_rank(h, 0.5):.0f}  "
                f"p90 {nearest_rank(h, 0.9):.0f}  max {max(h)}  (n={len(h)})")
