"""Training-time RTC: 动作前缀条件（action-prefix conditioning）。

论文：Black, Ren, Equi, Levine. *Training-Time Action Conditioning for Efficient Real-Time
Chunking.* arXiv:2512.05964。中文解读见仓库根目录 `RTCtrain_论文解读.md`。

核心思想：推理时一定有 `d` 步延迟，那就在训练时把它模拟出来 —— 把 chunk 的前 `d` 步当作
**已知的干净动作前缀**喂给模型，只在后缀上算 loss。于是模型直接学到条件分布
`p(A_{t+d:H} | o_t, A_{t:t+d})`，推理时零额外开销（对比 inference-time RTC 的
pseudoinverse guidance 每个去噪步一次 VJP）。

> ⚠️ **本仓库的 flow matching 约定与论文相反。**
> `WanContinuousFlowMatchScheduler` 用 `sigma = timestep / num_train_timesteps`、
> `add_noise = (1-σ)·A + σ·ε`，即 **σ=0 是干净数据、σ=1 是纯噪声**，推理时 σ 从 1 积分到 0。
> 所以论文里的「前缀 τ=1」在这里必须写成 **`timestep = 0`**（见 `apply_prefix_timestep`）。
> 好处是 `add_noise` 在 σ=0 处自动给出 `1·A + 0·ε = A`，干净前缀是**自动涌现**的，无需特判。

本模块只放不依赖 `nn.Module` 的纯函数与配置，便于单独校验。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

UNIFORM = "uniform"
EXP_DECAY = "exp_decay"
_DISTRIBUTIONS = (UNIFORM, EXP_DECAY)


@dataclass(frozen=True)
class RTCConfig:
    """Training-time RTC 的配置。`enabled=False` 时所有代码路径与原实现逐比特一致。"""

    enabled: bool = False

    # 模拟推理延迟 d 的采样上界。**半开区间**：d ∈ [0, max_delay)，对齐论文 Algorithm 1 的
    # `jax.random.randint(rng, (b,), 0, max_delay)`。max_delay=12 -> d ∈ {0..11}。
    # 这是最容易差一的地方。d=0 必须落在支撑内，否则冷启动第一个 chunk 无法生成。
    max_delay: int = 12

    # uniform  : 论文真机做法（Unif[0, 10) 覆盖 200ms @ 50Hz）
    # exp_decay: 论文仿真做法，权重 ∝ exp(-rate·d)，理由是「d 越大后缀越短、任务越简单、
    #            需要的监督越少」
    delay_distribution: str = UNIFORM
    exp_decay_rate: float = 0.5

    # execution horizon s，仅用于校验论文的时序约束 d ≤ H - s。不参与训练计算。
    execution_horizon: int = 8

    # 论文 §9.2 指出（论文本身没提）的 exposure bias 缓解手段：训练前缀取自数据集 GT，
    # 推理前缀来自模型自生成，二者分布不匹配且沿 chunk 链递归累积。>0 时给前缀加高斯抖动。
    # 默认关闭，保持与论文一致。
    prefix_jitter_std: float = 0.0

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "RTCConfig":
        if not payload:
            return cls()
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"Unknown `rtc` config keys: {unknown}. Expected a subset of {sorted(known)}."
            )
        return cls(**payload)

    def describe(self) -> str:
        if not self.enabled:
            return "RTC disabled (standard action flow-matching training)"
        return (
            f"RTC enabled | delay d ~ {self.delay_distribution}[0, {self.max_delay}) "
            f"(i.e. d in 0..{self.max_delay - 1}) | execution_horizon s={self.execution_horizon} "
            f"| prefix_jitter_std={self.prefix_jitter_std}"
        )


def validate(cfg: RTCConfig, action_horizon: int) -> None:
    """校验配置与 chunk 长度自洽。仅在 `cfg.enabled` 时做时序约束检查。"""
    if not cfg.enabled:
        return
    if cfg.delay_distribution not in _DISTRIBUTIONS:
        raise ValueError(
            f"`rtc.delay_distribution` must be one of {list(_DISTRIBUTIONS)}, "
            f"got {cfg.delay_distribution!r}"
        )
    if cfg.max_delay < 1:
        raise ValueError(
            f"`rtc.max_delay` must be >= 1 so that d=0 is sampleable, got {cfg.max_delay}. "
            "d=0 is required for cold-starting the first chunk."
        )
    if cfg.max_delay > action_horizon:
        raise ValueError(
            f"`rtc.max_delay` ({cfg.max_delay}) must be <= action_horizon ({action_horizon}); "
            "otherwise a sampled delay could cover the whole chunk and leave no postfix."
        )
    if cfg.execution_horizon < 1:
        raise ValueError(
            f"`rtc.execution_horizon` must be >= 1, got {cfg.execution_horizon}"
        )
    # 论文 §2.1 的时序约束 d <= H - s。max_delay 是半开上界，故最大实际 d 为 max_delay-1。
    max_feasible_delay = action_horizon - cfg.execution_horizon
    if cfg.max_delay - 1 > max_feasible_delay:
        raise ValueError(
            f"RTC timing constraint violated: max sampled delay d={cfg.max_delay - 1} exceeds "
            f"H - s = {action_horizon} - {cfg.execution_horizon} = {max_feasible_delay}. "
            "Lower `rtc.max_delay`, or lower `rtc.execution_horizon` if the deployed "
            "execution horizon is actually shorter."
        )
    if cfg.prefix_jitter_std < 0.0:
        raise ValueError(
            f"`rtc.prefix_jitter_std` must be >= 0, got {cfg.prefix_jitter_std}"
        )


def sample_delay(
    batch_size: int,
    cfg: RTCConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """采样模拟推理延迟，返回 `[B]` int64，取值 `{0 .. max_delay-1}`。

    **逐样本采样，不是逐 batch** —— 同一 batch 里不同样本有不同的 d，梯度信号更均匀
    （论文解读 §5.1）。
    """
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")
    if cfg.delay_distribution == UNIFORM:
        return torch.randint(
            low=0,
            high=cfg.max_delay,
            size=(batch_size,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
    if cfg.delay_distribution == EXP_DECAY:
        weights = torch.exp(
            -float(cfg.exp_decay_rate)
            * torch.arange(cfg.max_delay, device=device, dtype=torch.float32)
        )
        return torch.multinomial(
            weights, num_samples=batch_size, replacement=True, generator=generator
        ).to(torch.int64)
    raise ValueError(f"Unsupported `rtc.delay_distribution`: {cfg.delay_distribution!r}")


def build_prefix_mask(delay: torch.Tensor, action_horizon: int) -> torch.Tensor:
    """`[B]` 的延迟 -> `[B, H]` bool 前缀掩码，True 表示「该 token 是已知的干净前缀」。"""
    if delay.ndim != 1:
        raise ValueError(f"`delay` must be 1D [B], got shape {tuple(delay.shape)}")
    positions = torch.arange(action_horizon, device=delay.device).unsqueeze(0)
    return positions < delay.to(device=delay.device).unsqueeze(1)


def apply_prefix_timestep(
    base_timestep: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """把逐样本 timestep `[B]` 展成逐 token `[B, H]`，前缀位置置 **0.0**。

    0.0 而非论文的 1.0 —— 本仓库 σ=0 才是干净数据，见模块 docstring。
    `d` 不需要单独喂进模型：有多少个 token 的 timestep 为 0，d 就是多少
    （论文 Fig. 2 图注：“The flow matching timestep differs between tokens, which indicates
    the inference delay to the model.”）。
    """
    if base_timestep.ndim != 1:
        raise ValueError(
            f"`base_timestep` must be 1D [B], got shape {tuple(base_timestep.shape)}"
        )
    if prefix_mask.ndim != 2 or prefix_mask.shape[0] != base_timestep.shape[0]:
        raise ValueError(
            f"`prefix_mask` must be [B, H] matching `base_timestep` [B], got "
            f"{tuple(prefix_mask.shape)} vs {tuple(base_timestep.shape)}"
        )
    zero = torch.zeros((), device=base_timestep.device, dtype=base_timestep.dtype)
    return torch.where(prefix_mask, zero, base_timestep.unsqueeze(1))


def apply_prefix_timestep_scalar(
    step_timestep: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """推理版：把当前去噪步的标量 timestep 广播成 `[B, H]`，前缀位置置 0.0。"""
    if prefix_mask.ndim != 2:
        raise ValueError(f"`prefix_mask` must be [B, H], got {tuple(prefix_mask.shape)}")
    flat = step_timestep.reshape(-1)
    if flat.numel() == 1:
        flat = flat.expand(prefix_mask.shape[0])
    elif flat.numel() != prefix_mask.shape[0]:
        raise ValueError(
            f"`step_timestep` must have 1 or {prefix_mask.shape[0]} elements, got {flat.numel()}"
        )
    return apply_prefix_timestep(flat, prefix_mask)


def clamp_prefix(
    x: torch.Tensor,
    prefix: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """把前缀位置硬性 clamp 回真实前缀值。

    推理时**每个去噪步、模型调用之前**都要做，保证模型看到的前缀与训练时的输入分布严格一致
    （论文 §5.2）。
    """
    if x.shape != prefix.shape:
        raise ValueError(
            f"`x` and `prefix` must have the same shape, got {tuple(x.shape)} vs {tuple(prefix.shape)}"
        )
    return torch.where(prefix_mask.unsqueeze(-1), prefix.to(dtype=x.dtype), x)


def jitter_prefix(
    noisy_action: torch.Tensor,
    prefix_mask: torch.Tensor,
    std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """只给前缀位置加高斯抖动，缓解论文 §9.2 的 exposure bias。`std<=0` 时原样返回。"""
    if std <= 0.0:
        return noisy_action
    noise = torch.randn(
        noisy_action.shape,
        device=noisy_action.device,
        dtype=noisy_action.dtype,
        generator=generator,
    )
    return torch.where(
        prefix_mask.unsqueeze(-1), noisy_action + std * noise, noisy_action
    )
