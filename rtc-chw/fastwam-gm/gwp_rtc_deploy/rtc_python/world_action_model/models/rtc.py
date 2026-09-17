"""Training-time RTC helpers for action-prefix conditioning.

This repository uses ``sigma=0`` for clean data, so known prefix tokens use
zero sigma/timestep.  The functions here are intentionally model-independent
so their shape and numerical properties can be tested without loading GWP.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    max_delay: int = 12
    delay_distribution: str = "uniform"
    exp_decay_rate: float = 0.5
    execution_horizon: int = 8
    prefix_jitter_std: float = 0.0

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "RTCConfig":
        if not payload:
            return cls()
        values = dict(payload)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown RTC config keys: {unknown}; expected {sorted(known)}")
        return cls(**values)

    def describe(self) -> str:
        if not self.enabled:
            return "RTC disabled (standard action flow-matching training)"
        return (
            f"RTC enabled | delay={self.delay_distribution}[0,{self.max_delay}) "
            f"| execution_horizon={self.execution_horizon} "
            f"| prefix_jitter_std={self.prefix_jitter_std}"
        )


def validate(cfg: RTCConfig, action_horizon: int) -> None:
    if not cfg.enabled:
        return
    if cfg.delay_distribution not in ("uniform", "exp_decay"):
        raise ValueError("rtc.delay_distribution must be 'uniform' or 'exp_decay'")
    if cfg.max_delay < 1:
        raise ValueError("rtc.max_delay must be >= 1 so d=0 can be sampled")
    if cfg.execution_horizon < 1:
        raise ValueError("rtc.execution_horizon must be >= 1")
    if cfg.max_delay - 1 > action_horizon - cfg.execution_horizon:
        raise ValueError(
            f"max sampled delay {cfg.max_delay - 1} exceeds H-s="
            f"{action_horizon - cfg.execution_horizon}"
        )
    if cfg.prefix_jitter_std < 0:
        raise ValueError("rtc.prefix_jitter_std must be >= 0")


def sample_delay(
    batch_size: int,
    cfg: RTCConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.delay_distribution == "uniform":
        return torch.randint(
            0, cfg.max_delay, (batch_size,), device=device,
            dtype=torch.int64, generator=generator,
        )
    if cfg.delay_distribution == "exp_decay":
        weights = torch.exp(
            -float(cfg.exp_decay_rate)
            * torch.arange(cfg.max_delay, device=device, dtype=torch.float32)
        )
        return torch.multinomial(
            weights, batch_size, replacement=True, generator=generator
        ).to(torch.int64)
    raise ValueError(f"Unsupported delay distribution: {cfg.delay_distribution!r}")


def build_prefix_mask(delay: torch.Tensor, action_horizon: int) -> torch.Tensor:
    if delay.ndim != 1:
        raise ValueError(f"delay must be [B], got {tuple(delay.shape)}")
    return torch.arange(action_horizon, device=delay.device).unsqueeze(0) < delay.unsqueeze(1)


def apply_prefix_sigma(base_sigma: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
    if base_sigma.ndim != 1 or prefix_mask.shape[0] != base_sigma.shape[0]:
        raise ValueError(
            f"base_sigma must be [B] and prefix_mask [B,H], got "
            f"{tuple(base_sigma.shape)} and {tuple(prefix_mask.shape)}"
        )
    zero = torch.zeros((), dtype=base_sigma.dtype, device=base_sigma.device)
    return torch.where(prefix_mask, zero, base_sigma[:, None]).unsqueeze(-1)


def apply_prefix_timestep(
    base_timestep: torch.Tensor, prefix_mask: torch.Tensor
) -> torch.Tensor:
    if base_timestep.ndim != 1 or prefix_mask.shape[0] != base_timestep.shape[0]:
        raise ValueError(
            f"base_timestep must be [B] and prefix_mask [B,H], got "
            f"{tuple(base_timestep.shape)} and {tuple(prefix_mask.shape)}"
        )
    zero = torch.zeros((), dtype=base_timestep.dtype, device=base_timestep.device)
    return torch.where(prefix_mask, zero, base_timestep[:, None])


def clamp_prefix(x: torch.Tensor, prefix: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
    if x.shape != prefix.shape:
        raise ValueError(f"x and prefix shapes differ: {tuple(x.shape)} vs {tuple(prefix.shape)}")
    return torch.where(prefix_mask.unsqueeze(-1), prefix.to(dtype=x.dtype), x)


def jitter_prefix(
    noisy_action: torch.Tensor,
    prefix_mask: torch.Tensor,
    std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if std <= 0:
        return noisy_action
    noise = torch.randn(
        noisy_action.shape,
        device=noisy_action.device,
        dtype=noisy_action.dtype,
        generator=generator,
    )
    return torch.where(prefix_mask.unsqueeze(-1), noisy_action + std * noise, noisy_action)


def masked_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    diff = (pred.float() - target.float()).pow(2)
    if prefix_mask is None:
        return diff.mean()
    postfix = (~prefix_mask).unsqueeze(-1).to(diff.dtype)
    denominator = (postfix.sum() * diff.shape[-1]).clamp_min(1.0)
    return (diff * postfix).sum() / denominator


def rebase_prefix_norm(
    prefix_norm: torch.Tensor,
    state_prev: torch.Tensor,
    state_new: torch.Tensor,
    action_q01: torch.Tensor,
    action_q99: torch.Tensor,
    delta_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rebase a normalized delta-action prefix from old state to new state.

    For delta dimensions, ``x_new = x_prev + 2*(state_prev-state_new)/(q99-q01)``.
    Absolute dimensions (the two grippers in the aligned 14D layout) are unchanged.
    """
    if prefix_norm.ndim not in (2, 3):
        raise ValueError(f"prefix_norm must be [L,D] or [B,L,D], got {tuple(prefix_norm.shape)}")
    dim = prefix_norm.shape[-1]
    for name, tensor in (
        ("state_prev", state_prev),
        ("state_new", state_new),
        ("action_q01", action_q01),
        ("action_q99", action_q99),
        ("delta_mask", delta_mask),
    ):
        if tensor.shape[-1] != dim:
            raise ValueError(f"{name} last dimension {tensor.shape[-1]} != {dim}")
    if state_prev.shape != state_new.shape:
        raise ValueError("state_prev and state_new must have the same shape")

    work_dtype = torch.float32
    value_range = (
        action_q99.to(work_dtype) - action_q01.to(work_dtype)
    ).clamp_min(eps)
    drift = (
        state_prev.to(work_dtype) - state_new.to(work_dtype)
    ) * delta_mask.to(work_dtype)
    shift = (2.0 * drift / value_range).unsqueeze(-2)
    return (prefix_norm.to(work_dtype) + shift).to(prefix_norm.dtype)
