"""Training-time RTC 的正确性校验（不需要 GPU / 不加载模型的部分全部在此）。

    python scripts/rtc_selftest.py

覆盖：
  1. `rtc.py` 纯函数（延迟采样、前缀掩码、timestep 置零、clamp）
  2. `validate` 的时序约束 d <= H - s
  3. scheduler `add_noise` 的广播向后兼容 + 逐 token 新路径
  4. **干净前缀性质**：前缀位置 sigma=0 -> 加噪结果精确等于真实动作
  5. **前缀 loss 被剔除**：sum/sum(mask) 归一化，且 d 大的样本不被稀释
"""

from __future__ import annotations

import sys

import torch

from fastwam.models.wan22 import rtc
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def test_sample_delay() -> None:
    print("[1] sample_delay")
    cfg = rtc.RTCConfig(enabled=True, max_delay=12)
    torch.manual_seed(0)
    delay = rtc.sample_delay(4096, cfg, device=torch.device("cpu"))
    check("shape [B]", tuple(delay.shape) == (4096,), str(tuple(delay.shape)))
    check("dtype int64", delay.dtype == torch.int64, str(delay.dtype))
    check("半开区间 [0, max_delay)", int(delay.min()) == 0 and int(delay.max()) == 11,
          f"min={int(delay.min())} max={int(delay.max())}")
    check("d=0 在支撑内（冷启动必需）", bool((delay == 0).any()))
    check("逐样本而非逐 batch", int(delay.unique().numel()) == 12,
          f"unique={int(delay.unique().numel())}")
    mean = float(delay.float().mean())
    check("uniform 均值 ≈ (max_delay-1)/2 = 5.5", abs(mean - 5.5) < 0.25, f"mean={mean:.3f}")

    exp_cfg = rtc.RTCConfig(enabled=True, max_delay=12, delay_distribution="exp_decay")
    exp_delay = rtc.sample_delay(8192, exp_cfg, device=torch.device("cpu"))
    exp_mean = float(exp_delay.float().mean())
    check("exp_decay 均值明显小于 uniform（小 d 拿更多监督）", exp_mean < 3.0,
          f"mean={exp_mean:.3f}")
    check("exp_decay 值域合法", int(exp_delay.min()) >= 0 and int(exp_delay.max()) <= 11)


def test_prefix_mask_and_timestep() -> None:
    print("[2] build_prefix_mask / apply_prefix_timestep")
    horizon = 32
    delay = torch.tensor([0, 1, 5, 31])
    mask = rtc.build_prefix_mask(delay, horizon)
    check("shape [B, H]", tuple(mask.shape) == (4, horizon), str(tuple(mask.shape)))
    check("行和等于 delay", torch.equal(mask.sum(dim=1), delay), str(mask.sum(dim=1).tolist()))
    check("d=0 时全 False（退化为标准 flow matching）", not bool(mask[0].any()))
    check("前缀是连续的开头段", bool(mask[2, :5].all()) and not bool(mask[2, 5:].any()))

    base = torch.tensor([100.0, 200.0, 300.0, 400.0])
    token_t = rtc.apply_prefix_timestep(base, mask)
    check("shape [B, H]", tuple(token_t.shape) == (4, horizon), str(tuple(token_t.shape)))
    check("前缀位置精确为 0.0（本仓库 σ=0 = 干净数据）",
          bool((token_t[2, :5] == 0.0).all()), str(token_t[2, :5].tolist()))
    check("后缀位置保持采样到的 τ", bool((token_t[2, 5:] == 300.0).all()))
    check("d=0 的样本整行不变", bool((token_t[0] == 100.0).all()))

    # 推理版：标量 timestep 广播
    step_t = torch.tensor([777.0])
    scalar_out = rtc.apply_prefix_timestep_scalar(step_t, mask[2:3])
    check("apply_prefix_timestep_scalar 前缀为 0",
          bool((scalar_out[0, :5] == 0.0).all()) and bool((scalar_out[0, 5:] == 777.0).all()))


def test_clamp_prefix() -> None:
    print("[3] clamp_prefix")
    x = torch.randn(2, 8, 3)
    prefix = torch.full((2, 8, 3), 9.0)
    mask = rtc.build_prefix_mask(torch.tensor([3, 0]), 8)
    out = rtc.clamp_prefix(x, prefix, mask)
    check("前缀位被替换", bool((out[0, :3] == 9.0).all()))
    check("后缀位不动", torch.equal(out[0, 3:], x[0, 3:]))
    check("d=0 的样本完全不动", torch.equal(out[1], x[1]))


def test_validate() -> None:
    print("[4] validate（时序约束 d <= H - s）")

    def raises(cfg, horizon) -> bool:
        try:
            rtc.validate(cfg, horizon)
        except ValueError:
            return True
        return False

    check("disabled 时不校验", not raises(rtc.RTCConfig(enabled=False, max_delay=999), 32))
    check("H=32, s=8, max_delay=12 合法（max d=11 <= 24）",
          not raises(rtc.RTCConfig(enabled=True, max_delay=12, execution_horizon=8), 32))
    check("max_delay=26 违反 d <= H-s=24 -> 抛错",
          raises(rtc.RTCConfig(enabled=True, max_delay=26, execution_horizon=8), 32))
    check("max_delay=0 -> 抛错（d=0 必须可采样）",
          raises(rtc.RTCConfig(enabled=True, max_delay=0), 32))
    check("max_delay > H -> 抛错", raises(rtc.RTCConfig(enabled=True, max_delay=40, execution_horizon=1), 32))
    check("未知分布 -> 抛错",
          raises(rtc.RTCConfig(enabled=True, delay_distribution="gaussian"), 32))
    check("未知配置键 -> 抛错",
          _raises_value_error(lambda: rtc.RTCConfig.from_dict({"enabled": True, "typo": 1})))


def _raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def test_add_noise_broadcast() -> None:
    print("[5] add_noise 广播（向后兼容 + 逐 token）")
    sched = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)

    def legacy(samples, noise, timestep):
        """改动前的实现，用于逐比特对比。"""
        sigma = (timestep / 1000.0).to(samples.device, dtype=samples.dtype)
        if sigma.ndim == 0:
            return (1 - sigma) * samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (samples.ndim - 1)))
        return (1 - sigma) * samples + sigma * noise

    torch.manual_seed(0)
    action = torch.randn(4, 32, 14)
    noise = torch.randn_like(action)
    t_b = torch.rand(4) * 1000.0
    check("[B] + 3D action 与旧实现逐比特一致",
          torch.equal(sched.add_noise(action, noise, t_b), legacy(action, noise, t_b)))

    video = torch.randn(2, 48, 9, 24, 20)
    vnoise = torch.randn_like(video)
    t_v = torch.rand(2) * 1000.0
    check("[B] + 5D video 与旧实现逐比特一致",
          torch.equal(sched.add_noise(video, vnoise, t_v), legacy(video, vnoise, t_v)))

    t_bh = torch.rand(4, 32) * 1000.0
    out = sched.add_noise(action, noise, t_bh)
    check("[B, H] + 3D action shape 正确", tuple(out.shape) == (4, 32, 14), str(tuple(out.shape)))
    manual = (1 - (t_bh / 1000.0).unsqueeze(-1)) * action + (t_bh / 1000.0).unsqueeze(-1) * noise
    check("[B, H] 逐 token 数值正确", torch.allclose(out, manual, atol=1e-6))

    try:
        sched.add_noise(action, noise, torch.rand(4, 32, 14, 2))
        check("timestep 维度超过样本 -> 抛错", False)
    except ValueError:
        check("timestep 维度超过样本 -> 抛错", True)


def test_clean_prefix_property() -> None:
    print("[6] 干净前缀性质（论文改动 2：前缀自动涌现，无需特判）")
    sched = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    cfg = rtc.RTCConfig(enabled=True, max_delay=12, execution_horizon=8)
    torch.manual_seed(0)
    action = torch.randn(8, 32, 14)
    noise = torch.randn_like(action)
    base_t = sched.sample_training_t(8, torch.device("cpu"), torch.float32)
    delay = rtc.sample_delay(8, cfg, torch.device("cpu"))
    mask = rtc.build_prefix_mask(delay, 32)
    token_t = rtc.apply_prefix_timestep(base_t, mask)
    noisy = sched.add_noise(action, noise, token_t)

    m3 = mask.unsqueeze(-1).expand_as(action)
    check("前缀位置加噪结果 == 真实动作（精确相等）",
          torch.equal(noisy[m3], action[m3]))
    check("后缀位置确实被加了噪（与真实动作不同）",
          not torch.equal(noisy[~m3], action[~m3]))

    target = sched.training_target(action, noise, token_t)
    check("target = ε - A（与 timestep 无关，逐 token 亦然）",
          torch.equal(target, noise - action))


def test_prefix_loss_masking() -> None:
    print("[7] 前缀 loss 被剔除 + sum/sum(mask) 归一化（论文改动 3）")
    torch.manual_seed(0)
    horizon = 32
    # 构造：前缀位置 loss 极大，后缀位置 loss = 1。若前缀没被剔除，结果会被拉高。
    loss_token = torch.ones(3, horizon)
    mask = rtc.build_prefix_mask(torch.tensor([0, 8, 24]), horizon)
    loss_token = torch.where(mask, torch.full_like(loss_token, 1000.0), loss_token)

    valid = (~mask).to(loss_token.dtype)
    per_sample = (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    check("剔除前缀后每个样本 loss 都是 1.0（不被大值污染）",
          torch.allclose(per_sample, torch.ones(3)), str(per_sample.tolist()))
    check("d 大的样本没有被稀释（sum/sum 而非 sum/H）",
          abs(float(per_sample[2]) - float(per_sample[0])) < 1e-6,
          f"d=24 -> {float(per_sample[2]):.6f} vs d=0 -> {float(per_sample[0]):.6f}")

    # 对照：若错误地用 mean(dim=1)（即 sum/H），d 大的样本会被显著稀释
    wrong = (loss_token * valid).sum(dim=1) / horizon
    check("确认 sum/H 会稀释（说明这个归一化选择是必要的）",
          float(wrong[2]) < 0.3 * float(wrong[0]),
          f"d=24 -> {float(wrong[2]):.3f} vs d=0 -> {float(wrong[0]):.3f}")


def main() -> int:
    for test in (
        test_sample_delay,
        test_prefix_mask_and_timestep,
        test_clamp_prefix,
        test_validate,
        test_add_noise_broadcast,
        test_clean_prefix_property,
        test_prefix_loss_masking,
    ):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} 项失败：")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
