"""Training-time RTC 的端到端校验（需要 GPU，加载真实模型）。

    source env.sh
    CUDA_VISIBLE_DEVICES=0 python scripts/rtc_e2e_check.py \
      task=agilex_rtc_3cam_384_1e-4 resume=null output_dir=/tmp/rtc_e2e

覆盖 `scripts/rtc_selftest.py` 覆盖不到的部分 —— 真实模型上的：
  1. RTC 开启时 `training_loss` 跑通，且 ActionDiT 真的收到 **2-D** per-token timestep
  2. `loss_dict["rtc_mean_delay"]` 落在 [0, max_delay) 内
  3. `infer_action(delay=0)` 与既有路径**完全一致**（1-D timestep，零回归）
  4. `infer_action(action_prefix=p, delay=d)` 的返回值前缀**精确等于**传入前缀（clamp 生效）
  5. 前缀确实改变了后缀（模型没有忽略条件）
  6. 前缀路径不改变张量形状（torch.compile 友好性）
"""

from __future__ import annotations

import logging
import sys

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed

register_default_resolvers()

logger = logging.getLogger(__name__)
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        logger.info("  ok   %s", name)
    else:
        logger.error("  FAIL %s %s", name, detail)
        FAILURES.append(f"{name} {detail}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    misc.register_work_dir(str(cfg.output_dir))
    set_global_seed(42)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    if not model.rtc.enabled:
        raise SystemExit("本脚本需要 model.rtc.enabled=true（用 task=agilex_rtc_3cam_384_1e-4）")

    dataset = instantiate(cfg.data.train)
    action_horizon = int(cfg.data.train.num_frames) - 1

    # ---- 拦截 ActionDiT.prepare，记录它实际收到的 timestep 形状 ----
    seen_timestep_shapes: list[tuple[int, ...]] = []
    original_prepare = model.action_expert.prepare

    def spying_prepare(action_tokens, timestep, context, context_mask):
        seen_timestep_shapes.append(tuple(timestep.shape))
        return original_prepare(action_tokens, timestep, context, context_mask)

    model.action_expert.prepare = spying_prepare

    # ---------------- 1/2. 训练路径 ----------------
    logger.info("[1] RTC 训练路径")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    sample = next(iter(loader))

    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.train()

    seen_timestep_shapes.clear()
    set_global_seed(7)
    with torch.no_grad():
        loss, loss_dict = model.training_loss(sample)
    check("training_loss 跑通且 loss 有限", torch.isfinite(loss).item(), f"loss={float(loss)}")
    check(
        "ActionDiT 收到 2-D per-token timestep [B, H]",
        seen_timestep_shapes and seen_timestep_shapes[0] == (2, action_horizon),
        str(seen_timestep_shapes[:1]),
    )
    check("loss_dict 含 rtc_mean_delay", "rtc_mean_delay" in loss_dict, str(sorted(loss_dict)))
    mean_delay = loss_dict.get("rtc_mean_delay", -1.0)
    check(
        "rtc_mean_delay 落在 [0, max_delay)",
        0.0 <= mean_delay < model.rtc.max_delay,
        f"rtc_mean_delay={mean_delay} max_delay={model.rtc.max_delay}",
    )

    # ---------------- 3. delay=0 走 1-D 路径 ----------------
    logger.info("[2] 推理：delay=0 保持 1-D timestep（零回归）")
    single = dataset[0]
    infer_kwargs = dict(
        prompt=None,
        input_image=single["video"][:, 0].unsqueeze(0).to(model.device, model.torch_dtype),
        action_horizon=action_horizon,
        proprio=single["proprio"][0].to(model.device, model.torch_dtype),
        context=single["context"].to(model.device, model.torch_dtype),
        context_mask=single["context_mask"].to(model.device),
        num_inference_steps=4,
        seed=123,
        tiled=False,
    )

    seen_timestep_shapes.clear()
    out_d0 = model.infer_action(**infer_kwargs)
    check(
        "delay=0 时 timestep 仍是 1-D",
        all(len(shape) == 1 for shape in seen_timestep_shapes),
        str(set(seen_timestep_shapes)),
    )
    check("返回 delay=0", out_d0.get("delay") == 0, str(out_d0.get("delay")))
    check(
        "action 形状 [H, D]",
        tuple(out_d0["action"].shape) == (action_horizon, model.action_expert.action_dim),
        str(tuple(out_d0["action"].shape)),
    )

    # ---------------- 4/5/6. 前缀条件路径 ----------------
    logger.info("[3] 推理：delay=5 前缀条件路径")
    delay = 5
    # 用 d=0 的输出当前缀 —— 这与部署时「前缀来自上一个 chunk」的来源一致。
    prefix = out_d0["action"][:delay].clone()

    seen_timestep_shapes.clear()
    out_d5 = model.infer_action(**infer_kwargs, action_prefix=prefix, delay=delay)
    check(
        "delay>0 时 timestep 变成 2-D [1, H]",
        all(shape == (1, action_horizon) for shape in seen_timestep_shapes),
        str(set(seen_timestep_shapes)),
    )
    check("返回 delay=5", out_d5.get("delay") == delay, str(out_d5.get("delay")))

    returned_prefix = out_d5["action"][:delay]
    max_prefix_err = float((returned_prefix - prefix).abs().max())
    check(
        "返回值前缀精确等于传入前缀（循环内 + 循环外 clamp 都生效）",
        max_prefix_err < 1e-5,
        f"max_abs_err={max_prefix_err:.3e}",
    )

    postfix_diff = float((out_d5["action"][delay:] - out_d0["action"][delay:]).abs().max())
    check(
        "前缀确实影响了后缀（模型没有忽略条件）",
        postfix_diff > 1e-4,
        f"max_abs_diff={postfix_diff:.3e}",
    )

    # 形状稳定性：换一个 delay，timestep 形状必须不变（torch.compile 不会重编译）
    seen_timestep_shapes.clear()
    model.infer_action(
        **infer_kwargs, action_prefix=out_d0["action"][:9].clone(), delay=9
    )
    check(
        "delay 变化不改变 timestep 形状（compile 友好）",
        all(shape == (1, action_horizon) for shape in seen_timestep_shapes),
        str(set(seen_timestep_shapes)),
    )

    # 接受完整上一 chunk 作为 prefix（只取前 delay 步）
    out_full = model.infer_action(
        **infer_kwargs, action_prefix=out_d0["action"].clone(), delay=delay
    )
    check(
        "传入完整 chunk 时只取前 delay 步，结果与切片版一致",
        torch.allclose(out_full["action"], out_d5["action"], atol=1e-5),
        f"max_abs_diff={float((out_full['action'] - out_d5['action']).abs().max()):.3e}",
    )

    # ---------------- 错误处理 ----------------
    logger.info("[4] 错误处理")

    def raises(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    check(
        "delay>0 但没给 action_prefix -> 抛错",
        raises(lambda: model.infer_action(**infer_kwargs, delay=3)),
    )
    check(
        "delay >= action_horizon -> 抛错",
        raises(
            lambda: model.infer_action(
                **infer_kwargs, action_prefix=out_d0["action"].clone(), delay=action_horizon
            )
        ),
    )
    check(
        "action_prefix 短于 delay -> 抛错",
        raises(
            lambda: model.infer_action(
                **infer_kwargs, action_prefix=out_d0["action"][:2].clone(), delay=5
            )
        ),
    )

    logger.info("")
    if FAILURES:
        logger.error("%d 项失败：", len(FAILURES))
        for line in FAILURES:
            logger.error("  - %s", line)
        sys.exit(1)
    logger.info("端到端校验全部通过。")


if __name__ == "__main__":
    main()
