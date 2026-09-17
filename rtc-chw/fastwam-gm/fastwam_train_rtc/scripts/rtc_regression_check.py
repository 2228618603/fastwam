"""RTC 改动的零回归校验：记录 / 比对 `training_loss` 的逐比特基线。

用途：`rtc.enabled=false` 时，RTC 相关改动（scheduler 广播泛化、action 分支 helper 抽取、
ActionDiT per-token timestep）必须让既有训练路径**数值完全不变**。

改动前记录基线：
    python scripts/rtc_regression_check.py task=agilex_final_3cam_384_1e-4 \
      +REGRESSION.record=/tmp/rtc_baseline.json

改动后比对：
    python scripts/rtc_regression_check.py task=agilex_final_3cam_384_1e-4 \
      +REGRESSION.check=/tmp/rtc_baseline.json

只跑 forward，不建 optimizer / deepspeed，所以比起真训练快得多。
"""

from __future__ import annotations

import json
import logging
import sys

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed

register_default_resolvers()

logger = logging.getLogger(__name__)


def _regression_config(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create(
        {
            "record": None,
            "check": None,
            "num_batches": 3,
            "batch_size": 2,
            "seed": 42,
            "device": "cuda:0",
        }
    )
    return OmegaConf.merge(defaults, cfg.get("REGRESSION", {}))


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = _regression_config(cfg)
    if reg.record is None and reg.check is None:
        raise SystemExit("需要 +REGRESSION.record=<path> 或 +REGRESSION.check=<path>")

    misc.register_work_dir(str(cfg.output_dir))
    device = str(reg.device)
    # 必须在建模型**之前**播种：ActionDiT 的 `action_encoder` / `head` 属于
    # ACTION_BACKBONE_SKIP_PREFIXES，不从 backbone 载入而是随机初始化；`proprio_encoder`
    # 也是新建的 Linear。不播种的话两次运行权重不同，loss 无法比对。
    set_global_seed(int(reg.seed))
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    # 与 trainer 一致：只有 dit(+proprio_encoder) 处于 train 模式。
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.train()

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=int(reg.batch_size),
        shuffle=False,
        num_workers=0,
    )

    records: list[dict] = []
    data_iter = iter(loader)
    for index in range(int(reg.num_batches)):
        sample = next(data_iter)
        # 每个 batch 前重置全局种子，让 randn_like / 时间步采样可复现。
        set_global_seed(int(reg.seed) + index)
        with torch.no_grad():
            loss, loss_dict = model.training_loss(sample)
        row = {"batch": index, "loss": float(loss.detach().double().item())}
        row.update({key: float(value) for key, value in sorted(loss_dict.items())})
        logger.info("batch %d: %s", index, json.dumps(row, sort_keys=True))
        records.append(row)

    if reg.record is not None:
        with open(str(reg.record), "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
        logger.info("baseline written to %s", reg.record)
        return

    with open(str(reg.check), "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    if len(baseline) != len(records):
        raise SystemExit(f"batch 数不一致: baseline={len(baseline)} current={len(records)}")

    mismatches = []
    for expected, actual in zip(baseline, records):
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"batch {expected['batch']} {key}: baseline={expected_value!r} current={actual_value!r}"
                )
    if mismatches:
        logger.error("零回归校验失败：")
        for line in mismatches:
            logger.error("  %s", line)
        sys.exit(1)
    logger.info("零回归校验通过：%d 个 batch 全部逐比特一致。", len(records))


if __name__ == "__main__":
    main()
