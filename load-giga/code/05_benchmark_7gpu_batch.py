#!/usr/bin/env python3
"""Benchmark FastWAM train micro-batch size without eval/checkpoint I/O."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.fs import ensure_dir
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.samplers import ResumableEpochSampler


register_default_resolvers()
logger = get_logger(__name__)


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _rank_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank = _rank_int("RANK", 0)
    setup_logging(log_level=logging.INFO, is_main_process=(rank == 0))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/data/chw/fastwam/checkpoints")

    output_dir = Path(str(cfg.output_dir))
    if rank == 0:
        ensure_dir(str(output_dir))
        with (output_dir / "resolved_config.yaml").open("w") as f:
            OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = _resolve_train_device()

    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_dataset = instantiate(cfg.data.train)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.is_main_process:
        ds_plugin = accelerator.state.deepspeed_plugin
        zero_stage = None
        if ds_plugin is not None:
            zero_stage = ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage")
        logger.info(
            "Benchmark start: batch_size_per_gpu=%d world_size=%d global_batch=%d grad_accum=%d zero_stage=%s",
            int(cfg.batch_size),
            accelerator.num_processes,
            int(cfg.batch_size) * accelerator.num_processes,
            int(cfg.gradient_accumulation_steps),
            zero_stage,
        )

    Wan22Trainer._apply_dit_only_train_mode(model)
    trainable_params = list(model.dit.parameters())
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        trainable_params.extend(list(proprio_encoder.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    sampler = ResumableEpochSampler(
        dataset=train_dataset,
        seed=int(cfg.seed),
        batch_size=int(cfg.batch_size),
        num_processes=accelerator.num_processes,
    )
    worker_init_fn = set_global_seed(int(cfg.seed), get_worker_init_fn=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=max(int(cfg.max_steps or 1), 1))

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    optimizer.zero_grad(set_to_none=True)

    if cfg.resume:
        if accelerator.is_main_process:
            logger.info("Loading benchmark checkpoint: %s", cfg.resume)
        accelerator.unwrap_model(model).load_checkpoint(str(cfg.resume), optimizer=None)
        accelerator.wait_for_everyone()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

    max_steps = max(int(cfg.max_steps or 1), 1)
    step_times: list[float] = []
    losses: list[float] = []
    data_iter = iter(train_loader)

    for step_idx in range(max_steps):
        sample = next(data_iter)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        start = time.perf_counter()
        with accelerator.accumulate(model):
            with accelerator.autocast():
                train_model = model if hasattr(model, "training_loss") else accelerator.unwrap_model(model)
                loss, loss_dict = train_model.training_loss(sample)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)
        gathered_loss = accelerator.gather(loss.detach().float().reshape(1)).mean()
        losses.append(float(gathered_loss.item()))
        if accelerator.is_main_process:
            logger.info(
                "bench_step=%d/%d time=%.4fs loss=%.4f loss_detail=%s",
                step_idx + 1,
                max_steps,
                elapsed,
                losses[-1],
                {k: float(v) for k, v in sorted(loss_dict.items())},
            )

    peak_alloc = 0.0
    peak_reserved = 0.0
    current_alloc = 0.0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(accelerator.device) / 1024**3
        current_alloc = torch.cuda.memory_allocated(accelerator.device) / 1024**3
    local_metrics = torch.tensor(
        [peak_alloc, peak_reserved, current_alloc],
        device=accelerator.device,
        dtype=torch.float32,
    )
    all_metrics = accelerator.gather(local_metrics).reshape(accelerator.num_processes, 3)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        mean_step_time = sum(step_times) / max(len(step_times), 1)
        samples_per_sec = int(cfg.batch_size) * accelerator.num_processes / mean_step_time
        report = {
            "batch_size_per_gpu": int(cfg.batch_size),
            "world_size": accelerator.num_processes,
            "global_batch_size": int(cfg.batch_size) * accelerator.num_processes,
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
            "steps": max_steps,
            "step_times_sec": step_times,
            "mean_step_time_sec": mean_step_time,
            "steps_per_sec": 1.0 / mean_step_time,
            "samples_per_sec": samples_per_sec,
            "losses": losses,
            "peak_allocated_gib_by_rank": all_metrics[:, 0].cpu().tolist(),
            "peak_reserved_gib_by_rank": all_metrics[:, 1].cpu().tolist(),
            "current_allocated_gib_by_rank": all_metrics[:, 2].cpu().tolist(),
            "estimated_5000_steps_hours": 5000 * mean_step_time / 3600.0,
        }
        report_path = output_dir / "benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
