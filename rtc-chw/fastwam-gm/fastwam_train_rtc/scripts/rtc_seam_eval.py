#!/usr/bin/env python
"""Training-time RTC 的离线接缝评测：量化 chunk 之间的连续性。

为什么需要这个脚本：论文丢掉了 inference-time RTC 的 soft masking，硬前缀只保证 C⁰ 连续，
**前缀之后的过渡是模型学出来的、不是显式约束的**。论文自己把这条列为局限 L1，且只给了
「真机成功率打平」这个经验结论。解读文档 §9.1 明确要求：*看 chunk 接缝处的关节速度/加速度
曲线，这是判断连续性是否真的保住了最直接的信号，因为丢了 soft masking，这一条必须实测。*

评测协议（如实复现部署时的时序，不用 GT 前缀）：

    chunk_A = infer_action(obs[t],   delay=0)          # 上一个 chunk
    prefix  = chunk_A[s : s+d]                         # 上一 chunk 对重叠时刻的预测
    chunk_B = infer_action(obs[t+s], prefix, delay=d)  # 新 chunk

    实际执行的轨迹 = concat(chunk_A[0 : s+d], chunk_B[d :])
                                     ↑ 接缝在这里：committed 前缀交棒给新生成的后缀

前缀取自**模型自己生成的** chunk_A，而非数据集 GT —— 这一点是刻意的：它如实暴露了
解读文档 §9.2 指出的 exposure bias（训练前缀来自 GT，推理前缀来自模型自身，且沿 chunk
链递归累积），论文完全没讨论这个失配面。

对照基线：拿**非 RTC 的 checkpoint** 跑 `--delays 0`，得到 naive async（直接拼接、
无任何条件）的 jerk 参考量。RTC 若有效，接缝跳变应显著低于该基线。

用法：
    source env.sh
    # RTC checkpoint 扫 d
    python scripts/rtc_seam_eval.py \
      --task agilex_rtc_3cam_384_1e-4 \
      --ckpt runs/agilex_rtc_3cam_384_1e-4/rtc_A/checkpoints/weights/step_004000.pt \
      --delays 0,2,4,8 --num-samples 32 --out eval_offline/rtc_seam/rtc_A_step004000

    # naive async 基线（非 RTC 的 final_A）
    python scripts/rtc_seam_eval.py \
      --task agilex_final_3cam_384_1e-4 \
      --ckpt /mnt/data/gaomeng/FastWAM/runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_032000.pt \
      --delays 0 --num-samples 32 --out eval_offline/rtc_seam/final_A_step032000_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_eval import EpisodeIndex, denorm_action, load_and_verify  # noqa: E402

from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


# ---------------------------------------------------------------- 指标

# action 的 14 维排布是**交错**的（与 deploy_real.py:111-112 一致，别改顺序）：
#   [0..5] 左臂关节(rad)  [6] 左夹爪(0~1)  [7..12] 右臂关节(rad)  [13] 右夹爪(0~1)
# 关节是弧度、夹爪是 0~1 分数，**量纲不同，不能混在一起求平均**。
ACT_JOINT_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ACT_GRIP_IDX = [6, 13]
# 夹爪 0~1 归一化对应满开 0.07 m（Agilex Cobot Magic）。
GRIP_TRAVEL_M = 0.07


def accuracy_metrics(
    chunk_phys: torch.Tensor,
    gt_phys: torch.Tensor,
    delay: int,
    exec_window: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """新 chunk 自生成段 vs GT 的误差，关节与夹爪**分开**报，并区分全段与执行窗口。

    为什么要区分执行窗口：`DEPLOY_DESIGN.md` 实测动作误差随 chunk 步号近似线性增长
    （`error(k) ≈ 1.37 + 0.232·(k-1)` 度，k=1 -> 1.37 度，k=32 -> 8.55 度，6 倍差距），
    而部署时 `replan_steps <= 10`，**只执行 chunk 前若干步**。在全 32 步上平均会让指标被
    「永远不会被执行的尾部」主导，从而高估实际影响。

    返回 (aggregates, joint_err_deg_by_step[H], grip_err_frac_by_step[H])，
    后两个是**按 chunk 步号**的误差曲线（前 `delay` 步是 clamp 死的前缀，置 nan）。
    """
    diff = chunk_phys - gt_phys                                   # [H, 14]
    joint_by_step = diff[:, ACT_JOINT_IDX].abs().mean(dim=1)      # [H] rad
    grip_by_step = diff[:, ACT_GRIP_IDX].abs().mean(dim=1)        # [H] 0~1 分数
    joint_deg_by_step = torch.rad2deg(joint_by_step)

    horizon = chunk_phys.shape[0]
    window_end = min(delay + exec_window, horizon)

    def _mean(values: torch.Tensor, start: int, end: int) -> float:
        return float(values[start:end].mean()) if end > start else float("nan")

    aggregates = {
        # 全后缀 [delay, H)：与之前口径一致，便于对比历史数字
        "joint_deg_full": _mean(joint_deg_by_step, delay, horizon),
        "grip_frac_full": _mean(grip_by_step, delay, horizon),
        # 执行窗口 [delay, delay+exec_window)：**部署时真正会被执行的那几步**
        "joint_deg_win": _mean(joint_deg_by_step, delay, window_end),
        "grip_frac_win": _mean(grip_by_step, delay, window_end),
        "grip_mm_win": _mean(grip_by_step, delay, window_end) * GRIP_TRAVEL_M * 1000.0,
        # 首步误差：接缝之后第一个被执行的动作，最能反映「接得准不准」
        "joint_deg_first": float(joint_deg_by_step[delay]) if delay < horizon else float("nan"),
    }

    # 前缀段是 clamp 死的，不属于「模型自生成」，从曲线里剔除
    joint_curve = joint_deg_by_step.clone()
    grip_curve = grip_by_step.clone()
    if delay > 0:
        joint_curve[:delay] = float("nan")
        grip_curve[:delay] = float("nan")
    return aggregates, joint_curve, grip_curve


def seam_metrics(
    trajectory: torch.Tensor,
    seam_index: int,
    fps: float,
) -> dict[str, float]:
    """接缝处的速度不连续 / 加速度尖峰，以及同一条轨迹内的「正常」水平作参照。

    `trajectory` 是**反归一化后**的 [T, D] 绝对关节位置；`seam_index` 是新 chunk 自己
    生成的第一步在 trajectory 里的下标。

    v[k] = traj[k] - traj[k-1]                    （单位：物理量/步）
    接缝跳变 = |v[seam] - v[seam-1]|              （即接缝处的加速度）
    参照     = 轨迹内所有 |v[k] - v[k-1]| 的中位数（这条轨迹「正常」的 jerk 水平）

    返回值里 `seam_ratio` 是最该看的数：接近 1 说明接缝与普通相邻步无异（连续性保住了），
    远大于 1 说明接缝处有可见的抖动。
    """
    if trajectory.ndim != 2:
        raise ValueError(f"`trajectory` must be [T, D], got {tuple(trajectory.shape)}")
    total_steps = trajectory.shape[0]
    if not 1 <= seam_index <= total_steps - 1:
        raise ValueError(f"`seam_index` {seam_index} out of range for T={total_steps}")

    velocity = trajectory[1:] - trajectory[:-1]            # [T-1, D]
    accel = velocity[1:] - velocity[:-1]                   # [T-2, D]
    # v[seam] - v[seam-1] 对应 accel[seam-1]
    seam_accel = accel[seam_index - 1].abs()

    accel_abs = accel.abs()
    per_step_accel = accel_abs.mean(dim=1)                 # [T-2]
    reference = float(per_step_accel.median())
    seam_value = float(seam_accel.mean())

    return {
        "seam_accel": seam_value * fps * fps,
        "seam_accel_max_joint": float(seam_accel.max()) * fps * fps,
        "traj_accel_median": reference * fps * fps,
        "seam_ratio": seam_value / reference if reference > 1e-12 else float("nan"),
        "seam_vel_before": float(velocity[seam_index - 1].abs().mean()) * fps,
        "seam_vel_after": float(velocity[seam_index].abs().mean()) * fps,
    }


# ---------------------------------------------------------------- 主流程


def pick_sample_indices(epi: EpisodeIndex, num_samples: int, lookahead: int) -> list[int]:
    """等间隔挑起点，保证 idx 与 idx+lookahead 的窗口都不触发 padding、且在同一 episode。

    等间隔而非随机：覆盖整个 split 且完全可复现，是多 checkpoint 可比的前提。
    """
    candidates: list[int] = []
    for start, end in epi.unpadded_ranges():
        # 需要 idx+lookahead 仍在无 padding 区间内
        usable_end = end - lookahead
        if usable_end <= start:
            continue
        candidates.extend(range(start, usable_end))
    if not candidates:
        raise SystemExit(
            f"没有可用样本：每个 episode 都短于 num_frames + lookahead({lookahead})。"
        )
    if num_samples >= len(candidates):
        return candidates
    stride = len(candidates) / num_samples
    return [candidates[min(len(candidates) - 1, int(i * stride))] for i in range(num_samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="hydra task 配置名（须与训练时一致）")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--norm-stats", default="./runs/_shared/dataset_stats.json")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument(
        "--delays", default="0,2,4,8",
        help="要扫的模拟推理延迟 d（逗号分隔）。d=0 = naive async（直接拼接，无条件）",
    )
    parser.add_argument(
        "--replan-steps", type=int, default=8,
        help="execution horizon s，与部署 replan_steps 一致（约束 d <= H - s）",
    )
    parser.add_argument(
        "--exec-window", type=int, default=8,
        help="部署时每个 chunk 实际执行的步数（= replan_steps）。误差会在 [d, d+exec_window) "
             "上单独汇总——这才是部署口径，全 32 步平均会被永不执行的尾部主导",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10, help="去噪步数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-episodes", action="store_true", help="用全部 episode 而非 split")
    args = parser.parse_args()

    delays = [int(x) for x in str(args.delays).split(",") if x.strip() != ""]
    execution_horizon = int(args.replan_steps)
    args.out.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../configs", version_base="1.3"):
        hcfg = compose(config_name="train", overrides=[f"task={args.task}"])

    action_horizon = int(hcfg.data[args.split].num_frames) - 1
    for delay in delays:
        if delay >= action_horizon:
            raise SystemExit(f"delay={delay} 必须 < action_horizon={action_horizon}")
        if delay > action_horizon - execution_horizon:
            print(
                f"    ⚠️ delay={delay} 超过论文的时序约束 d <= H - s = "
                f"{action_horizon - execution_horizon}；真机上这个工作点不可行，仅作诊断。"
            )

    ds_kwargs = {"pretrained_norm_stats": str(args.norm_stats)}
    if args.dataset_dir:
        ds_kwargs["dataset_dirs"] = [str(args.dataset_dir)]
    if args.all_episodes:
        ds_kwargs["val_set_proportion"] = 0.0

    print(f">>> 构建数据集 (split={args.split}) ...")
    dataset = instantiate(hcfg.data[args.split], **ds_kwargs)
    processor = dataset.lerobot_dataset.processor

    info_path = Path(dataset.lerobot_dataset.dataset_dirs[0]) / "meta" / "info.json"
    fps = float(json.loads(info_path.read_text()).get("fps", 30.0))

    print(f">>> 构建模型 (task={args.task}) ...")
    model = instantiate(hcfg.model, model_dtype=torch.bfloat16, device=str(args.device))
    print(f">>> 加载权重 {args.ckpt}")
    load_and_verify(model, args.ckpt)
    model.eval()
    rtc_enabled = bool(getattr(model, "rtc", None) and model.rtc.enabled)
    print(f"    模型 RTC 状态: {model.rtc.describe() if hasattr(model, 'rtc') else 'n/a'}")
    if not rtc_enabled and any(d > 0 for d in delays):
        print(
            "    ⚠️ 这个 checkpoint 没有用前缀条件训练过，d>0 的结果只能当「未训练时前缀条件"
            "有多差」的参照，不能当 RTC 的表现。"
        )

    epi = EpisodeIndex(dataset, action_horizon + 1)
    indices = pick_sample_indices(epi, int(args.num_samples), execution_horizon)
    print(
        f">>> {len(indices)} 个样本 x {len(delays)} 个 delay，s={execution_horizon}, "
        f"H={action_horizon}, fps={fps:g}, 去噪 {args.steps} 步"
    )

    def infer(sample, action_prefix, delay):
        return model.infer_action(
            prompt=None,
            input_image=sample["video"][:, 0].unsqueeze(0).to(model.device, model.torch_dtype),
            action_horizon=action_horizon,
            proprio=sample["proprio"][0].to(model.device, model.torch_dtype),
            context=sample["context"].to(model.device, model.torch_dtype),
            context_mask=sample["context_mask"].to(model.device),
            num_inference_steps=int(args.steps),
            seed=int(args.seed),
            tiled=False,
            action_prefix=action_prefix,
            delay=delay,
        )["action"]

    exec_window = int(args.exec_window)
    rows: list[dict] = []
    curves: dict[int, list] = {}       # delay -> [joint_deg_by_step]
    grip_curves: dict[int, list] = {}  # delay -> [grip_frac_by_step]
    t_start = time.perf_counter()
    for order, idx in enumerate(indices):
        sample_a = dataset[idx]
        sample_b = dataset[idx + execution_horizon]
        episode_id, frame_in_ep, _ = epi.locate(idx)

        chunk_a = infer(sample_a, None, 0)
        chunk_a_phys = denorm_action(processor, chunk_a, sample_a["proprio"])

        for delay in delays:
            prefix = None if delay == 0 else chunk_a[execution_horizon : execution_horizon + delay]
            chunk_b = infer(sample_b, prefix, delay)
            chunk_b_phys = denorm_action(processor, chunk_b, sample_b["proprio"])

            # 实际执行的轨迹：committed 段（含前缀）+ 新生成的后缀
            executed = torch.cat(
                [chunk_a_phys[: execution_horizon + delay], chunk_b_phys[delay:]], dim=0
            )
            metrics = seam_metrics(executed, seam_index=execution_horizon + delay, fps=fps)

            # 前缀一致性：clamp 是否真的生效（应 ≈ 0）
            prefix_err = 0.0
            if delay > 0:
                prefix_err = float((chunk_b[:delay] - prefix).abs().max())

            # 新 chunk 自己生成的那段 vs GT：关节/夹爪分开，全段与执行窗口分开
            gt_b_phys = denorm_action(processor, sample_b["action"], sample_b["proprio"])
            acc, joint_curve, grip_curve = accuracy_metrics(
                chunk_b_phys, gt_b_phys, delay=delay, exec_window=exec_window
            )
            curves.setdefault(delay, []).append(joint_curve)
            grip_curves.setdefault(delay, []).append(grip_curve)

            rows.append(
                {
                    "sample_order": order,
                    "frame_index": int(idx),
                    "episode": int(episode_id),
                    "frame_in_episode": int(frame_in_ep),
                    "delay": int(delay),
                    "prefix_max_abs_err": prefix_err,
                    **acc,
                    **metrics,
                }
            )

        if (order + 1) % 5 == 0 or order + 1 == len(indices):
            elapsed = time.perf_counter() - t_start
            print(
                f"    {order + 1}/{len(indices)} 样本，已用 {elapsed:.0f}s "
                f"(约 {elapsed / (order + 1):.1f}s/样本)"
            )

    # ---------------- 落盘 ----------------
    csv_path = args.out / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "seam_accel",
        "seam_accel_max_joint",
        "traj_accel_median",
        "seam_ratio",
        "joint_deg_full",
        "joint_deg_win",
        "joint_deg_first",
        "grip_frac_full",
        "grip_frac_win",
        "grip_mm_win",
        "prefix_max_abs_err",
    ]
    summary = {
        "task": args.task,
        "ckpt": str(args.ckpt),
        "rtc_enabled_in_model": rtc_enabled,
        "split": args.split,
        "num_samples": len(indices),
        "action_horizon": action_horizon,
        "execution_horizon": execution_horizon,
        "exec_window": exec_window,
        "num_inference_steps": int(args.steps),
        "fps": fps,
        "per_delay": {},
    }
    for delay in delays:
        subset = [row for row in rows if row["delay"] == delay]
        summary["per_delay"][str(delay)] = {
            key: statistics.mean(row[key] for row in subset) for key in metric_keys
        }
        summary["per_delay"][str(delay)]["seam_ratio_median"] = statistics.median(
            row["seam_ratio"] for row in subset
        )

    # 按 chunk 步号的误差曲线：验证 DEPLOY_DESIGN.md 的「误差随步号线性增长」，
    # 并让「执行窗口内 vs 尾部」的差别一眼可见。
    import math

    curve_path = args.out / "error_by_step.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delay", "chunk_step_k", "joint_err_deg", "grip_err_frac", "n_samples"])
        for delay in delays:
            stacked = torch.stack(curves[delay])        # [N, H]
            gstacked = torch.stack(grip_curves[delay])
            for k in range(stacked.shape[1]):
                col = stacked[:, k]
                col = col[~torch.isnan(col)]
                gcol = gstacked[:, k]
                gcol = gcol[~torch.isnan(gcol)]
                if col.numel() == 0:
                    continue
                writer.writerow(
                    [delay, k, f"{float(col.mean()):.6f}", f"{float(gcol.mean()):.6f}", col.numel()]
                )
    summary["error_by_step_csv"] = str(curve_path)

    with open(args.out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # ---------------- 打印 ----------------
    print()
    print(f"结果已写入 {args.out}")
    print()
    header = (
        f"{'d':>3} | {'seam_accel':>10} | {'seam_ratio':>10} | "
        f"{'关节°win':>9} | {'关节°首步':>10} | {'关节°全段':>10} | "
        f"{'夹爪mm_win':>10} | {'prefix_err':>10}"
    )
    print(header)
    print("-" * (len(header) + 6))
    for delay in delays:
        s = summary["per_delay"][str(delay)]
        print(
            f"{delay:>3} | {s['seam_accel']:>10.3f} | {s['seam_ratio']:>10.3f} | "
            f"{s['joint_deg_win']:>9.3f} | {s['joint_deg_first']:>10.3f} | {s['joint_deg_full']:>10.3f} | "
            f"{s['grip_mm_win']:>10.3f} | {s['prefix_max_abs_err']:>10.2e}"
        )
    print()
    print(f"怎么读（exec_window={exec_window}，即部署的 replan_steps）：")
    print("  seam_ratio  接缝加速度 / 该轨迹加速度中位数。≈1 = 接缝与普通相邻步无异（好）。")
    print("              注意分母会随模型变化，所以也要看 seam_accel 这个绝对量。")
    print("  关节°win    执行窗口 [d, d+exec_window) 内的关节误差（度）。**这是部署口径。**")
    print("  关节°首步   接缝之后第一个被执行的动作的误差，最能反映「接得准不准」。")
    print("  关节°全段   全后缀 [d, 32) 的均值。DEPLOY_DESIGN.md 实测误差随步号线性增长 6 倍，")
    print("              所以这个数被永不执行的尾部主导，**不要用它判断部署影响**。")
    print("  d=0 一行    naive async（直接拼接、无条件），RTC 的 d>0 应显著更低。")
    print("  prefix_err  前缀 clamp 的数值误差，应 ≈ 0。")
    print(f"  逐步曲线    {curve_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
