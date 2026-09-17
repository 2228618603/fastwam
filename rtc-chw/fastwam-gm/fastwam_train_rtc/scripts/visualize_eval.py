#!/usr/bin/env python
"""把 offline_eval.py 存下的结果画成 PNG 图表（matplotlib）。

为什么用 matplotlib 而不是 HTML/SVG：
  1. 查看成本低 —— 远程 DSW 上 PNG 在 Jupyter / VS Code / 图片浏览器里直接打开，
     HTML 要下载或被 Jupyter 的受限 iframe 拦掉 inline 脚本
  2. 可自检 —— 这台机器没有无头浏览器，HTML 图表没法渲染出来检查标签重叠/溢出，
     PNG 可以直接看
  3. 14 维 × 2 样本 = 28 个子图，一张网格 PNG 比 288KB 的 SVG 更好用
代价是没有 hover tooltip —— 精确读数看配套的 metrics.csv / compare.csv。

配色沿用 dataviz 参考调色板，已用 validate_palette.js 在 light/dark 两模式、
--pairs all（small multiples 需要）下验证通过。**分类色只用前 3 槽**，因为该调色板
只有前 3 槽能过 all-pairs 门槛；多 ckpt 对比一律改用「训练步为 x 轴的折线」+ 表格，
而不是给每个 ckpt 配一个颜色。aqua 在亮色面上对比度 2.74:1 < 3:1，触发 relief
规则，所以所有系列都带图例 + 直接标注，且配套 CSV 可查数。

时间标注：动作步 ↔ 秒的换算写在 summary.json 的 annotate 里（fps / action_dt_s /
horizon_s），x 轴一律用**秒**，顶部再挂一条同轴换算的「动作步」刻度（单位换算，
不是第二个数据尺度 —— 双 y 轴是禁止的）。每张图页脚都盖上 ckpt 步数 / epoch /
验证集 / 样本量 / 评测时间戳。

用法:
  # 单个 ckpt 的结果目录
  python scripts/visualize_eval.py eval_offline/ab_A_step5000

  # 多 ckpt 扫描（offline_eval.py --config 的产出，含 compare.csv）
  python scripts/visualize_eval.py --sweep eval_offline/final_A_sweep

  # 几个独立目录横向对比
  python scripts/visualize_eval.py DIR_A DIR_B DIR_C
  python scripts/visualize_eval.py DIR --dpi 200 --pdf
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ------------------------------------------------------------------ 样式
# dataviz 参考调色板（亮色模式，已验证）
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # 分类色前 3 槽：过 all-pairs 门槛
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"]
DIV_LO, DIV_HI = "#2a78d6", "#e34948"            # diverging 两极：blue <-> red（冷/暖）
DIV_MID = "#f0efec"                              # 中点必须是中性灰
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DEEMPH = "#c3c2b7"                               # emphasis 形式里"其余系列"的灰

# 越小越好 / 越大越好，决定"最优点"往哪边找
LOWER_BETTER = {"action_rmse", "action_l2", "action_l1", "action_max_abs", "infer_s"}


def setup_style() -> None:
    # 注册系统 CJK 字体，否则中文标签是豆腐块
    for p in glob.glob("/usr/share/fonts/**/*.tt[cf]", recursive=True):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    have = {f.name for f in fm.fontManager.ttflist}
    fam = [n for n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans") if n in have]
    plt.rcParams.update({
        "font.family": fam or ["sans-serif"],
        "axes.unicode_minus": False,          # CJK 字体的减号会缺字形
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        # recessive 网格与坐标轴：只留 y 向细网格，去掉上/右边框
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        # 网格必须在数据之下：matplotlib 默认画在 patch 之上，
        # 会让条形/柱形看起来被切成好几段（实测踩到）
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,               # 2px 线
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


# ------------------------------------------------------------------ 读盘
def load_run(d: Path) -> dict:
    """读一个 ckpt 结果目录。"""
    S = json.loads((d / "summary.json").read_text())
    rows = list(csv.DictReader((d / "metrics.csv").open()))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return dict(dir=d, name=d.name, summary=S, rows=rows,
                samples=sorted((d / "samples").glob("sample_*.npz")))


def load_sweep(d: Path) -> dict:
    """读多 ckpt 扫描目录（offline_eval.py --config 的产出）。"""
    sw = json.loads((d / "sweep.json").read_text())
    runs = []
    for c in sw["ckpts"]:
        rd = d / c["dir"]
        if (rd / "summary.json").is_file():
            r = load_run(rd)
            r["name"] = c["tag"]
            r["step"] = c.get("step")
            r["epoch"] = c.get("epoch")
            runs.append(r)
    sw["runs"] = runs
    return sw


def runs_to_sweep(runs: list[dict]) -> dict:
    """把几个独立目录拼成 sweep 结构，好走同一套对比出图代码。"""
    keys = [k for k in ("action_rmse", "action_l2", "action_l1", "action_max_abs",
                        "psnr_rd", "ssim_rd", "psnr_rg", "ssim_rg", "infer_s")
            if any(k in r["summary"]["metrics"] for r in runs)]
    for r in runs:
        r.setdefault("step", r["summary"].get("step"))
        r.setdefault("epoch", r["summary"].get("epoch"))
    s0 = runs[0]["summary"]
    return {
        "runs": runs, "metric_keys": keys,
        "reference": runs[-1]["name"], "highlight": [r["name"] for r in runs[:3]],
        "annotate": s0.get("annotate", {}), "split": s0.get("split"),
        "episodes": s0.get("episodes", []), "num_samples": s0.get("num_samples"),
        "num_inference_steps": s0.get("num_inference_steps"), "mode": s0.get("mode"),
        "dataset_dir": s0.get("dataset_dir"),
    }


# ------------------------------------------------------------------ 公共组件
def footer(fig, text: str) -> None:
    """页脚：谁、评的什么、什么时候评的。

    用 supxlabel 而不是 fig.text —— constrained_layout 会给 supxlabel 预留空间，
    fig.text 不会，会压在最下面一行子图的 x 轴标签上。
    """
    fig.supxlabel(text, x=0.01, ha="left", fontsize=7.5, color=MUTED)


def run_footer(S: dict) -> str:
    a = S.get("annotate", {})
    ep = S.get("epoch")
    bits = [f"ckpt {S.get('ckpt_tag', '?')}"]
    if S.get("step") is not None:
        bits[-1] += f"（step {S['step']:,}" + (f" · epoch {ep:.2f}）" if ep else "）")
    eps = S.get("episodes") or []
    bits.append(f"{S.get('split', '?')} split · episode {eps if len(eps) <= 8 else str(len(eps)) + ' 个'}")
    bits.append(f"{S.get('num_samples', '?')} 样本 · {S.get('num_inference_steps', '?')} 步去噪 · mode={S.get('mode', '?')}")
    if a.get("horizon_steps") and a.get("fps"):
        bits.append(f"动作 {a['horizon_steps']} 步 @ {a['fps']:.0f}fps（1 步 = {1000*a['action_dt_s']:.1f}ms）")
    npad = S.get("num_padded_samples") or 0
    if npad:
        bits.append(f"⚠ 含 {npad} 个 padding 窗口")
    bits.append(f"评测于 {S.get('eval_finished_at', '?')}")
    return "　·　".join(bits)


def sweep_footer(sw: dict) -> str:
    eps = sw.get("episodes") or []
    return ("　·　".join([
        f"{len(sw['runs'])} 个 checkpoint，评的是同一批 {sw.get('num_samples', '?')} 个样本",
        f"{sw.get('split', '?')} split · episode {eps if len(eps) <= 8 else str(len(eps)) + ' 个'}",
        f"{sw.get('num_inference_steps', '?')} 步去噪 · mode={sw.get('mode', '?')}",
        f"出图于 {_dt.datetime.now().isoformat(timespec='minutes')}",
    ]))


def step_twin(ax, dt: float, n_steps: int, label: str = "动作步") -> None:
    """在顶部挂一条「动作步」刻度，与底部的「秒」是同一个轴的单位换算。

    x 数据是秒：第 k 步（k 从 1 起）对应 t = (k-1)·dt。
    刻度必须显式给定 —— 否则 matplotlib 会自己选到 step=0，而第 0 步并不存在。
    """
    if not dt or n_steps < 2:
        return
    sec = ax.secondary_xaxis("top",
                             functions=(lambda t: t / dt + 1.0, lambda k: (k - 1.0) * dt))
    every = max(1, int(round(n_steps / 4.0)))
    ticks = sorted({1, *range(every, n_steps, every), n_steps})
    sec.set_xticks(ticks)
    # labelpad 要够大，否则 "动作步" 会和正中间那个刻度数字压在一起（实测）
    sec.set_xlabel(label, fontsize=8, color=MUTED, labelpad=9)
    sec.tick_params(labelsize=7.5, colors=MUTED)
    sec.spines["top"].set_color(AXIS)


def _tcrit(n: int) -> float:
    """t 分布 0.975 分位。没装 scipy，用小表 + 1.96 兜底。"""
    if n < 2:
        return 0.0
    tt = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
          9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145, 20: 2.093, 25: 2.064,
          30: 2.045, 40: 2.021, 60: 2.000, 120: 1.980}
    df = n - 1
    return next((v for k, v in sorted(tt.items()) if df <= k), 1.96)


def ci95(a: np.ndarray) -> float:
    """均值的 95% 置信半宽。"""
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return 0.0
    return float(_tcrit(a.size) * a.std(ddof=1) / np.sqrt(a.size))


def save(fig, path: Path, dpi: int, pdf: bool) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"    {path.name}")


# ------------------------------------------------------------------ 单 ckpt 图
def fig_overview(S, rows, out: Path, dpi, pdf) -> None:
    """概览：KPI 文字 + 三个误差分布直方图。单一数字用文字，不用图表。"""
    M = S["metrics"]
    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1])

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    kpis = [("action RMSE", f'{M["action_rmse"]["mean"]:.4f}', "物理量（关节 rad）"),
            ("action L2 (MSE)", f'{M["action_l2"]["mean"]:.4f}',
             f'p90 {M["action_l2"]["p90"]:.4f}·与训练日志同定义'),
            ("action L1", f'{M["action_l1"]["mean"]:.4f}', "平均绝对误差")]
    if "psnr_rg" in M:
        kpis += [("PSNR 生成vs真实", f'{M["psnr_rg"]["mean"]:.2f} dB',
                  f'VAE 上限 {M["psnr_dg"]["mean"]:.2f}'),
                 ("SSIM 生成vs真实", f'{M["ssim_rg"]["mean"]:.3f}',
                  f'VAE 上限 {M["ssim_dg"]["mean"]:.3f}')]
    kpis += [("推理耗时", f'{M["infer_s"]["mean"]:.2f} s', "每样本")]
    for i, (lab, val, hint) in enumerate(kpis):
        x = i / len(kpis)
        ax.text(x, 0.74, lab, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.text(x, 0.34, val, transform=ax.transAxes, fontsize=17, color=INK, weight="600")
        ax.text(x, 0.08, hint, transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ep = f'（epoch {S["epoch"]:.2f}）' if S.get("epoch") else ""
    ax.set_title(f'FastWAM 离线评测 · {S.get("ckpt_tag", "?")}{ep} · '
                 f'{S["num_samples"]} 样本 / split={S["split"]}',
                 loc="left", fontsize=11, pad=10)

    for j, (key, lab, c) in enumerate([("action_rmse", "action RMSE", S1),
                                       ("action_l1", "action L1", S2),
                                       ("action_max_abs", "单点最大绝对误差", S3)]):
        a = fig.add_subplot(gs[1, j])
        v = np.array([r[key] for r in rows if isinstance(r.get(key), float)])
        a.hist(v, bins=min(16, max(4, v.size // 2)), color=c, edgecolor=SURFACE, linewidth=1.4)
        a.axvline(v.mean(), color=MUTED, ls="-", lw=1)
        # 均值放标题、不放图内：图内 annotate 会压在柱子上（实测）
        a.set_title(f"{lab}\n均值 {v.mean():.4g} · p90 {np.percentile(v,90):.4g} · "
                    f"95%CI ±{ci95(v):.4g}", loc="left", fontsize=9)
        a.set_xlabel(lab); a.set_ylabel("样本数")
    footer(fig, run_footer(S))
    save(fig, out / "01_overview.png", dpi, pdf)


def fig_per_dim(S, out: Path, dpi, pdf) -> None:
    """逐维误差：横向条形（14 个类别的量级比较；维度名长所以横向）。"""
    names = S.get("action_dim_names") or [f"dim{i}" for i in range(len(S["abs_err_per_dim_mean"]))]
    v = np.array(S["abs_err_per_dim_mean"])
    order = np.arange(len(v))[::-1]                 # 保持原顺序，从上到下
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(v) + 1.8))
    ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
    worst = int(v.argmax())
    cols = [S1 if i != worst else S2 for i in order]     # emphasis：只强调最差那一维
    ax.barh([names[i] for i in order], v[order], height=0.66, color=cols)
    for i, idx in enumerate(order):
        ax.text(v[idx], i, f"  {v[idx]:.4f}", va="center", fontsize=8,
                color=INK if idx == worst else INK2)
    ax.set_xlim(0, v.max() * 1.16)
    ax.set_xlabel("平均绝对误差（物理量：关节 rad / 夹爪开度）")
    ax.set_title(f"逐维平均绝对误差 —— 最差的是 {names[worst]}（{v[worst]:.4f}）",
                 loc="left")
    footer(fig, run_footer(S))
    save(fig, out / "02_per_dim_error.png", dpi, pdf)


def fig_per_step(S, out: Path, dpi, pdf) -> None:
    """误差随预测时域：折线 + p90 带。x 轴用秒，顶部挂动作步。"""
    mean = np.array(S["abs_err_per_step_mean"])
    p90 = np.array(S["abs_err_per_step_p90"])
    dt = float(S.get("annotate", {}).get("action_dt_s") or 0.0)
    t = np.arange(len(mean)) * dt if dt else np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.fill_between(t, mean, p90, color=S1, alpha=0.16, lw=0)
    ax.plot(t, mean, color=S1)
    ax.annotate("平均", (t[-1], mean[-1]), xytext=(-28, 7), textcoords="offset points",
                fontsize=8.5, color=INK2, weight="600")
    ax.annotate("p90", (t[-1], p90[-1]), xytext=(-24, 4), textcoords="offset points",
                fontsize=8.5, color=MUTED)
    g = mean[-1] / mean[0] if mean[0] else float("nan")
    if dt:
        ax.set_xlabel(f"预测时域（秒）—— {len(mean)} 步 × {1000*dt:.1f}ms，末步在 {t[-1]:.3f}s")
        step_twin(ax, dt, len(mean))
    else:
        ax.set_xlabel(f"预测步（1..{len(mean)}）")
    ax.set_ylabel("平均绝对误差")
    ax.set_title(f"误差随预测时域的累积 —— 末步/首步 = {g:.2f}×"
                 f"{'（>1.5 说明远端预测不可靠，真机应减小 replan_steps）' if g > 1.5 else ''}",
                 loc="left", pad=30)
    footer(fig, run_footer(S))
    save(fig, out / "03_error_vs_horizon.png", dpi, pdf)


def fig_video(S, rows, out: Path, dpi, pdf) -> None:
    """视频质量：3 组对比各一格直方图（small multiples）。"""
    if "psnr_rg" not in S["metrics"]:
        return
    items = [("psnr_rd", "PSNR 生成 vs VAE重建", S1), ("psnr_rg", "PSNR 生成 vs 真实", S2),
             ("psnr_dg", "PSNR VAE重建 vs 真实（上限）", S3),
             ("ssim_rd", "SSIM 生成 vs VAE重建", S1), ("ssim_rg", "SSIM 生成 vs 真实", S2),
             ("ssim_dg", "SSIM VAE重建 vs 真实（上限）", S3)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
    for a, (key, lab, c) in zip(axes.ravel(), items):
        v = np.array([r[key] for r in rows if isinstance(r.get(key), float)])
        a.hist(v, bins=min(14, max(4, v.size // 2)), color=c, edgecolor=SURFACE, linewidth=1.4)
        a.axvline(v.mean(), color=MUTED, ls="-", lw=1)
        a.set_title(f"{lab}\n均值 {v.mean():.3g}", loc="left", fontsize=9)
        a.set_ylabel("样本数")
    fig.suptitle("视频生成质量 —— 「生成vs真实」含 VAE 重建损失，"
                 "与「VAE上限」的差距才是模型自身误差", x=0.01, ha="left", fontsize=10.5)
    footer(fig, run_footer(S))
    save(fig, out / "04_video_quality.png", dpi, pdf)


def fig_traj(S, rows, samples, out: Path, dpi, pdf, which: str, ridx: int) -> None:
    """动作轨迹 pred vs GT：14 维 small multiples。x 轴用秒。"""
    row = rows[ridx]
    z = np.load(samples[ridx])
    pred, gt = z["pred_action_phys"], z["gt_action_phys"]
    names = S.get("action_dim_names") or [f"dim{d}" for d in range(pred.shape[1])]
    dt = float(S.get("annotate", {}).get("action_dt_s") or 0.0)
    t = np.arange(pred.shape[0]) * dt if dt else np.arange(1, pred.shape[0] + 1)

    D = pred.shape[1]
    ncol = 4
    nrow = int(np.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.1 * nrow), sharex=True)
    for d in range(nrow * ncol):
        a = axes.ravel()[d]
        if d >= D:
            a.axis("off"); continue
        a.plot(t, gt[:, d], color=S2, label="真实")
        a.plot(t, pred[:, d], color=S1, label="预测")
        mae = np.abs(pred[:, d] - gt[:, d]).mean()
        a.set_title(f"{names[d] if d < len(names) else f'dim{d}'}  MAE={mae:.4f}",
                    loc="left", fontsize=8.5)
        a.tick_params(labelsize=7.5)
        if d < ncol and dt:                       # 顶部步轴只挂第一行，否则太吵
            step_twin(a, dt, pred.shape[0])
        if d >= D - ncol:
            a.set_xlabel("秒" if dt else "预测步", fontsize=8)
    # 只在第一个子图放图例（2 序列必须有图例，避免只靠颜色区分）
    h0, l0 = axes.ravel()[0].get_legend_handles_labels()
    order = [l0.index("预测"), l0.index("真实")]      # 主体在前
    axes.ravel()[0].legend([h0[i] for i in order], [l0[i] for i in order],
                           fontsize=8, loc="best")
    span = f"，x 轴 0→{t[-1]:.3f}s" if dt else ""
    fig.suptitle(f"动作轨迹：预测 vs 真实 —— {which}"
                 f"（episode {int(row.get('episode_index', -1))} 第 "
                 f"{int(row.get('frame_in_episode', -1))} 帧, "
                 f"RMSE={row['action_rmse']:.4f}{span}）",
                 x=0.01, ha="left", fontsize=11)
    tag = "best" if "最小" in which else "worst"
    footer(fig, run_footer(S))
    save(fig, out / f"05_traj_{tag}.png", dpi, pdf)


# ------------------------------------------------------------------ 多 ckpt 图
def _xaxis_steps(ax, sw: dict, runs: list[dict]) -> bool:
    """x 轴用训练步；有 steps_per_epoch 就在顶部挂 epoch（同轴换算）。"""
    spe = (sw.get("annotate") or {}).get("steps_per_epoch")
    if not spe:
        return False
    sec = ax.secondary_xaxis("top", functions=(lambda s: s / spe, lambda e: e * spe))
    sec.set_xlabel("epoch", fontsize=8, color=MUTED, labelpad=9)
    sec.tick_params(labelsize=7.5, colors=MUTED)
    sec.spines["top"].set_color(AXIS)
    return True


def fig_sweep(sw: dict, out: Path, dpi, pdf) -> None:
    """指标随训练步的变化 —— checkpoint 选择的主图。

    训练步是**连续有序**变量，所以是折线而不是给每个 ckpt 配一个颜色。
    带 95% CI：所有 ckpt 评的是同一批样本，但样本间方差很大，CI 重叠就说明
    那个差异不显著，别当成"变好了"。
    """
    runs = [r for r in sw["runs"] if r.get("step") is not None]
    if len(runs) < 2:
        return
    runs = sorted(runs, key=lambda r: r["step"])
    # 排除 infer_s（跟模型质量无关）和 *_dg（VAE 天花板，按构造就是常数，
    # 与训练无关 —— 放进趋势图只会得到一条平线和一个毫无意义的"最优点"标注）。
    # 天花板改成写进 *_rg 的标题里当参照，数值本身在 compare.csv 和 04_video_quality 里。
    keys = [k for k in sw["metric_keys"] if k != "infer_s" and not k.endswith("_dg")]
    if not keys:
        return
    ceiling = {}
    for rg, dg in (("psnr_rg", "psnr_dg"), ("ssim_rg", "ssim_dg")):
        vals = [r["summary"]["metrics"][dg]["mean"] for r in runs
                if dg in r["summary"]["metrics"]]
        if vals:
            ceiling[rg] = float(np.mean(vals))

    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow), squeeze=False)
    steps = np.array([r["step"] for r in runs], dtype=float)

    for j, key in enumerate(keys):
        a = axes.ravel()[j]
        mean = np.array([r["summary"]["metrics"][key]["mean"] for r in runs])
        halfs = np.array([ci95(np.array([row[key] for row in r["rows"]
                                         if isinstance(row.get(key), float)]))
                          for r in runs])
        a.fill_between(steps, mean - halfs, mean + halfs, color=S1, alpha=0.15, lw=0)
        a.plot(steps, mean, color=S1, marker="o", markersize=4.5,
               markerfacecolor=S1, markeredgecolor=SURFACE, markeredgewidth=1.2)

        pick = int(mean.argmin() if key in LOWER_BETTER else mean.argmax())
        # emphasis：最优点用 slot2 + 白环，其余保持 slot1
        a.plot([steps[pick]], [mean[pick]], marker="o", markersize=8.5, color=S2,
               markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5, linestyle="none")
        a.annotate(f"{runs[pick]['name']}\n{mean[pick]:.4g}",
                   (steps[pick], mean[pick]), xytext=(0, 11),
                   textcoords="offset points", ha="center", fontsize=8,
                   color=INK, weight="600")
        arrow = "越小越好" if key in LOWER_BETTER else "越大越好"
        cap = f"；VAE 上限 {ceiling[key]:.4g}" if key in ceiling else ""
        a.set_title(f"{key}（{arrow}{cap}）", loc="left", fontsize=9.5, pad=26)
        a.set_xlabel("训练步")
        a.set_ylabel(key)
        # 上方多留 30%，给最优点那两行标注腾地方（否则会顶出坐标区）
        lo, hi = float((mean - halfs).min()), float((mean + halfs).max())
        span = (hi - lo) or (max(abs(hi), 1e-6) * 0.1)
        a.set_ylim(lo - 0.10 * span, hi + 0.30 * span)
        _xaxis_steps(a, sw, runs)
    for j in range(len(keys), nrow * ncol):
        axes.ravel()[j].axis("off")

    fig.suptitle("指标随训练步的变化（阴影 = 均值的 95% 置信区间）　—— "
                 "两点的 CI 若重叠，那个差异就不显著，别读成改进",
                 x=0.01, ha="left", fontsize=10.5)
    footer(fig, sweep_footer(sw))
    save(fig, out / "10_ckpt_sweep.png", dpi, pdf)


def fig_table(sw: dict, out: Path, dpi, pdf) -> None:
    """排名表。39 个 checkpoint 全都带意义 —— 这种情况 dataviz 要求用表，不是更多颜色。"""
    runs = sw["runs"]
    if not runs:
        return
    keys = [k for k in ("action_rmse", "action_l1", "action_l2", "action_max_abs",
                        "psnr_rd", "ssim_rd", "psnr_rg", "infer_s")
            if any(k in r["summary"]["metrics"] for r in runs)]
    order = sorted(runs, key=lambda r: r["summary"]["metrics"]["action_rmse"]["mean"])

    # 列宽用**英寸**定死，再换成 axes 分数 —— 用相对宽度会让表头文字互相压
    # （action_rmse 这种标签比列宽长，右对齐时会伸到左邻列里去）
    short = {"action_rmse": "rmse", "action_l1": "L1", "action_l2": "L2 (MSE)",
             "action_max_abs": "max|err|", "infer_s": "耗时 s"}
    spec = [("#", 0.34, "left"), ("checkpoint", 1.42, "left"),
            ("step", 0.78, "right"), ("epoch", 0.60, "right")]
    spec += [(short.get(k, k), max(0.72, 0.105 * len(short.get(k, k)) + 0.42), "right")
             for k in keys]
    total = sum(w for _, w, _ in spec) + 0.30
    edges = np.cumsum([0.15] + [w for _, w, _ in spec])      # 每列的右边界（英寸）
    def px(i, align):                                        # 文字锚点 -> axes 分数
        return (edges[i] if align == "left" else edges[i + 1]) / total

    # 每列的最优值，用来加粗
    best = {}
    for k in keys:
        vals = [r["summary"]["metrics"][k]["mean"] for r in runs if k in r["summary"]["metrics"]]
        best[k] = (min(vals) if k in LOWER_BETTER else max(vals)) if vals else None

    rh = 0.235
    fig = plt.figure(figsize=(total, rh * (len(order) + 3.6)))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(order) + 2)
    ytop = len(order) + 1.1

    for i, (name, _, align) in enumerate(spec):
        ax.text(px(i, align), ytop, name, fontsize=8.5, color=INK2, weight="600",
                va="center", ha=align)
    ax.plot([0.004, 0.996], [ytop - 0.5] * 2, color=AXIS, lw=0.9)

    for i, r in enumerate(order):
        y = ytop - 1.1 - i
        M = r["summary"]["metrics"]
        is_top = i == 0
        base = INK if is_top else INK2
        if is_top:                                  # 最优行加一条极淡的底纹
            ax.axhspan(y - 0.45, y + 0.45, xmin=0.004, xmax=0.996, color=SEQ[0], zorder=0)
        cells = [f"{i+1}", r["name"],
                 f"{r['step']:,}" if r.get("step") is not None else "-",
                 f"{r['epoch']:.2f}" if r.get("epoch") else "-"]
        for j, txt in enumerate(cells):
            align = spec[j][2]
            ax.text(px(j, align), y, txt, fontsize=8.5 if j == 1 else 8,
                    color=MUTED if j == 0 else base, va="center", ha=align,
                    weight="600" if (is_top and j == 1) else "normal")
        for j, k in enumerate(keys, start=len(cells)):
            x = px(j, "right")
            if k not in M:
                ax.text(x, y, "-", fontsize=8, color=MUTED, va="center", ha="right"); continue
            v = M[k]["mean"]
            hit = best[k] is not None and abs(v - best[k]) < 1e-12
            ax.text(x, y, f"{v:.4f}", fontsize=8, color=INK if hit else base,
                    va="center", ha="right", weight="600" if hit else "normal")

    ax.set_title("checkpoint 排名（按 action_rmse 升序；加粗 = 该列最优）"
                 "　—— 指标均为均值，物理量；rmse/L1/L2/max|err| 越小越好，psnr/ssim 越大越好",
                 loc="left", fontsize=10.5, pad=12)
    footer(fig, sweep_footer(sw) + "　·　精确读数见 compare.csv")
    save(fig, out / "11_ckpt_table.png", dpi, pdf)


def fig_paired_delta(sw: dict, out: Path, dpi, pdf) -> None:
    """相对基准 ckpt 的**成对**逐样本误差差。

    所有 ckpt 评的是同一批样本，所以可以做配对比较 —— 这比比较两个均值灵敏得多：
    样本难度带来的方差在相减时被消掉了。
    """
    runs = sw["runs"]
    ref_name = sw.get("reference") or runs[-1]["name"]
    ref = next((r for r in runs if r["name"] == ref_name), runs[-1])
    others = [r for r in runs if r["name"] != ref["name"]]
    if not others:
        return

    def by_idx(r):
        return {int(row["idx"]): float(row["action_rmse"]) for row in r["rows"]}

    ref_map = by_idx(ref)
    items = []
    for r in sorted(others, key=lambda r: (r.get("step") is None, r.get("step") or 0)):
        m = by_idx(r)
        common = sorted(set(m) & set(ref_map))
        if not common:
            continue
        d = np.array([m[i] - ref_map[i] for i in common])
        items.append((r["name"], d.mean(), ci95(d), len(common)))
    if not items:
        return

    names = [i[0] for i in items]
    means = np.array([i[1] for i in items])
    halfs = np.array([i[2] for i in items])
    n = items[0][3]

    fig, ax = plt.subplots(figsize=(9.4, 0.34 * len(items) + 2.6))
    ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
    y = np.arange(len(items))[::-1]
    # diverging：更好(负) = blue，更差(正) = red，中点 0 是中性
    cols = [DIV_LO if m < 0 else DIV_HI for m in means]
    # 少数几个 ckpt 时把条形再收窄 —— 否则每个槽位很高，条形看起来是个色块
    ax.barh(y, means, height=0.42 if len(items) >= 6 else 0.24, color=cols)
    ax.errorbar(means, y, xerr=halfs, fmt="none", ecolor=MUTED, elinewidth=1.1,
                capsize=2.5, capthick=1.1)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(-0.75, len(items) - 0.25)

    # 读数放在坐标区右侧的固定一列（跟着条形末端走会被误差棒拉得七零八落）
    tr = ax.get_yaxis_transform()
    for yy, m, h in zip(y, means, halfs):
        sig = abs(m) > h                            # CI 不跨 0 才算显著
        ax.text(1.012, yy, f"{m:+.4f}" + ("" if sig else "  n.s."), transform=tr,
                va="center", ha="left", fontsize=7.8, color=INK if sig else MUTED,
                clip_on=False)
    reach = float(max(np.abs(means) + halfs))
    ax.set_xlim(-reach * 1.15, reach * 1.15)
    ax.set_xlabel(f"Δ action RMSE（相对 {ref['name']}；<0 = 比基准好）")
    better = [nm for nm, m, h in zip(names, means, halfs) if m < 0 and abs(m) > h]
    ax.set_title(f"成对比较：每个 ckpt 相对基准 {ref['name']} 的逐样本误差差"
                 f"（配对 n={n}，误差棒 = 95% CI）\n"
                 + (f"显著优于基准的：{', '.join(better)}" if better
                    else "没有一个显著优于基准 —— 差异都在噪声范围内"),
                 loc="left", fontsize=9.5)
    footer(fig, sweep_footer(sw))
    save(fig, out / "12_paired_delta.png", dpi, pdf)


def _spearman(a, b) -> float:
    def rk(x):
        x = np.asarray(x, float); o = x.argsort()
        r = np.empty_like(o, dtype=float); r[o] = np.arange(x.size); return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial_spearman(a, b, c) -> float:
    """控制 c 之后 a~b 的偏相关（秩空间）。"""
    ab, ac, bc = _spearman(a, b), _spearman(a, c), _spearman(b, c)
    den = np.sqrt(max(1e-12, (1 - ac ** 2) * (1 - bc ** 2)))
    return float((ab - ac * bc) / den)


def fig_video_vs_action(sw: dict, out: Path, dpi, pdf) -> None:
    """视频质量和动作质量到底有没有关系。

    ⚠️ 先记住架构事实：部署走的 `infer_action`（`fastwam.py:1080-1157`）**根本不生成
    未来视频帧** —— 它只把第 0 帧编码成 KV cache，动作去噪循环 cross-attend 这个静态
    cache。所以"视频推得不好导致动作差"在部署路径上**没有物理通道**。
    这里量的是相关性，不是因果。

    关键是要把**样本难度**这个混淆变量摘掉：`psnr_dg`（VAE 重建 vs 真值）只取决于
    场景本身好不好压缩，与模型无关。如果 action 误差与 psnr_rd 的相关在控制 psnr_dg
    后就消失了，那说明"视频差"和"动作差"只是同一批难样本的两个表现。
    """
    runs = sw["runs"]
    ref = next((r for r in runs if r["name"] == sw.get("reference")), runs[-1] if runs else None)
    if ref is None or "psnr_rd" not in ref["summary"].get("metrics", {}):
        return                                    # action 模式没有视频指标

    rows = [r for r in ref["rows"]
            if isinstance(r.get("psnr_rd"), float) and isinstance(r.get("action_rmse"), float)]
    if len(rows) < 8:
        return
    A = np.array([r["action_rmse"] for r in rows])
    RD = np.array([r["psnr_rd"] for r in rows])
    DG = np.array([r["psnr_dg"] for r in rows])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.4, 4.6))

    def scat(ax, x, y, xlab, title):
        ax.grid(True, axis="both", color=GRID, lw=0.8)
        ax.scatter(x, y, s=16, color=S1, alpha=0.55, edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlabel(xlab); ax.set_ylabel("action RMSE（物理量）")
        ax.set_title(title, loc="left", fontsize=9.5)

    r1 = _spearman(A, RD)
    scat(a1, RD, A, "psnr_rd（生成 vs VAE重建，dB）",
         f"① 动作误差 vs 视频质量\nSpearman ρ = {r1:+.3f}（n={len(rows)}）")
    # 四象限：中位线切开，代表样本标出来供人眼比对
    mx, my = float(np.median(RD)), float(np.median(A))
    a1.axvline(mx, color=AXIS, lw=1.0); a1.axhline(my, color=AXIS, lw=1.0)
    quad = {"视频好·动作好": (RD > mx) & (A < my), "视频差·动作好": (RD <= mx) & (A < my),
            "视频好·动作差": (RD > mx) & (A >= my), "视频差·动作差": (RD <= mx) & (A >= my)}
    picks = {}
    for lab, mask in quad.items():
        n = int(mask.sum())
        if not n:
            continue
        # 每象限挑离该象限中心最远的那个，最有代表性
        ii = np.where(mask)[0]
        cx = (RD[ii] - RD[ii].mean()) / (RD.std() + 1e-9)
        cy = (A[ii] - A[ii].mean()) / (A.std() + 1e-9)
        k = int(ii[np.argmax(cx ** 2 + cy ** 2)])
        picks[lab] = (k, n)
        a1.plot([RD[k]], [A[k]], marker="o", markersize=9, color=S2,
                markeredgecolor=SURFACE, markeredgewidth=2.0, linestyle="none", zorder=5)
    # 计数文字放右上 —— 左上是数据最密的区域（低 psnr_rd / 高 rmse），会被点和标记压住
    a1.text(0.98, 0.97, "\n".join(f"{lab}: {n} 个" for lab, (_, n) in picks.items()),
            transform=a1.transAxes, va="top", ha="right", fontsize=7.8, color=INK2)

    r2 = _spearman(A, DG)
    scat(a2, DG, A, "psnr_dg（VAE重建 vs 真值，dB）—— **与模型无关**".replace("**", ""),
         f"② 同样的图，x 换成「场景难度」\nSpearman ρ = {r2:+.3f} —— 和①一样强")

    steps, ar, rd = [], [], []
    for r in runs:
        if r.get("step") is None or "psnr_rd" not in r["summary"]["metrics"]:
            continue
        steps.append(r["step"])
        ar.append(r["summary"]["metrics"]["action_rmse"]["mean"])
        rd.append(r["summary"]["metrics"]["psnr_rd"]["mean"])
    r3 = _spearman(ar, rd) if len(ar) > 4 else float("nan")
    a3.grid(True, axis="both", color=GRID, lw=0.8)
    a3.scatter(rd, ar, s=26, color=S3, alpha=0.75, edgecolor=SURFACE, linewidth=0.6)
    a3.set_xlabel("psnr_rd 均值（dB）"); a3.set_ylabel("action RMSE 均值")
    a3.set_title(f"③ 跨 checkpoint（每点一个 ckpt）\nSpearman ρ = {r3:+.3f}（n={len(ar)}）",
                 loc="left", fontsize=9.5)

    pr = _partial_spearman(A, RD, DG)
    verdict = ("基本无关" if abs(pr) < 0.35 else "有中等相关" if abs(pr) < 0.6 else "强相关")
    fig.suptitle(
        "视频质量与动作质量的关系　—— "
        f"控制场景难度后偏相关 ρ = {pr:+.3f} → **{verdict}**".replace("**", "") + "\n"
        "①的强相关几乎全部来自②那个混淆：难压缩的场景动作也难预测。"
        "部署路径 infer_action 不生成视频，两者之间没有因果通道",
        x=0.01, ha="left", fontsize=10.5)
    footer(fig, f"单 ckpt 部分取自 {ref['name']}　·　" + sweep_footer(sw))
    save(fig, out / "17_video_vs_action.png", dpi, pdf)

    # 四象限代表样本的 mp4 路径写出来，供人眼比对
    if picks:
        txt = out / "17_video_vs_action_samples.txt"
        lines = ["# 四象限代表样本（mp4 内部竖向拼接：上=pred / 中=VAE重建 / 下=GT，7.5fps）",
                 f"# ckpt = {ref['name']}   评测集 = {sw.get('split')} / "
                 f"{sw.get('num_samples')} 样本", ""]
        for lab, (k, n) in picks.items():
            row = rows[k]
            i = ref["rows"].index(row)
            lines.append(f"[{lab}]  该象限 {n} 个样本")
            lines.append(f"  action_rmse={row['action_rmse']:.4f}  psnr_rd={row['psnr_rd']:.2f}"
                         f"  psnr_dg={row['psnr_dg']:.2f}  episode={int(row.get('episode_index',-1))}"
                         f"  frame={int(row.get('frame_in_episode',-1))}")
            lines.append(f"  {ref['dir']}/videos/sample_{i:04d}.mp4")
            lines.append("")
        txt.write_text("\n".join(lines))
        print(f"    {txt.name}")


def fig_sweep_compare(sweeps, out: Path, dpi, pdf) -> None:
    """多个**评测集**在同一批 checkpoint 上的对比 —— 过拟合曲线。

    典型用法：train64 / oldval5 / newval71 三套一起给，第一个当基准。
    左图看绝对误差随训练步怎么走，右图看相对基准的倍数（泛化差距）。

    最多 3 套（分类色只有前 3 槽过 all-pairs 门槛）。
    """
    sweeps = sweeps[:3]
    if len(sweeps) < 2:
        return
    series = []
    for d, sw in sweeps:
        pts = {}
        for r in sw["runs"]:
            if r.get("step") is None:
                continue
            per = [row["action_rmse"] for row in r["rows"]
                   if isinstance(row.get("action_rmse"), float)]
            pts[int(r["step"])] = (float(np.mean(per)), ci95(np.array(per)), len(per))
        if pts:
            series.append((d.name, pts, sw))
    if len(series) < 2:
        return

    common = sorted(set.intersection(*[set(p) for _, p, _ in series]))
    if len(common) < 2:
        return
    steps = np.array(common, dtype=float)
    cols = [S1, S2, S3][:len(series)]
    spe = (series[0][2].get("annotate") or {}).get("steps_per_epoch")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 4.6))

    for (nm, pts, sw), c in zip(series, cols):
        mean = np.array([pts[s][0] for s in common])
        half = np.array([pts[s][1] for s in common])
        n = pts[common[0]][2]
        axA.fill_between(steps, mean - half, mean + half, color=c, alpha=0.13, lw=0)
        axA.plot(steps, mean, color=c, lw=2.0, marker="o", markersize=3.6,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{nm}（n={n}）")
        j = int(mean.argmin())
        axA.plot([steps[j]], [mean[j]], marker="o", markersize=8.5, color=c,
                 markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5, linestyle="none")
        axA.annotate(f"最优 {int(steps[j]):,}", (steps[j], mean[j]), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=7.8, color=INK)
    axA.set_ylim(bottom=0)
    axA.set_xlabel("训练步"); axA.set_ylabel("action RMSE（物理量）")
    axA.legend(fontsize=8.5, loc="upper right")
    axA.set_title("各评测集上的 action RMSE（阴影 = 95% CI）", loc="left",
                  fontsize=9.5, pad=26 if spe else 6)

    base_nm, base_pts, _ = series[0]
    for (nm, pts, _), c in list(zip(series, cols))[1:]:
        ratio = np.array([pts[s][0] / base_pts[s][0] for s in common])
        axB.plot(steps, ratio, color=c, lw=2.0, marker="o", markersize=3.6,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{nm} / {base_nm}")
        axB.annotate(f"{ratio[-1]:.2f}×", (steps[-1], ratio[-1]), xytext=(-30, 6),
                     textcoords="offset points", fontsize=8.5, color=INK, weight="600")
    axB.axhline(1.0, color=AXIS, lw=1.2)
    axB.set_ylim(bottom=1.0)
    axB.set_xlabel("训练步"); axB.set_ylabel(f"倍数（相对 {base_nm}）")
    axB.legend(fontsize=8.5, loc="upper left")
    axB.set_title("泛化差距 —— 曲线上扬 = 过拟合在加重", loc="left",
                  fontsize=9.5, pad=26 if spe else 6)

    if spe:
        for ax in (axA, axB):
            sec = ax.secondary_xaxis("top", functions=(lambda s: s / spe, lambda e: e * spe))
            sec.set_xlabel("epoch", fontsize=8, color=MUTED, labelpad=9)
            sec.tick_params(labelsize=7.5, colors=MUTED)
            sec.spines["top"].set_color(AXIS)

    # 精确读数落一份 CSV，图上不堆数字
    csv_path = out / "compare_sweeps.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch"] + [f"{nm}_rmse" for nm, _, _ in series]
                   + [f"{nm}_over_{base_nm}" for nm, _, _ in series[1:]])
        for s in common:
            row = [s, f"{s/spe:.3f}" if spe else ""]
            row += [f"{pts[s][0]:.5f}" for _, pts, _ in series]
            row += [f"{pts[s][0]/base_pts[s][0]:.4f}" for _, pts, _ in series[1:]]
            w.writerow(row)
    print(f"    {csv_path.name}")

    names = "　vs　".join(nm for nm, _, _ in series)
    fig.suptitle(f"评测集对比：{names}　—— 同一批 {len(common)} 个 checkpoint",
                 x=0.01, ha="left", fontsize=11)
    footer(fig, "各评测集的样本量不同（图例里的 n），所以 CI 宽度不可比；"
                "曲线本身可比，因为同一评测集内所有 ckpt 评的是同一批样本"
                f"　·　精确读数见 {csv_path.name}"
                f"　·　出图于 {_dt.datetime.now().isoformat(timespec='minutes')}")
    save(fig, out / "16_evalset_compare.png", dpi, pdf)


def fig_angle_matrix(sw: dict, out: Path, dpi, pdf) -> None:
    """**全部** checkpoint × 全部维度的平均误差矩阵（热力图 + 数字）。

    为什么是热力图而不是分组条形：39 个 checkpoint 各配一个颜色是不行的
    （分类色只有 3 槽过色盲门槛）。而这里颜色的职责不是"区分身份"而是"表示大小"，
    用单色深浅梯度即可，数量就没有上限了。每格再写上数字，精确读数也不用查 CSV。

    ⚠️ 关节维单位是**度**，夹爪两维是 0–1 开度 —— 两组量纲不同，所以分成
    左右两块、各自独立的色阶，不能共用一个色标。
    """
    runs = sw["runs"]
    if not runs:
        return
    runs = sorted(runs, key=lambda r: (r.get("step") is None, r.get("step") or 0))

    # 每个 ckpt 逐维平均绝对误差：summary.json 里已有，不用再读 npz
    names = runs[0]["summary"].get("action_dim_names") or []
    M = np.array([r["summary"]["abs_err_per_dim_mean"] for r in runs])   # [K, D]
    K, D = M.shape
    if not names or len(names) < D:
        names = [f"dim{d}" for d in range(D)]
    ang = [d for d, nm in enumerate(names) if "gripper" not in nm.lower()]
    grip = [d for d, nm in enumerate(names) if "gripper" in nm.lower()]

    blocks = [("关节（度）", ang, np.degrees(M[:, ang]) if ang else None, "°", "{:.2f}")]
    if grip:
        blocks.append(("夹爪（0–1 开度）", grip, M[:, grip], "", "{:.3f}"))
    blocks = [b for b in blocks if b[2] is not None and len(b[1])]
    if not blocks:
        return

    ratios = [max(1, len(b[1])) for b in blocks]
    fig = plt.figure(figsize=(0.82 * sum(ratios) + 3.0, 0.30 * K + 3.2))
    gs = fig.add_gridspec(1, len(blocks), width_ratios=ratios, wspace=0.14)
    ylabels = [r["name"] for r in runs]

    for bi, (title, dims, V, unit, fmt) in enumerate(blocks):
        ax = fig.add_subplot(gs[0, bi])
        # 单色 light->dark：越深 = 误差越大。sequential 就该一个色相
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f"seq{bi}", SEQ)
        # **按列各自归一化**。各关节量级差 3 倍以上（motor_0_left 约 2–5°，
        # motor_1_left 约 5–12°），共用一个色阶的话早期那行会把量程撑满，
        # 其余 90% 的格子挤成同一个蓝，颜色就不携带信息了。
        # 这里要回答的问题是"同一个关节上哪个 ckpt 最好"，本来就是列内比较。
        lo, hi = V.min(0, keepdims=True), V.max(0, keepdims=True)
        Vn = (V - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
        ax.imshow(Vn, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([names[d] for d in dims], rotation=38, ha="right", fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(ylabels if bi == 0 else [""] * K, fontsize=7.8)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        worst_j = int(np.unravel_index(V.argmax(), V.shape)[1])
        ax.set_title(f"{title}　最差的维：{names[dims[worst_j]]} {V.max():.2f}{unit}",
                     loc="left", fontsize=9.5)
        # 数字直接写格子里：深底用白字、浅底用墨色，保证对比度
        for i in range(K):
            for j in range(len(dims)):
                ax.text(j, i, fmt.format(V[i, j]), ha="center", va="center",
                        fontsize=6.4, color=SURFACE if Vn[i, j] > 0.62 else INK)
        # 每列最优的 ckpt 框出来 —— 描边比小圆点清楚得多（圆点会被数字盖住）
        for j in range(len(dims)):
            i = int(V[:, j].argmin())
            ax.add_patch(matplotlib.patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=S2, lw=1.8, zorder=5))

    n = runs[0]["summary"].get("num_samples", "?")
    best = min(runs, key=lambda r: r["summary"]["metrics"]["action_rmse"]["mean"])
    fig.suptitle(f"每个 checkpoint × 每个维度的平均绝对误差 —— 全部 {n} 个验证样本平均\n"
                 f"颜色**按列**归一化：同一维里最浅 = 最好、最深 = 最差；"
                 f"跨维比大小看数字（各维量级差 3 倍以上，共用色阶颜色就不携带信息了）\n"
                 f"橙框 = 该维最优的 checkpoint　·　整体 action_rmse 最优：{best['name']}"
                 .replace("**", ""),
                 x=0.01, ha="left", fontsize=10.5)
    footer(fig, sweep_footer(sw) + "　·　1 rad = 57.3°；夹爪两维不是角度，单位 0–1 开度")
    save(fig, out / "15_angle_matrix.png", dpi, pdf)


def _mean_err_curves(sw: dict, cap: int = 3):
    """取 highlight 的 ≤cap 个 ckpt，各算出 [T,D] 的跨样本平均绝对误差。

    返回 (sel, means, halfs, n_samples)。多张图共用，避免重复读 npz。
    """
    runs = sw["runs"]
    want = list(sw.get("highlight") or [])[:cap]
    sel = [r for r in runs if r["name"] in want] or runs[:cap]
    if not sel:
        return [], [], [], 0
    pos = [{int(row["idx"]): k for k, row in enumerate(r["rows"])} for r in sel]
    common = sorted(set.intersection(*[set(p) for p in pos]))
    if not common:
        return [], [], [], 0
    means, halfs = [], []
    for r, p in zip(sel, pos):
        errs = []
        for i in common:
            z = np.load(r["samples"][p[i]])
            errs.append(np.abs(z["pred_action_phys"] - z["gt_action_phys"]))
        E = np.stack(errs)                                    # [N, T, D]
        n = E.shape[0]
        means.append(E.mean(0))
        halfs.append(_tcrit(n) * E.std(0, ddof=1) / np.sqrt(n) if n > 1
                     else np.zeros_like(E[0]))
    return sel, means, halfs, len(common)


def fig_angle_error(sw: dict, out: Path, dpi, pdf) -> None:
    """关节角度误差，单位换成**度** —— 比 rad 直观得多（0.1 rad 要在脑子里换算成 5.7°）。

    ⚠️ 14 个 action 维里**只有 12 个 motor 维是弧度**；两个 gripper_width 维是
    0–1 归一化开度（实测取值 -0.08~0.92），换算成度会得出没有意义的数字。
    所以这里把它们分开画，夹爪那格保留自己的单位。
    """
    sel, means, halfs, n = _mean_err_curves(sw)
    if not sel:
        return
    names = sel[0]["summary"].get("action_dim_names") or \
        [f"dim{d}" for d in range(means[0].shape[1])]
    ang = [d for d, nm in enumerate(names) if "gripper" not in nm.lower()]
    grip = [d for d, nm in enumerate(names) if "gripper" in nm.lower()]
    if not ang:
        return
    dt = float((sw.get("annotate") or {}).get("action_dt_s") or 0.0)
    T = means[0].shape[0]
    t = np.arange(T) * dt if dt else np.arange(1, T + 1)
    cols = [S1, S2, S3][:len(sel)]

    per_joint = [np.degrees(m[:, ang].mean(0)) for m in means]      # [12] 度
    horizon = [np.degrees(m[:, ang].mean(1)) for m in means]        # [T]  度
    hz_half = [np.degrees(h[:, ang].mean(1)) for h in halfs]

    fig = plt.figure(figsize=(13.6, 0.42 * len(ang) + 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1])
    axL = fig.add_subplot(gs[:, 0])
    axT = fig.add_subplot(gs[0, 1])
    axG = fig.add_subplot(gs[1, 1]) if grip else None

    # ---- 左：逐关节平均误差（度），分组横向条形 ----
    axL.grid(axis="x", color=GRID, lw=0.8); axL.grid(axis="y", visible=False)
    y = np.arange(len(ang))[::-1]                       # 自上而下按关节顺序
    slot = 0.74 / len(sel)
    for k, (r, v) in enumerate(zip(sel, per_joint)):
        off = (k - (len(sel) - 1) / 2) * slot
        # height 只占槽位的 0.82：留**真实间隙**，而不是给条形描边
        axL.barh(y - off, v, height=slot * 0.82, color=cols[k], label=r["name"])
        for yy, vv in zip(y - off, v):
            axL.text(vv, yy, f" {vv:.2f}°", va="center", fontsize=7.2, color=INK2)
    axL.set_yticks(y); axL.set_yticklabels([names[d] for d in ang], fontsize=8.5)
    axL.set_xlim(0, max(v.max() for v in per_joint) * 1.20)
    axL.set_xlabel("平均绝对误差（度）")
    worst = int(np.argmax(per_joint[-1]))
    axL.set_title(f"逐关节平均角度误差 —— 最差是 {names[ang[worst]]}"
                  f"（{per_joint[-1][worst]:.2f}°）", loc="left", fontsize=9.5)
    axL.legend(fontsize=8, loc="lower right")

    # ---- 右上：角度误差随预测时域（度，12 关节平均）----
    for k, (r, v, hf) in enumerate(zip(sel, horizon, hz_half)):
        axT.fill_between(t, v - hf, v + hf, color=cols[k], alpha=0.12, lw=0)
        axT.plot(t, v, color=cols[k], lw=1.8, label=r["name"])
    v0, v1 = horizon[-1][0], horizon[-1][-1]
    axT.annotate(f"{v1:.2f}°", (t[-1], v1), xytext=(-30, 6), textcoords="offset points",
                 fontsize=8.5, color=INK, weight="600")
    axT.annotate(f"{v0:.2f}°", (t[0], v0), xytext=(4, -12), textcoords="offset points",
                 fontsize=8.5, color=INK2)
    axT.set_ylim(bottom=0)
    axT.set_ylabel("平均绝对误差（度）")
    axT.set_xlabel(f"预测时域（秒）" if dt else "预测步")
    if dt:
        step_twin(axT, dt, T)
    axT.set_title(f"角度误差随预测时域增长 —— 首步 {v0:.2f}° → 末步 {v1:.2f}°"
                  f"（{v1/v0 if v0 else float('nan'):.1f}×）",
                  loc="left", fontsize=9.5, pad=26)

    # ---- 右下：夹爪单独一格，保留自己的单位 ----
    if axG is not None:
        for k, (r, m) in enumerate(zip(sel, means)):
            axG.plot(t, m[:, grip].mean(1), color=cols[k], lw=1.8, label=r["name"])
        axG.set_ylim(bottom=0)
        axG.set_ylabel("开度误差（0–1）")
        axG.set_xlabel("秒" if dt else "预测步")
        gv = means[-1][:, grip].mean(1)
        axG.set_title(f"夹爪开度误差（{'、'.join(names[d] for d in grip)}）—— "
                      f"**不是角度**，单位是 0–1 归一化宽度".replace("**", ""),
                      loc="left", fontsize=9)
        axG.annotate(f"{gv[-1]:.3f}", (t[-1], gv[-1]), xytext=(-30, 6),
                     textcoords="offset points", fontsize=8.5, color=INK2)

    overall = "　·　".join(
        f"{r['name']}：12 关节整体平均 {v.mean():.2f}°"
        for r, v in zip(sel, per_joint))
    fig.suptitle(f"关节角度误差（度）—— 全部 {n} 个验证样本平均\n" + overall,
                 x=0.01, ha="left", fontsize=11)
    footer(fig, sweep_footer(sw) + "　·　1 rad = 57.3°；夹爪两维不是角度，见右下格")
    save(fig, out / "14_angle_error_deg.png", dpi, pdf)


def fig_dim_error(sw: dict, out: Path, dpi, pdf) -> None:
    """逐维误差随预测时域 —— **全部样本平均**，按 checkpoint 分线。

    为什么画误差而不是平均轨迹：64 个样本各自起始于不同 episode 的不同时刻、
    机械臂在不同位姿，把关节角跨样本求平均得到的是一个"平均姿态"，不对应任何
    真实动作，GT 的平均也一样是一团糊。能求平均又有物理意义的是**误差**。

    最多 3 个 ckpt —— 分类色只有前 3 槽过 all-pairs 色盲门槛。
    """
    sel, means, halfs, n = _mean_err_curves(sw)
    if not sel:
        return
    curves = list(zip(sel, means, halfs))

    D = curves[0][1].shape[1]
    T = curves[0][1].shape[0]
    names = sel[0]["summary"].get("action_dim_names") or [f"dim{d}" for d in range(D)]
    dt = float((sw.get("annotate") or {}).get("action_dt_s") or 0.0)
    t = np.arange(T) * dt if dt else np.arange(1, T + 1)

    ncol, nrow = 4, int(np.ceil(D / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.15 * nrow), sharex=True)
    cols = [S1, S2, S3][:len(curves)]
    for d in range(nrow * ncol):
        a = axes.ravel()[d]
        if d >= D:
            a.axis("off"); continue
        for (r, mean, half), c in zip(curves, cols):
            a.fill_between(t, mean[:, d] - half[:, d], mean[:, d] + half[:, d],
                           color=c, alpha=0.12, lw=0)
            a.plot(t, mean[:, d], color=c, lw=1.8, label=r["name"])
        # 逐维总体 MAE 放标题，省得每根线都标数
        maes = "  ".join(f"{mean[:, d].mean():.3f}" for _, mean, _ in curves)
        a.set_title(f"{names[d] if d < len(names) else f'dim{d}'}   MAE {maes}",
                    loc="left", fontsize=8.5)
        a.set_ylim(bottom=0)
        a.tick_params(labelsize=7.5)
        if d < ncol and dt:
            step_twin(a, dt, T)
        if d >= D - ncol:
            a.set_xlabel("秒" if dt else "预测步", fontsize=8)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[0].legend(h, l, fontsize=7.5, loc="best")

    overall = "　·　".join(
        f"{r['name']}：全 {n} 样本 RMSE {r['summary']['metrics']['action_rmse']['mean']:.4f}"
        for r, _, _ in curves)
    fig.suptitle(f"逐维平均绝对误差随预测时域的变化 —— 全部 {n} 个验证样本平均"
                 f"（阴影 = 95% CI）\n"
                 "标题里的 MAE 按图例顺序，是该维在整个时域上的平均\n" + overall,
                 x=0.01, ha="left", fontsize=10)
    footer(fig, sweep_footer(sw))
    save(fig, out / "13_dim_error_by_ckpt.png", dpi, pdf)


# ------------------------------------------------------------------ 主流程
def draw_single(run: dict, out: Path, dpi: int, pdf: bool) -> None:
    S, rows, samples = run["summary"], run["rows"], run["samples"]
    fig_overview(S, rows, out, dpi, pdf)
    fig_per_dim(S, out, dpi, pdf)
    fig_per_step(S, out, dpi, pdf)
    fig_video(S, rows, out, dpi, pdf)
    if samples:
        rr = [r.get("action_rmse", float("nan")) for r in rows]
        fig_traj(S, rows, samples, out, dpi, pdf, "误差最小的样本", int(np.nanargmin(rr)))
        fig_traj(S, rows, samples, out, dpi, pdf, "误差最大的样本", int(np.nanargmax(rr)))


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="把 offline_eval.py 的结果画成 PNG")
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="单 ckpt 结果目录，或 --sweep 的扫描目录")
    ap.add_argument("--sweep", action="store_true",
                    help="目录是多 ckpt 扫描的产出（含 sweep.json）；有 sweep.json 时会自动识别")
    ap.add_argument("--out", type=Path, default=None, help="图片输出目录，默认第一个 DIR 下的 figs/")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pdf", action="store_true", help="另存矢量 PDF（放论文/汇报用）")
    a = ap.parse_args()

    d0 = a.dirs[0]
    is_sweep = a.sweep or (d0 / "sweep.json").is_file()
    if not is_sweep:
        for d in a.dirs:
            if not (d / "summary.json").exists():
                print(f"[错误] {d} 下既没有 summary.json 也没有 sweep.json —— "
                      f"先跑 scripts/offline_eval.py")
                return 1
    elif not (d0 / "sweep.json").is_file():
        print(f"[错误] {d0}/sweep.json 不存在 —— 先跑 offline_eval.py --config，"
              f"或用 --merge-only 补出来")
        return 1

    setup_style()
    out = a.out or (d0 / "figs")
    out.mkdir(parents=True, exist_ok=True)
    print(f">>> 出图到 {out}")

    if is_sweep:
        sweeps = []
        for d in a.dirs:
            if not (d / "sweep.json").is_file():
                print(f"[错误] {d}/sweep.json 不存在（第一个目录是 sweep，其余也必须是）")
                return 1
            sweeps.append((d, load_sweep(d)))
        sw = sweeps[0][1]
        if not sw["runs"]:
            print("[错误] sweep.json 里没有任何可读的 ckpt 结果")
            return 1
        # 单 ckpt 图用基准那一份（不给基准就用最后一个）
        ref = next((r for r in sw["runs"] if r["name"] == sw.get("reference")), sw["runs"][-1])
        print(f"    单 ckpt 图取自 {ref['name']}")
        draw_single(ref, out, a.dpi, a.pdf)
        fig_sweep(sw, out, a.dpi, a.pdf)
        fig_table(sw, out, a.dpi, a.pdf)
        fig_paired_delta(sw, out, a.dpi, a.pdf)
        fig_dim_error(sw, out, a.dpi, a.pdf)
        fig_angle_error(sw, out, a.dpi, a.pdf)
        fig_angle_matrix(sw, out, a.dpi, a.pdf)
        fig_video_vs_action(sw, out, a.dpi, a.pdf)
        if len(sweeps) > 1:
            # 给了多个 sweep 目录 = 想比较不同**评测集**（train vs val …）。
            # 以前这里会静默只用第一个目录，其余被丢掉。
            fig_sweep_compare(sweeps, out, a.dpi, a.pdf)
        tail = f"{d0}/compare.csv"
    else:
        runs = [load_run(d) for d in a.dirs]
        draw_single(runs[0], out, a.dpi, a.pdf)
        if len(runs) > 1:
            sw = runs_to_sweep(runs)
            fig_sweep(sw, out, a.dpi, a.pdf)
            fig_table(sw, out, a.dpi, a.pdf)
            fig_paired_delta(sw, out, a.dpi, a.pdf)
            fig_dim_error(sw, out, a.dpi, a.pdf)
            fig_angle_error(sw, out, a.dpi, a.pdf)
            fig_angle_matrix(sw, out, a.dpi, a.pdf)
            fig_video_vs_action(sw, out, a.dpi, a.pdf)
        tail = f"{a.dirs[0]}/metrics.csv"

    print(f"\n>>> 完成，共 {len(list(out.glob('*.png')))} 张图")
    print(f"    Jupyter / VS Code / 图片浏览器直接打开 {out}")
    print(f"    精确读数看 {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
