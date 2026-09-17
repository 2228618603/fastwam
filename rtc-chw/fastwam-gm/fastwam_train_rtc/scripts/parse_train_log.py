#!/usr/bin/env python
"""解析 FastWAM 训练日志，抽出 loss / eval 指标，并支持两个 run 的 A/B 对比。

为什么需要这个脚本：trainer 用 rich 输出，日志有三个坑
  1. 带 ANSI 颜色码
  2. 会按宽度**折行**，`[eval] step=N val_loss=... action_l2=...` 会被拆到多行
  3. `[train]` / `[eval]` 这类前缀被 rich 重排，直接 grep 抓不到
再加上容器默认 NCCL_DEBUG=INFO，日志里绝大部分是 NCCL 噪声。

用法:
  python scripts/parse_train_log.py runs/<task>/ab_A/train.log
  python scripts/parse_train_log.py runs/<task>/ab_A/train.log --csv out.csv
  python scripts/parse_train_log.py A=runs/.../ab_A/train.log B=runs/.../ab_B/train.log --compare
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = re.compile(r"NCCL INFO|NCCL WARN|^[a-z0-9-]+:\d+:\d+ \[")


def _clean(path: Path) -> str:
    """去 ANSI、去 NCCL 噪声，并把折行重新拼成逻辑行。

    rich 的续行是缩进的（没有 "MM/DD [HH:MM:SS]" 或 "INFO" 标记），
    所以遇到不含 '|  >>' 也不含时间戳的缩进行就并到上一行。
    """
    out: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", raw)
        if NOISE.search(line):
            continue
        if not line.strip():
            continue
        is_new = bool(re.match(r"^\d+/\d+ \[", line)) or "INFO" in line or "WARNING" in line
        if is_new or not out:
            out.append(line.rstrip())
        else:
            out[-1] += " " + line.strip()
    return "\n".join(out)


def parse(path: Path) -> dict:
    text = _clean(path)

    train: list[dict] = []                       # {step, loss, loss_action, loss_video, lr, samples_per_sec}
    evals: list[dict] = []                       # eval 指标
    times: dict[int, int] = {}                   # step -> 秒

    for line in text.splitlines():
        ts = re.search(r"\[(\d+):(\d+):(\d+)\]", line)
        sec = None
        if ts:
            h, m, s = map(int, ts.groups())
            sec = h * 3600 + m * 60 + s

        # 注意：不能用 `step=N/M\s+loss=` 这种连写的正则。
        # rich 在 step>=1000 后会折行，把 `trainer.py:738` 塞到 step= 和 loss= 之间，
        # 合并后长这样：`... step=1000/5000  trainer.py:738 loss=0.5239 loss_action=...`
        # 所以 step 和各个 loss 字段必须**分别**匹配。
        ms = re.search(r"epoch=(\d+)\s+step=(\d+)/(\d+)", line)
        if ms and "loss=" in line:
            step = int(ms.group(2))
            rec: dict = {"step": step}
            for key in ["loss_action", "loss_video", "loss"]:
                # loss= 要用边界，否则会被 loss_action= 抢先匹配
                mm = re.search(rf"(?<![a-z_]){key}=([0-9.]+)", line)
                if mm:
                    rec[key] = float(mm.group(1))
            mlr = re.search(r"lr=([0-9.eE+-]+)", line)
            if mlr:
                try:
                    rec["lr"] = float(mlr.group(1))
                except ValueError:
                    pass
            msp = re.search(r"([0-9.]+)\s+samples/s", line)
            if msp:
                rec["samples_per_sec"] = float(msp.group(1))
            if "loss" in rec:
                train.append(rec)
                if sec is not None:
                    times[step] = sec
            continue

        me = re.search(r"step=(\d+)\s+val_loss=([0-9.]+)", line)
        if me:
            rec = {"step": int(me.group(1)), "val_loss": float(me.group(2))}
            for key in ["infer_psnr", "infer_ssim", "action_l2", "action_l1"]:
                mm = re.search(rf"{key}=([0-9.]+)", line)
                if mm:
                    rec[key] = float(mm.group(1))
            evals.append(rec)

    # 稳态步时间：跳过前 5 步预热。注意这个值**包含** eval 的开销，
    # 想要纯训练步时间就看日志里的 samples/s 字段。
    steptime = None
    warm = sorted(s for s in times if s >= 6)
    if len(warm) >= 2:
        s0, s1 = warm[0], warm[-1]
        if s1 > s0:
            steptime = (times[s1] - times[s0]) / (s1 - s0)

    done = "max_steps reached" in text or "training finished" in text
    err = len(re.findall(r"Error processing sample", text))
    oom = bool(re.search(r"OutOfMemoryError|CUDA out of memory", text))

    return {
        "path": str(path), "train": train, "evals": evals,
        "steptime": steptime, "done": done, "errors": err, "oom": oom,
    }


def _tail_mean(items: list[dict], key: str, n: int) -> float | None:
    vals = [e[key] for e in items if key in e][-n:]
    return sum(vals) / len(vals) if vals else None


def show(r: dict, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"\n=== {tag}{r['path']} ===")
    print(f"  完成: {r['done']}   Error processing sample: {r['errors']}   OOM: {r['oom']}")
    if r["steptime"]:
        print(f"  步时间(含 eval 开销): {r['steptime']:.2f} s")
    sps = _tail_mean(r["train"], "samples_per_sec", 20)
    if sps:
        print(f"  末 20 点吞吐: {sps:.1f} 样本/s")
    if r["train"]:
        first, last = r["train"][0], r["train"][-1]
        print(f"  train loss: step {first['step']} = {first['loss']:.4f}"
              f"  ->  step {last['step']} = {last['loss']:.4f}   (共 {len(r['train'])} 个点)")
        for k in ["loss", "loss_action", "loss_video"]:
            v = _tail_mean(r["train"], k, 20)
            if v is not None:
                print(f"    末 20 点 {k:12s} 均值: {v:.4f}")
    if r["evals"]:
        print(f"  eval 点数: {len(r['evals'])}")
        print("  " + f"{'step':>7} {'val_loss':>9} {'psnr':>7} {'ssim':>7} {'action_l2':>10} {'action_l1':>10}")
        for e in r["evals"]:
            print("  " + f"{e['step']:>7} {e.get('val_loss',float('nan')):>9.4f} "
                         f"{e.get('infer_psnr',float('nan')):>7.2f} {e.get('infer_ssim',float('nan')):>7.4f} "
                         f"{e.get('action_l2',float('nan')):>10.4f} {e.get('action_l1',float('nan')):>10.4f}")


def compare(runs: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print(" A/B 对比")
    print(" 主看 train loss_action（每步在真实 batch 上算，噪声最小）")
    print(" action_l2 每个 eval 点只有 8 个样本（trainer.py:409 每 rank 采 1 个），噪声大，只作辅证")
    print("=" * 78)
    names = list(runs)
    print(f"\n{'指标':<36}" + "".join(f"{n:>13}" for n in names) + f"{'B-A 变化':>14}")
    print("-" * 78)

    def row(label, fn, fmt="{:.4f}", lower_better=True):
        vals = [fn(runs[n]) for n in names]
        cells = "".join(fmt.format(v).rjust(13) if v is not None else "?".rjust(13) for v in vals)
        delta = ""
        if len(vals) == 2 and None not in vals and vals[0]:
            pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
            better = (pct < 0) if lower_better else (pct > 0)
            delta = f"{pct:+.1f}% {'✓优' if better else '✗劣'}"
        print(f"{label:<36}{cells}{delta:>14}")

    print("--- 主指标（train，每步真实 batch）---")
    row("末 50 点 loss_action 均值", lambda r: _tail_mean(r["train"], "loss_action", 50))
    row("末 50 点 loss_video 均值", lambda r: _tail_mean(r["train"], "loss_video", 50))
    row("末 50 点 loss 总均值", lambda r: _tail_mean(r["train"], "loss", 50))
    print("--- 辅指标（eval，每点仅 8 样本）---")
    row("末 10 个 eval action_l2 均值", lambda r: _tail_mean(r["evals"], "action_l2", 10))
    row("末 10 个 eval action_l1 均值", lambda r: _tail_mean(r["evals"], "action_l1", 10))
    row("末 10 个 eval val_loss 均值", lambda r: _tail_mean(r["evals"], "val_loss", 10))
    row("末 10 个 eval psnr 均值", lambda r: _tail_mean(r["evals"], "infer_psnr", 10),
        fmt="{:.2f}", lower_better=False)
    print("--- 性能 ---")
    row("步时间 (s, 含 eval)", lambda r: r["steptime"], fmt="{:.2f}")

    # 热启动的价值主要体现在**收敛更快**，所以"达到同一 loss 需要多少步"
    # 比"末端 loss 谁低"更贴近决策。以 A 的末端水平为基准，看 B 多早达到。
    if len(names) == 2:
        a, b = runs[names[0]], runs[names[1]]
        for key in ["loss_action", "loss"]:
            target = _tail_mean(a["train"], key, 50)
            if target is None:
                continue
            print(f"\n--- 收敛速度：达到 A 末端 {key}={target:.4f} 所需步数 ---")
            for nm, r in [(names[0], a), (names[1], b)]:
                # 用 20 点滑动均值穿越阈值的位置，避免单点噪声误判
                pts = [(t["step"], t[key]) for t in r["train"] if key in t]
                hit = None
                for i in range(19, len(pts)):
                    win = [v for _, v in pts[i - 19:i + 1]]
                    if sum(win) / len(win) <= target:
                        hit = pts[i][0]
                        break
                print(f"  {nm:<28}{(str(hit)+' 步') if hit else '未达到':>14}")

    print("\n判读规则（AGILEX_TRAINING_PLAN.md Phase 6）：")
    print("  B 在 loss_action 与 action_l2 上均优 ≥10% -> 正式训练用 B")
    print("  差距 <5%（落在噪声内）                    -> 用 A")
    print("  B 明显更差                                -> 用 A，并考虑只加载 action expert")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="日志路径，或 LABEL=路径")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    runs: dict[str, dict] = {}
    for spec in args.logs:
        label, _, p = spec.partition("=") if "=" in spec else ("", "", spec)
        path = Path(p)
        if not path.exists():
            print(f"[warn] 不存在: {path}", file=sys.stderr)
            continue
        r = parse(path)
        key = label or path.parent.name
        runs[key] = r
        show(r, key)

    if args.compare and len(runs) >= 2:
        compare(runs)

    if args.csv and runs:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run","kind","step","loss/val_loss","loss_action","loss_video",
                        "lr","psnr","ssim","action_l2","action_l1"])
            for name, r in runs.items():
                for t in r["train"]:
                    w.writerow([name, "train", t["step"], t.get("loss",""),
                                t.get("loss_action",""), t.get("loss_video",""),
                                t.get("lr",""), "", "", "", ""])
                for e in r["evals"]:
                    w.writerow([name, "eval", e["step"], e.get("val_loss",""), "", "", "",
                                e.get("infer_psnr",""), e.get("infer_ssim",""),
                                e.get("action_l2",""), e.get("action_l1","")])
        print(f"\nCSV 已写出: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
