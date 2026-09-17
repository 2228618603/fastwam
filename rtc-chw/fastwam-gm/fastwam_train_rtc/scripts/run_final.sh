#!/usr/bin/env bash
# 正式训练（final_A）的无人值守守护：等 B 结束 -> 启动 -> 监控 -> 崩溃自动续训 -> 清理磁盘。
#
# 为什么需要它：Claude 会话结束后，chain_ab.sh 只做到「生成 A/B 对比表」就退出了，
# 没有任何东西会启动正式训练、监控 42 小时的运行、或在半夜崩溃后恢复。
# 这个脚本用 setsid detach（PPID=1）补掉这个缺口。
#
# 已确认的决策（不再依赖 A/B 结论）：
#   final_A 采用 **A 方案** = 纯 Wan2.2 初始化，resume 保持 null。
#   A/B 对比只作为结论产出，不改变 final_A 的配置。
#
# 训练规格（configs/task/agilex_final_3cam_384_1e-4.yaml）：
#   5 epoch = 77,690 步 ≈ 42 小时 / bs8 x 8卡 / warmup 2% / save_every 2000
#
# 磁盘安全（关键）：单次保存 92GB（weights 12 + ZeRO state 80），38 次 = 3.5TB。
# 共享磁盘只剩约 850G 且其他用户在增长 —— 所以每轮监控都跑 prune_ckpt.sh，
# 只留最近 1 份 state + 3 份 weights（稳态约 116GB）。
#
# 用法:
#   bash scripts/run_final.sh start
#   bash scripts/run_final.sh status
#   bash scripts/run_final.sh stop        # 停守护，不杀训练
#   bash scripts/run_final.sh abort       # 停守护 + 杀训练
#
# ---------------------------------------------------------------------------
#  可覆盖的环境变量（默认值 = 原 final_A 行为，不传就完全等价）
# ---------------------------------------------------------------------------
#  RID           run 目录名与 wandb run 名，默认 final_A
#  INIT_RESUME   第 0 次启动的 resume 值。**文件**=只加载权重、step 从 0 开始
#                （`trainer.py:315-327`）。默认空 = 纯 Wan2.2 初始化。
#                ⚠️ 崩溃续训时不会再用它，而是用最新的 state **目录**恢复完整状态，
#                   否则会把已经训练的进度丢掉、从 step 0 重来。
#  WGROUP        wandb group，默认 final_A。同 group 的 run 在 wandb 里可叠图。
#  SKIP_PRELUDE  =1 时跳过「等 ab_B + 补跑 A/B 离线评测」那段一次性前奏
#                （只对 final_A 首次运行有意义）
#
#  例：用具身预训练权重热启动，与 final_A 叠图对比
#    RID=final_D WGROUP=final_wanpre SKIP_PRELUDE=1 \
#      INIT_RESUME=./checkpoints/wanpre60k_video_only.pt \
#      bash scripts/run_final.sh start

set -uo pipefail
cd /home/gaomeng/FastWAM

TASK=agilex_final_3cam_384_1e-4
RID="${RID:-final_A}"
INIT_RESUME="${INIT_RESUME:-}"
WGROUP="${WGROUP:-final_A}"
SKIP_PRELUDE="${SKIP_PRELUDE:-0}"
RUNDIR="runs/$TASK/$RID"
AB_B_LOG="runs/agilex_uncond_3cam_384_1e-4/ab_B/train.log"

# 按 RID 分开，避免两个 run 的守护互相误判（_running_pid 读 PIDFILE 判活）
PIDFILE=".run_final.${RID}.pid"
LOGFILE=".run_final.${RID}.log"
ATTEMPTFILE="$RUNDIR/.attempts"

# 兼容：final_A 那次用的是不带 RID 的旧文件名，沿用以便 status/stop 仍能管到它
if [[ "$RID" == "final_A" ]]; then
  PIDFILE=".run_final.pid"
  LOGFILE=".run_final.log"
fi

MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"      # 崩溃自动续训的次数上限
CHECK_EVERY="${CHECK_EVERY:-300}"      # 监控间隔（秒）
DISK_MIN_GB="${DISK_MIN_GB:-150}"      # 磁盘低于此值就激进清理

_log() { echo "$(date '+%F %T') $*" >> "$LOGFILE"; }
# 不能用 pgrep -f "run_final.sh _daemon" —— 调用方的命令行里只要出现这个字符串
# （例如 grep 模式、ps 管道）就会被匹配到自己，曾导致 kill -9 杀掉自己的 shell。
# 改为 PIDFILE 为准 + 读 /proc/<pid>/cmdline 校验（防 PID 复用）。
_running_pid() {
  local p
  [[ -f "$PIDFILE" ]] || return 0
  p=$(cat "$PIDFILE" 2>/dev/null); [[ -n "$p" ]] || return 0
  if [[ -r "/proc/$p/cmdline" ]] \
     && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "run_final.sh _daemon"; then
    echo "$p"
  fi
}
_train_running() { pgrep -f "scripts/train.py" > /dev/null 2>&1; }
_clean() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null; }

_done_ok()   { _clean "$RUNDIR/train.log" | grep -qE "max_steps reached|training finished"; }
_has_oom()   { _clean "$RUNDIR/train.log" | grep -qE "OutOfMemoryError|CUDA out of memory"; }
_has_nan()   { _clean "$RUNDIR/train.log" | grep -qE "loss=nan|loss=inf|loss=-inf"; }
_cur_step()  { _clean "$RUNDIR/train.log" | grep -oE "step=[0-9]+/" | tail -1 | tr -dc 0-9; }
_disk_gb()   { df -BG / | tail -1 | awk '{gsub("G","",$4); print $4}'; }

_latest_state() {
  find "$RUNDIR/checkpoints/state" -maxdepth 1 -mindepth 1 -type d -name 'step_*' \
    2>/dev/null | sort | tail -1
}

# ---------------------------------------------------------------- 启动一次训练
_launch() {
  local resume_arg="$1" attempt="$2" wname="$3"
  mkdir -p "$RUNDIR"
  source env.sh > /dev/null 2>&1        # PATH(accelerate) / NCCL_DEBUG / WANDB_API_KEY

  local extra=()
  [[ -n "$resume_arg" ]] && extra+=("resume=$resume_arg")

  _log "启动训练 attempt=$attempt resume=${resume_arg:-null} wandb.name=$wname"
  RUN_ID="$RID" bash scripts/train_zero1.sh 8 task=$TASK \
    "${extra[@]}" \
    wandb.enabled=true wandb.project=fastwam-agilex "wandb.name=$wname" \
    "wandb.group=$WGROUP" \
    >> "$RUNDIR/train.log" 2>&1
  local rc=$?
  _log "训练进程退出 rc=$rc"
  return $rc
}

# ---------------------------------------------------------------- 磁盘守护
_guard_disk() {
  local gb; gb=$(_disk_gb)
  bash scripts/prune_ckpt.sh "$RUNDIR" >> "$LOGFILE" 2>&1
  local after; after=$(_disk_gb)
  if (( after < DISK_MIN_GB )); then
    _log "⚠️ 清理后磁盘仍只剩 ${after}G（阈值 ${DISK_MIN_GB}G）。改为只留 1 份 weights。"
    KEEP_STATE=1 KEEP_WEIGHTS=1 bash scripts/prune_ckpt.sh "$RUNDIR" >> "$LOGFILE" 2>&1
    after=$(_disk_gb)
    if (( after < 60 )); then
      _log "❌ 磁盘仅剩 ${after}G，继续训练会写失败。停止守护，请人工处理"
      _log "   提示：本机磁盘为多人共享（/root/litianyu 约 1.9TB、/root/zzd 等）"
      return 1
    fi
  fi
  [[ "$gb" != "$after" ]] && _log "磁盘 ${gb}G -> ${after}G"
  return 0
}

_daemon_main() {
  _log "================ 守护启动 ================"
  _log "run=$RID · 初始化 resume=${INIT_RESUME:-null（纯 Wan2.2）} · wandb.group=$WGROUP"
  _log "5 epoch = 77,690 步 ≈ 42h"
  _log "崩溃续训上限 $MAX_ATTEMPTS 次 · 监控间隔 ${CHECK_EVERY}s · 磁盘下限 ${DISK_MIN_GB}G"

  local waited=0
if [[ "$SKIP_PRELUDE" == "1" ]]; then
  # ---- 1'. 只等 GPU 空闲（跳过 final_A 那次的一次性前奏）----
  _log "SKIP_PRELUDE=1：跳过「等 ab_B + 补跑 A/B 离线评测」，只等 GPU 空闲"
  while _train_running; do
    if (( waited > 86400 )); then
      _log "❌ 等其他训练进程退出超过 24h，放弃。请人工检查"
      rm -f "$PIDFILE"; exit 1
    fi
    _log "等待其他训练进程退出（已等 ${waited}s）"; sleep 60; waited=$((waited+60))
  done
  sleep 45   # 等显存真正释放
  _log "GPU 空闲 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1) MiB"
else
  # ---- 1. 等 ab_B 与显存（带总上限，防 ab_B 挂死后无限等待）----
  while :; do
    if (( waited > 14400 )); then      # 4 小时
      _log "⚠️ 等 ab_B 超过 4 小时仍未完成，不再等待，直接进入正式训练流程"
      break
    fi
    if [[ -f "$AB_B_LOG" ]] && ! _clean "$AB_B_LOG" | grep -qE "max_steps reached|training finished"; then
      _log "等待 ab_B 完成（当前 $(_clean "$AB_B_LOG" | grep -oE 'step=[0-9]+/5000' | tail -1)）"
      sleep 180; waited=$((waited+180)); continue
    fi
    if _train_running; then _log "等待其他训练进程退出"; sleep 60; waited=$((waited+60)); continue; fi
    break
  done
  sleep 45   # 等显存真正释放
  _log "前置条件满足，显存空闲 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1) MiB"

  # ---- 1.4 安全网：兜底 chain_ab.sh 的收尾工作 ----
  # 我在 chain_ab.sh 运行期间修改过该文件，而 bash 是增量读取脚本的，
  # 原地改动有让运行中实例错位的风险。所以这里不假设它一定完成了收尾。
  B_DIR="runs/agilex_uncond_3cam_384_1e-4/ab_B"
  if [[ -d "$B_DIR/checkpoints/state" ]]; then
    SZ=$(du -sh "$B_DIR/checkpoints/state" 2>/dev/null | cut -f1)
    rm -rf "$B_DIR/checkpoints/state"
    _log "[安全网] 删除 ab_B 的 ZeRO state ($SZ)，磁盘 -> $(_disk_gb)G"
  fi
  if [[ ! -f "runs/agilex_uncond_3cam_384_1e-4/ab_compare.csv" ]]; then
    _log "[安全网] 生成 A/B 对比（chain_ab 未产出）"
    "${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}/bin/python" scripts/parse_train_log.py \
      A="runs/agilex_uncond_3cam_384_1e-4/ab_A/train.log" \
      B="$B_DIR/train.log" --compare \
      --csv "runs/agilex_uncond_3cam_384_1e-4/ab_compare.csv" >> "$LOGFILE" 2>&1 \
      || _log "[安全网] 对比生成失败"
  fi

  # ---- 1.5 在 A/B 的 checkpoint 上跑离线评测（约 6 分钟）----
  # 一举三得：① 给出比训练期 eval 强得多的 A/B 结论（32 样本 vs 8 样本）
  #          ② 顺便验证 offline_eval.py（它写完时 GPU 被占，从未在真实数据上跑过）
  #          ③ 此刻是唯一的 GPU 空窗，final_A 一开始就没机会了
  # 用 || true 兜住：这一步失败绝不能阻塞正式训练。
  _log "---- 先跑 A/B 离线评测（不阻塞后续；失败也继续）----"
  for arm in ab_A ab_B; do
    ck=$(ls -1 "runs/agilex_uncond_3cam_384_1e-4/$arm/checkpoints/weights"/*.pt 2>/dev/null | tail -1)
    if [[ -z "$ck" ]]; then _log "  $arm 无 weights，跳过"; continue; fi
    _log "  评测 $arm ($ck)"
    # timeout 20 分钟：该脚本从未在真实数据上跑过，卡住的话不能拖住正式训练
    timeout -k 30 1200 "${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}/bin/python" \
      scripts/offline_eval.py \
      --ckpt "$ck" --task agilex_uncond_3cam_384_1e-4 \
      --split val --num-samples 32 --mode joint --save-video \
      --out-dir "eval_offline/$arm" >> "$LOGFILE" 2>&1
    rc=$?
    (( rc == 124 || rc == 137 )) && _log "  ⚠️ $arm 离线评测超时(20min)被终止（不影响训练）"
    (( rc != 0 && rc != 124 && rc != 137 )) && _log "  ⚠️ $arm 离线评测失败 rc=$rc（不影响训练）"
  done
  if [[ -f eval_offline/ab_A/summary.json && -f eval_offline/ab_B/summary.json ]]; then
    timeout -k 15 300 "${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}/bin/python" \
      scripts/visualize_eval.py eval_offline/ab_A eval_offline/ab_B \
      >> "$LOGFILE" 2>&1 || _log "  ⚠️ 出图失败或超时"
    _log "  A/B 离线评测图: eval_offline/ab_A/figs/（含 06_run_compare 对比图）"
  fi
  # 等评测进程退出；加上限（10 分钟），否则若别人此时起了训练，守护会被永远卡住
  w=0
  while _train_running && (( w < 20 )); do sleep 30; w=$((w+1)); done
  (( w >= 20 )) && _log "  ⚠️ 等待 GPU 空闲超时，仍尝试启动（若显存不足会 OOM 并停止）"
  sleep 30
  _log "---- 离线评测阶段结束，开始正式训练 ----"
fi

  mkdir -p "$RUNDIR"
  local attempt=0
  [[ -f "$ATTEMPTFILE" ]] && attempt=$(cat "$ATTEMPTFILE")

  # ---- 2. 训练 + 崩溃续训循环 ----
  while (( attempt <= MAX_ATTEMPTS )); do
    _guard_disk || { rm -f "$PIDFILE"; exit 1; }

    # attempt 0 用 INIT_RESUME（**文件** = 只加载权重、step 从 0 开始）；
    # attempt>0 用最新 state **目录** = 恢复优化器/调度器/step，绝不能再用 INIT_RESUME，
    # 否则已训练的进度会被丢掉、从 step 0 重来。
    local resume_arg="$INIT_RESUME" wname="$RID"
    if (( attempt > 0 )); then
      resume_arg="$(_latest_state)"
      wname="${RID}_resume${attempt}"
      if [[ -z "$resume_arg" ]]; then
        _log "❌ 需要续训但找不到 state 目录（可能第一次保存前就崩了）。从头重启。"
        resume_arg="$INIT_RESUME"      # 退回初始化权重，而不是退成纯 Wan2.2
        wname="${RID}_restart${attempt}"
      else
        _log "从 $resume_arg 续训（该目录恢复完整训练状态：优化器/调度器/step）"
      fi
    fi

    echo "$attempt" > "$ATTEMPTFILE"
    # 训练在后台，主循环负责监控
    _launch "$resume_arg" "$attempt" "$wname" &
    local tpid=$!
    sleep 120   # 给模型加载留时间

    while kill -0 $tpid 2>/dev/null; do
      sleep "$CHECK_EVERY"
      echo $$ > "$PIDFILE"          # 自愈：PIDFILE 万一被误删也能恢复
      local step; step=$(_cur_step)
      _log "进度 step=${step:-?}/77690 磁盘 $(_disk_gb)G GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)%"
      if _has_nan; then
        _log "❌ 检测到 loss=NaN/Inf。**不自动重启**（重启不会解决数值问题），停止并等人工处理。"
        pkill -f "scripts/train.py" 2>/dev/null
        rm -f "$PIDFILE"; exit 1
      fi
      _guard_disk || { pkill -f "scripts/train.py" 2>/dev/null; rm -f "$PIDFILE"; exit 1; }
    done
    wait $tpid; local rc=$?

    if _done_ok; then
      _log "✅ 训练完成（step $(_cur_step)）"
      _guard_disk
      _log "最终 weights: $(ls -1 "$RUNDIR/checkpoints/weights"/*.pt 2>/dev/null | tail -1)"
      _log "下一步（人工）：离线评测"
      _log "  python scripts/offline_eval.py --ckpt <上面的 weights> --task $TASK \\"
      _log "    --num-samples 64 --mode joint --save-video --out-dir eval_offline/$RID"
      _log "  python scripts/visualize_eval.py eval_offline/$RID"
      rm -f "$PIDFILE" "$ATTEMPTFILE"
      exit 0
    fi

    if _has_oom; then
      _log "❌ OOM。自动续训不会解决（配置层已探明 bs=8 是上限，梯度检查点不可用）。停止。"
      _log "   如需继续：手工降 batch_size 或加 gradient_accumulation_steps 后重启"
      rm -f "$PIDFILE"; exit 1
    fi

    attempt=$((attempt+1))
    _log "训练异常中止 rc=$rc，将进行第 $attempt 次续训（上限 $MAX_ATTEMPTS）"
    sleep 90   # 等显存释放
  done

  _log "❌ 已达续训上限 $MAX_ATTEMPTS 次仍未完成，停止。请人工检查 $RUNDIR/train.log"
  rm -f "$PIDFILE"
  exit 1
}

case "${1:-status}" in
  start)
    p=$(_running_pid)
    [[ -n "$p" ]] && { echo "已在运行 (PID $p)"; exit 0; }
    setsid nohup bash "$0" _daemon >> "$LOGFILE" 2>&1 &
    sleep 2
    echo "正式训练守护已启动 (PID $(_running_pid))"
    echo "  run:    $RID  ->  $RUNDIR"
    echo "  初始化: ${INIT_RESUME:-null（纯 Wan2.2）}"
    echo "  规格:   5 epoch = 77,690 步 ≈ 42h · wandb.group=$WGROUP"
    if [[ "$SKIP_PRELUDE" == "1" ]]; then
      echo "  会等:   仅显存释放（SKIP_PRELUDE=1）"
    else
      echo "  会等:   ab_B 跑完 + 显存释放"
    fi
    echo "  崩溃:   自动从最新 state 续训，上限 $MAX_ATTEMPTS 次"
    echo "  磁盘:   每 $((CHECK_EVERY/60)) 分钟清理，只留 1 份 state + 3 份 weights"
    echo "  日志:   $LOGFILE"
    echo "  状态:   RID=$RID bash scripts/run_final.sh status"
    ;;
  _daemon)
    echo $$ > "$PIDFILE"
    trap 'rm -f "$PIDFILE"; exit 0' TERM INT
    _daemon_main
    ;;
  status)
    p=$(_running_pid)
    if [[ -n "$p" ]]; then echo "守护运行中 (PID $p)"; else echo "守护未运行"; fi
    _train_running && echo "训练进程: 在跑" || echo "训练进程: 未运行"
    [[ -f "$RUNDIR/train.log" ]] && echo "当前进度: step $(_cur_step)/77690"
    echo "磁盘可用: $(_disk_gb)G"
    echo "--- 最近记录 ---"
    tail -15 "$LOGFILE" 2>/dev/null || echo "(无日志)"
    ;;
  stop)
    p=$(_running_pid)
    if [[ -z "$p" ]]; then echo "守护未运行"; rm -f "$PIDFILE"; exit 0; fi
    # 注意：bash 的 trap 要等当前前台命令（这里可能是 sleep 180）结束才执行，
    # 所以 TERM 之后可能要等最多 3 分钟。等不到就升级到 -9，并**如实报告**结果
    # （早先的版本无条件打印"已停止"，导致以为停了其实还在跑）。
    kill -TERM "$p" 2>/dev/null
    for _ in $(seq 1 10); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    if kill -0 "$p" 2>/dev/null; then
      echo "TERM 未即时生效（守护可能卡在 sleep），升级为 -9 ..."
      kill -9 "$p" 2>/dev/null
      for _ in $(seq 1 10); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    fi
    if kill -0 "$p" 2>/dev/null; then
      echo "❌ PID $p 仍存活，请手工处理"; exit 1
    fi
    echo "守护已停止 (PID $p)；训练进程未受影响"
    rm -f "$PIDFILE"
    ;;
  abort)
    bash "$0" stop
    pkill -f "scripts/train.py" 2>/dev/null && echo "训练进程已终止" || echo "无训练进程"
    ;;
  *) echo "用法: bash scripts/run_final.sh {start|status|stop|abort}"; exit 1 ;;
esac
