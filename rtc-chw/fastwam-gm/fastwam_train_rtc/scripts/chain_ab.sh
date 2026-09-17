#!/usr/bin/env bash
# 等 A 跑完后自动接着跑 B（A/B 热启动对比），完全脱离 Claude 会话。
#
# 为什么需要这个脚本：A 是用后台任务启动的，它退出时只会给 Claude 发一个通知。
# 如果那一刻会话不活跃，B 就不会被启动。用一个 setsid detach 的守护脚本来串联，
# 就不依赖会话是否在线了。
#
# 流程：
#   1. 轮询 A 的日志，等 "max_steps reached" / "training finished"
#   2. 校验 A 成功（无 OOM、进程正常退出）
#   3. 删掉 A 的 ZeRO state（80GB，A/B 是一次性对比，不需要续训）
#   4. 启动 B（与 A 完全相同的配置 + resume=RoboTwin 权重 + 同一份 dataset_stats）
#   5. B 跑完同样删 state，并输出对比
#
# 用法:
#   bash scripts/chain_ab.sh start
#   bash scripts/chain_ab.sh status
#   bash scripts/chain_ab.sh stop

set -uo pipefail
cd /home/gaomeng/FastWAM

TASK=agilex_uncond_3cam_384_1e-4
STATS=./runs/_shared/dataset_stats.json
ROBOTWIN=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
AB_STEPS=5000
BS=8
NW=8

A_DIR="runs/$TASK/ab_A"
B_DIR="runs/$TASK/ab_B"
PIDFILE=".chain_ab.pid"
LOGFILE=".chain_ab.log"

_log() { echo "$(date '+%F %T') $*" >> "$LOGFILE"; }

_a_done() {
  [[ -f "$A_DIR/train.log" ]] || return 1
  sed 's/\x1b\[[0-9;]*m//g' "$A_DIR/train.log" 2>/dev/null \
    | grep -qE "max_steps reached|training finished"
}

_a_broken() {
  [[ -f "$A_DIR/train.log" ]] || return 1
  sed 's/\x1b\[[0-9;]*m//g' "$A_DIR/train.log" 2>/dev/null \
    | grep -qE "OutOfMemoryError|CUDA out of memory"
}

_train_running() {
  pgrep -f "scripts/train.py" > /dev/null 2>&1
}

# 不依赖 PID 文件判断守护是否在跑（PID 文件可能被竞态误删）
_running_pid() {
  pgrep -f "chain_ab.sh _daemon" | head -1
}

_run() {
  _log "守护启动，等待 A 完成 (AB_STEPS=$AB_STEPS bs=$BS nw=$NW)"

  # ---- 1. 等 A ----
  while true; do
    if _a_broken; then
      _log "❌ A 日志里发现 OOM，中止链式执行，不启动 B"
      rm -f "$PIDFILE"; exit 1
    fi
    if _a_done; then
      _log "✅ A 已完成"
      break
    fi
    if ! _train_running && [[ -f "$A_DIR/train.log" ]]; then
      sleep 30
      if ! _train_running && ! _a_done; then
        _log "❌ A 的训练进程已退出但没有完成标记，疑似崩溃。不启动 B。"
        _log "   请人工检查 $A_DIR/train.log"
        rm -f "$PIDFILE"; exit 1
      fi
    fi
    sleep 60
  done

  while _train_running; do sleep 20; done
  sleep 30
  _log "A 的进程已全部退出，显存应已释放"

  # ---- 2. 删 A 的 state（80GB）----
  if [[ -d "$A_DIR/checkpoints/state" ]]; then
    SZ=$(du -sh "$A_DIR/checkpoints/state" 2>/dev/null | cut -f1)
    rm -rf "$A_DIR/checkpoints/state"
    _log "已删除 A 的 ZeRO state ($SZ)，释放磁盘"
  fi
  _log "磁盘可用: $(df -h / | tail -1 | awk '{print $4}')"

  # ---- 3. 跑 B ----
  _log "启动 B（热启动 RoboTwin 权重）"
  mkdir -p "$B_DIR"
  source env.sh > /dev/null 2>&1

  RUN_ID=ab_B bash scripts/train_zero1.sh 8 task=$TASK \
    max_steps=$AB_STEPS batch_size=$BS num_workers=$NW \
    eval_every=250 save_every=999999 \
    data.train.pretrained_norm_stats=$STATS \
    data.val.pretrained_norm_stats=$STATS \
    resume=$ROBOTWIN \
    wandb.enabled=true wandb.project=fastwam-agilex \
    wandb.name=ab_B_robotwin_warmstart wandb.group=ab_warmstart \
    > "$B_DIR/train.log" 2>&1
  RC=$?
  _log "B 结束 exit=$RC"

  # ---- 4. 删 B 的 state ----
  if [[ -d "$B_DIR/checkpoints/state" ]]; then
    SZ=$(du -sh "$B_DIR/checkpoints/state" 2>/dev/null | cut -f1)
    rm -rf "$B_DIR/checkpoints/state"
    _log "已删除 B 的 ZeRO state ($SZ)"
  fi

  # ---- 5. 出对比 ----
  {
    echo ""
    echo "################ A/B 对比结果 ################"
    "${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}/bin/python" scripts/parse_train_log.py \
      A="$A_DIR/train.log" B="$B_DIR/train.log" --compare \
      --csv "runs/$TASK/ab_compare.csv" 2>&1
  } >> "$LOGFILE" 2>&1

  _log "✅ A/B 全部完成，对比已写入本文件与 runs/$TASK/ab_compare.csv"
  rm -f "$PIDFILE"
}

case "${1:-status}" in
  start)
    EXIST=$(_running_pid)
    if [[ -n "$EXIST" ]]; then
      echo "已在运行 (PID $EXIST)"; exit 0
    fi
    setsid nohup bash "$0" _daemon >> "$LOGFILE" 2>&1 &
    sleep 1
    echo "链式守护已启动：A 完成后自动跑 B"
    echo "  日志: $LOGFILE"
    echo "  状态: bash scripts/chain_ab.sh status"
    ;;
  _daemon)
    echo $$ > "$PIDFILE"
    trap 'rm -f "$PIDFILE"; exit 0' TERM INT
    _run
    ;;
  status)
    PID=$(_running_pid)
    if [[ -n "$PID" ]]; then
      echo "链式守护运行中 (PID $PID)"
    else
      echo "链式守护未运行"
    fi
    echo "--- 最近记录 ---"
    tail -12 "$LOGFILE" 2>/dev/null || echo "(无日志)"
    ;;
  stop)
    PID=$(_running_pid)
    if [[ -n "$PID" ]]; then
      kill -TERM "$PID" 2>/dev/null
      for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
      echo "已停止 (PID $PID)"
      rm -f "$PIDFILE"
    else
      echo "未在运行"
    fi
    ;;
  *)
    echo "用法: bash scripts/chain_ab.sh {start|status|stop}"; exit 1 ;;
esac
