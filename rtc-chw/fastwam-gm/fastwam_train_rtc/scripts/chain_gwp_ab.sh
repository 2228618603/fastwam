#!/usr/bin/env bash
# GWP-0.5 预训练权重迁移的 A/B 短跑串联守护。
#
# 方案与判读规则见 GWP05_TRANSFER_PLAN.md。三臂设计：
#
#   A  (已有)  纯 Wan2.2 + ActionDiT 插值初始化    runs/$TASK/ab_A
#   C1 (本脚本) GWP-0.5 **只迁视觉专家**（825 张量） runs/$TASK/ab_C1
#   C2 (本脚本) GWP-0.5 视觉 + 动作专家（1645 张量） runs/$TASK/ab_C2
#
# C1/C2 与 A 的**唯一**差别是 resume 指向的初始化权重：其余超参、数据、归一化统计、
# 验证集、eval 间隔全部逐字相同（见下方命令），所以这是一个单变量受控对比。
#
# 两个 5000 步短跑各约 2 h 43 m（8xA800 实测，1.95 s/step），bs=8 峰值 76~78 GB /
# 81920 MiB 塞不进第二个 run，**必须串行**。
#
# ⚠️ 本机 8 卡是多人共享的。脚本**只等待、绝不抢占**：要求 8 张卡的显存占用全部低于
#    GPU_FREE_MIB，且连续 GPU_FREE_POLLS 次轮询都满足，才认为空闲。
#
# 用法:
#   bash scripts/chain_gwp_ab.sh start
#   bash scripts/chain_gwp_ab.sh status
#   bash scripts/chain_gwp_ab.sh stop

set -uo pipefail
cd /home/gaomeng/FastWAM

TASK=agilex_uncond_3cam_384_1e-4
STATS=./runs/_shared/dataset_stats.json
CKPT_C1=./checkpoints/gwp05_video_only.pt
CKPT_C2=./checkpoints/gwp05_full.pt
AB_STEPS=5000
BS=8
NW=8

# GPU 空闲判定
GPU_FREE_MIB=${GPU_FREE_MIB:-2000}     # 每卡已用显存低于该值算空闲
GPU_FREE_POLLS=${GPU_FREE_POLLS:-3}    # 连续满足次数（避免抢进别人两个 step 之间的空隙）
GPU_POLL_SEC=${GPU_POLL_SEC:-60}

A_DIR="runs/$TASK/ab_A"
C1_DIR="runs/$TASK/ab_C1"
C2_DIR="runs/$TASK/ab_C2"
PIDFILE=".chain_gwp_ab.pid"
LOGFILE=".chain_gwp_ab.log"

_log() { echo "$(date '+%F %T') $*" >> "$LOGFILE"; }

_train_running() { pgrep -f "scripts/train.py" > /dev/null 2>&1; }
_running_pid()   { pgrep -f "chain_gwp_ab.sh _daemon" | head -1; }

# 全部 8 卡是否空闲（单次快照）
_gpus_idle_once() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z "$used" ]] && return 1
  local n=0
  while read -r m; do
    [[ -z "$m" ]] && continue
    n=$((n + 1))
    (( m >= GPU_FREE_MIB )) && return 1
  done <<< "$used"
  (( n == 8 ))
}

# 等到 8 卡连续空闲，只等不抢
_wait_gpus() {
  local hits=0
  local reported=0
  while true; do
    if _gpus_idle_once; then
      hits=$((hits + 1))
      _log "GPU 空闲判定 $hits/$GPU_FREE_POLLS"
      if (( hits >= GPU_FREE_POLLS )); then
        _log "✅ 8 卡已连续空闲，开始占用"
        return 0
      fi
    else
      if (( hits > 0 )); then
        _log "GPU 又被占用，空闲计数归零"
      elif (( reported % 30 == 0 )); then
        local snap
        snap=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')
        _log "等待 GPU 释放中… [$snap]"
      fi
      hits=0
      reported=$((reported + 1))
    fi
    sleep "$GPU_POLL_SEC"
  done
}

# 跑一个臂：$1=RUN_ID  $2=输出目录  $3=resume 权重  $4=wandb name
_run_arm() {
  local run_id="$1" dir="$2" ckpt="$3" wname="$4"

  if [[ ! -f "$ckpt" ]]; then
    _log "❌ 权重不存在: $ckpt"
    return 1
  fi

  _wait_gpus
  _log "启动 $run_id （resume=$ckpt）"
  mkdir -p "$dir"
  source env.sh > /dev/null 2>&1

  # 除 resume 外，全部参数与 ab_A / ab_B 逐字相同
  RUN_ID="$run_id" bash scripts/train_zero1.sh 8 task=$TASK \
    max_steps=$AB_STEPS batch_size=$BS num_workers=$NW \
    eval_every=250 save_every=999999 \
    data.train.pretrained_norm_stats=$STATS \
    data.val.pretrained_norm_stats=$STATS \
    resume=$ckpt \
    wandb.enabled=true wandb.project=fastwam-agilex \
    wandb.name="$wname" wandb.group=ab_warmstart \
    > "$dir/train.log" 2>&1
  local rc=$?
  _log "$run_id 结束 exit=$rc"

  if sed 's/\x1b\[[0-9;]*m//g' "$dir/train.log" 2>/dev/null | grep -qE "OutOfMemoryError|CUDA out of memory"; then
    _log "❌ $run_id 日志里发现 OOM"
    return 1
  fi
  if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/train.log" 2>/dev/null | grep -qE "max_steps reached|training finished"; then
    _log "❌ $run_id 没有完成标记，疑似崩溃。请人工检查 $dir/train.log"
    return 1
  fi

  # 关键健全性检查：确认权重真的被加载了（strict=False 会静默吞掉错误）
  if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/train.log" | grep -q "Loading weight checkpoint only: $ckpt"; then
    _log "⚠️ $run_id 日志里没有找到 'Loading weight checkpoint only: $ckpt' —— 权重可能没被加载！"
  fi

  while _train_running; do sleep 20; done
  sleep 30

  if [[ -d "$dir/checkpoints/state" ]]; then
    local sz
    sz=$(du -sh "$dir/checkpoints/state" 2>/dev/null | cut -f1)
    rm -rf "$dir/checkpoints/state"
    _log "已删除 $run_id 的 ZeRO state ($sz)"
  fi
  _log "磁盘可用: $(df -h / | tail -1 | awk '{print $4}')"
  return 0
}

_compare() {
  {
    echo ""
    echo "################ A / C1 / C2 对比 ################"
    echo "A  = 纯 Wan2.2 初始化 (已有 baseline)"
    echo "C1 = GWP-0.5 只迁视觉专家"
    echo "C2 = GWP-0.5 视觉 + 动作专家"
    echo ""
    "${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}/bin/python" scripts/parse_train_log.py \
      A="$A_DIR/train.log" C1="$C1_DIR/train.log" C2="$C2_DIR/train.log" --compare \
      --csv "runs/$TASK/gwp_ab_compare.csv" 2>&1
    echo ""
    echo "判读规则 (GWP05_TRANSFER_PLAN.md §5.3):"
    echo "  C 在 loss_action 与 action_l2 上均优 >=10%  -> 采纳，跑满 5 epoch"
    echo "  差距 <5% 或更差                            -> 保持纯 Wan2.2 初始化"
    echo "  C1 赢而 C2 输                              -> 掩码语义指纹，转入方案 B 评估"
  } >> "$LOGFILE" 2>&1
}

_run() {
  _log "===== GWP-0.5 迁移 A/B 守护启动 (AB_STEPS=$AB_STEPS bs=$BS nw=$NW) ====="
  _log "C1 权重: $CKPT_C1"
  _log "C2 权重: $CKPT_C2"

  if _run_arm ab_C1 "$C1_DIR" "$CKPT_C1" ab_C1_gwp05_video_only; then
    _log "✅ C1 完成"
  else
    _log "❌ C1 失败，仍继续尝试 C2（两臂互相独立）"
  fi

  if _run_arm ab_C2 "$C2_DIR" "$CKPT_C2" ab_C2_gwp05_full; then
    _log "✅ C2 完成"
  else
    _log "❌ C2 失败"
  fi

  _compare
  _log "===== 全部结束，对比见本文件与 runs/$TASK/gwp_ab_compare.csv ====="
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
    echo "守护已启动：等 8 卡空闲 -> 跑 C1 -> 跑 C2 -> 出对比"
    echo "  日志: $LOGFILE"
    echo "  状态: bash scripts/chain_gwp_ab.sh status"
    ;;
  _daemon)
    echo $$ > "$PIDFILE"
    trap 'rm -f "$PIDFILE"; exit 0' TERM INT
    _run
    ;;
  status)
    PID=$(_running_pid)
    if [[ -n "$PID" ]]; then
      echo "守护运行中 (PID $PID)"
    else
      echo "守护未运行"
    fi
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    echo "--- 最近记录 ---"
    tail -15 "$LOGFILE" 2>/dev/null || echo "(无日志)"
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
    echo "用法: bash scripts/chain_gwp_ab.sh {start|status|stop}"; exit 1 ;;
esac
