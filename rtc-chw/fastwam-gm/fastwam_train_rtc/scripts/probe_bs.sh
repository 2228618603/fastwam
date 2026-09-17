#!/usr/bin/env bash
# Phase 5c/5d 探测：找出不 OOM 的最大 batch_size 与合适的 num_workers。
#
# 关键技巧：trainer.py:796-802 在 global_step >= max_steps 时**无条件**保存 checkpoint，
# 而一次保存 = weights 12GB + ZeRO state 80GB = 92GB。所以探测时把 max_steps 设得很大，
# 跑够步数后直接 kill 进程 —— 完全避免 92GB 的写入。
#
# 用法:
#   bash scripts/probe_bs.sh 8 12 16          # 探 batch_size（num_workers 固定 8）
#   NW=6 bash scripts/probe_bs.sh 8           # 指定 num_workers
#   STEPS=25 bash scripts/probe_bs.sh 8

set -uo pipefail
cd /home/gaomeng/FastWAM
source env.sh > /dev/null

TASK=agilex_uncond_3cam_384_1e-4
LAUNCHER="${LAUNCHER:-scripts/train_zero1.sh}"
STATS=./runs/_shared/dataset_stats.json
NW="${NW:-8}"
STEPS="${STEPS:-22}"          # 跑到这一步就 kill
TIMEOUT_S="${TIMEOUT_S:-900}" # 单次探测最长等待
EVAL_EVERY="${EVAL_EVERY:-0}" # >0 则开启 eval，用于测 eval 阶段的真实显存峰值

BS_LIST=("$@")
[[ ${#BS_LIST[@]} -eq 0 ]] && BS_LIST=(8)

printf '%-6s %-6s %-10s %-12s %-12s %-10s\n' BS NW 峰值显存 步时间 loss 结果
printf '%s\n' "--------------------------------------------------------------------------"

for BS in "${BS_LIST[@]}"; do
  RID="probe_bs${BS}_nw${NW}${EVAL_EVERY:+_ev$EVAL_EVERY}_$(basename $LAUNCHER .sh)"
  RUNDIR="runs/$TASK/$RID"
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
  LOG="$RUNDIR/train.log"
  MEMF="$RUNDIR/mem.txt"

  # 显存采样器：每 2 秒记录 8 卡中的最大已用量
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | sort -n | tail -1 >> "$MEMF"
      sleep 2
    done ) & SAMPLER=$!

  # max_steps 设很大 -> 永不触发末尾的强制保存
  RUN_ID="$RID" bash "$LAUNCHER" 8 task=$TASK \
    max_steps=999999 batch_size="$BS" num_workers="$NW" \
    eval_every="$EVAL_EVERY" save_every=999999 log_every=1 \
    data.train.pretrained_norm_stats=$STATS \
    data.val.pretrained_norm_stats=$STATS \
    wandb.enabled=false > "$LOG" 2>&1 & TRAIN=$!

  # 等到跑够 STEPS 步，或 OOM，或超时
  RESULT="timeout"; ELAPSED=0
  while (( ELAPSED < TIMEOUT_S )); do
    if ! kill -0 $TRAIN 2>/dev/null; then RESULT="exited"; break; fi
    if grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG" 2>/dev/null; then RESULT="OOM"; break; fi
    if sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -qE "step=${STEPS}/"; then
      RESULT="ok"; break
    fi
    sleep 5; ELAPSED=$((ELAPSED+5))
  done

  kill $SAMPLER 2>/dev/null
  # 先温和终止整个进程组，再兜底 pkill accelerate 的子进程
  kill -TERM $TRAIN 2>/dev/null
  sleep 8
  pkill -f "train.py.*batch_size=$BS" 2>/dev/null
  pkill -9 -f "scripts/train.py" 2>/dev/null
  sleep 12   # 等显存真正释放

  PEAK=$(sort -n "$MEMF" 2>/dev/null | tail -1)
  # 用第 10..STEPS 步的时间戳算稳态步时间（跳过前几步的预热）
  STEPTIME=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null \
    | grep -oE "^[0-9/]+ \[[0-9:]+\].*step=[0-9]+/" \
    | awk 'match($0,/\[([0-9]+):([0-9]+):([0-9]+)\]/,t) && match($0,/step=([0-9]+)\//,s){
             sec=t[1]*3600+t[2]*60+t[3];
             if(s[1]==10){t0=sec;s0=s[1]} if(s[1]>s0&&t0){print (sec-t0)/(s[1]-s0)} }' \
    | tail -1)
  LOSS=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -oE "loss=[0-9.]+" | tail -1)

  printf '%-6s %-6s %-10s %-12s %-12s %-10s\n' \
    "$BS" "$NW" "${PEAK:-?} MiB" "${STEPTIME:-?} s" "${LOSS:-?}" "$RESULT"
done

echo ""
echo "显存上限 81920 MiB；建议选峰值 < 70000 MiB 的最大 BS（给 eval 的 infer+VAE decode 留余量）"
echo "探测目录可删: rm -rf runs/$TASK/probe_bs*"
