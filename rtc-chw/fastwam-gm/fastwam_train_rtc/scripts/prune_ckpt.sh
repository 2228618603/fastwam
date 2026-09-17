#!/usr/bin/env bash
# 清理训练 run 里的旧 checkpoint，只保留最近若干份，防止磁盘打满。
#
# 背景（Phase 5b 实测）：
#   weights/step_NNNNNN.pt        = 12 GB / 次
#   state/step_NNNNNN/  (ZeRO)    = 80 GB / 次
#   单次保存合计                  = 92 GB
# 而这台机器磁盘是**多人共享**的（/root/litianyu 已占 1.9 TB，其他用户还在增长），
# 实测可用空间会在几小时内被别人吃掉几百 GB。所以长训练必须主动清理。
#
# 策略：state 只留最近 KEEP_STATE 份（默认 1，续训只需要最新的那份），
#       weights 留最近 KEEP_WEIGHTS 份（默认 3，用于回溯/对比）。
#
# 用法:
#   bash scripts/prune_ckpt.sh runs/<task>/<run_id>
#   KEEP_STATE=2 KEEP_WEIGHTS=5 bash scripts/prune_ckpt.sh runs/<task>/<run_id>
#   DRY=1 bash scripts/prune_ckpt.sh runs/<task>/<run_id>     # 只看会删什么

set -uo pipefail

RUNDIR="${1:?用法: bash scripts/prune_ckpt.sh <run_dir> }"
KEEP_STATE="${KEEP_STATE:-1}"
KEEP_WEIGHTS="${KEEP_WEIGHTS:-3}"
DRY="${DRY:-0}"

CK="$RUNDIR/checkpoints"
[[ -d "$CK" ]] || { echo "没有 checkpoints 目录: $CK"; exit 0; }

echo "=== 清理前 ==="
df -h / | tail -1 | awk '{print "  磁盘可用: "$4" (已用 "$5")"}'
du -sh "$CK" 2>/dev/null | awk '{print "  checkpoints: "$1}'

# --- state：按 step 排序，删掉除最后 KEEP_STATE 份以外的 ---
if [[ -d "$CK/state" ]]; then
  mapfile -t STATES < <(find "$CK/state" -maxdepth 1 -mindepth 1 -type d -name 'step_*' | sort)
  TOTAL=${#STATES[@]}
  DEL=$(( TOTAL - KEEP_STATE ))
  echo ""
  echo "state: 共 $TOTAL 份，保留最近 $KEEP_STATE 份，待删 $(( DEL > 0 ? DEL : 0 )) 份"
  if (( DEL > 0 )); then
    for ((i=0; i<DEL; i++)); do
      SZ=$(du -sh "${STATES[$i]}" 2>/dev/null | cut -f1)
      if [[ "$DRY" == "1" ]]; then
        echo "  [DRY] 会删 ${STATES[$i]} ($SZ)"
      else
        echo "  删除 ${STATES[$i]} ($SZ)"
        rm -rf "${STATES[$i]}"
      fi
    done
  fi
fi

# --- weights：同理 ---
if [[ -d "$CK/weights" ]]; then
  mapfile -t WS < <(find "$CK/weights" -maxdepth 1 -type f -name 'step_*.pt' | sort)
  TOTAL=${#WS[@]}
  DEL=$(( TOTAL - KEEP_WEIGHTS ))
  echo ""
  echo "weights: 共 $TOTAL 份，保留最近 $KEEP_WEIGHTS 份，待删 $(( DEL > 0 ? DEL : 0 )) 份"
  if (( DEL > 0 )); then
    for ((i=0; i<DEL; i++)); do
      SZ=$(du -sh "${WS[$i]}" 2>/dev/null | cut -f1)
      if [[ "$DRY" == "1" ]]; then
        echo "  [DRY] 会删 ${WS[$i]} ($SZ)"
      else
        echo "  删除 ${WS[$i]} ($SZ)"
        rm -f "${WS[$i]}"
      fi
    done
  fi
fi

echo ""
echo "=== 清理后 ==="
df -h / | tail -1 | awk '{print "  磁盘可用: "$4" (已用 "$5")"}'
du -sh "$CK" 2>/dev/null | awk '{print "  checkpoints: "$1}'
echo ""
echo "剩余 state（可用于 resume=<目录>）:"
find "$CK/state" -maxdepth 1 -mindepth 1 -type d -name 'step_*' 2>/dev/null | sort | sed 's/^/  /'
