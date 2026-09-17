#!/usr/bin/env bash
# 定时把 /home 的工作目录增量同步到 /mnt/data (OSS) 做持久化。
#
# 设计取舍：用脱离终端的后台循环（setsid nohup），而不是 Claude 的定时任务 ——
# 纯机械的 rsync 不需要模型参与，这样零上下文开销，也不依赖会话是否在线。
#
# 同步内容由 scripts/sync_to_oss.sh 决定，默认**排除 ZeRO state**（80GB/份，
# 只用于续训不用于持久化）和视频软链接目录。
#
# 用法:
#   bash scripts/autosync_daemon.sh start [间隔秒数]   # 默认 1800 秒 = 30 分钟
#   bash scripts/autosync_daemon.sh status
#   bash scripts/autosync_daemon.sh stop
#   bash scripts/autosync_daemon.sh once               # 立刻同步一次（前台）

set -uo pipefail
cd /home/gaomeng/FastWAM

PIDFILE=".autosync.pid"
LOGFILE=".autosync.log"
LOCKFILE=".autosync.lock"

_loop() {
  local interval="$1"
  while true; do
    # 每轮重新确认 PID 文件（自愈）。
    # 曾踩过的坑：stop 杀掉旧进程后立刻返回，start 随即写入新 PID 文件，
    # 而旧进程的 trap 才异步触发 rm -f，把新文件删掉 —— 导致 status/stop 失效
    # 而守护其实还活着。这里每轮重写一次，即使被误删也能自动恢复。
    echo $$ > "$PIDFILE"

    # 防止上一轮没跑完就叠加（OSS 慢，同步可能超过一个周期）
    if mkdir "$LOCKFILE" 2>/dev/null; then
      {
        echo "===== $(date '+%F %T') 开始同步 ====="
        bash scripts/sync_to_oss.sh --changing-only 2>&1 \
          | grep -E "^----|Number of regular files transferred|Total transferred file size|rsync error|Operation not supported"
        echo "===== $(date '+%F %T') 同步结束 ====="
        echo ""
      } >> "$LOGFILE" 2>&1
      rmdir "$LOCKFILE" 2>/dev/null
    else
      echo "$(date '+%F %T') 上一轮仍在进行，跳过本轮" >> "$LOGFILE"
    fi
    sleep "$interval"
  done
}

# 找出真正在跑的守护进程 PID（不依赖 PID 文件）
_running_pid() {
  pgrep -f "autosync_daemon.sh _run" | head -1
}

case "${1:-status}" in
  start)
    INTERVAL="${2:-1800}"
    EXIST=$(_running_pid)
    if [[ -n "$EXIST" ]]; then
      echo "已在运行 (PID $EXIST)，如需改间隔请先 stop"
      exit 0
    fi
    rmdir "$LOCKFILE" 2>/dev/null
    setsid nohup bash "$0" _run "$INTERVAL" >> "$LOGFILE" 2>&1 &
    sleep 2
    NEW=$(_running_pid)
    echo "已启动自动同步守护，间隔 ${INTERVAL}s ($(( INTERVAL / 60 )) 分钟)，PID ${NEW:-?}"
    echo "  日志: $LOGFILE"
    echo "  停止: bash scripts/autosync_daemon.sh stop"
    ;;
  _run)   # 内部入口，由 start 通过 setsid 调用
    echo $$ > "$PIDFILE"
    trap 'rm -f "$PIDFILE"; rmdir "$LOCKFILE" 2>/dev/null; exit 0' TERM INT
    _loop "${2:-1800}"
    ;;
  stop)
    PID=$(_running_pid)
    if [[ -z "$PID" ]]; then
      echo "未在运行"
      rm -f "$PIDFILE"; rmdir "$LOCKFILE" 2>/dev/null
      exit 0
    fi
    kill -TERM "$PID" 2>/dev/null
    # 等它真正退出再返回 —— 否则旧进程的 trap 会异步 rm 掉新进程的 PID 文件
    for _ in $(seq 1 20); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null; sleep 1
      echo "已强制停止 (PID $PID)"
    else
      echo "已停止 (PID $PID)"
    fi
    rm -f "$PIDFILE"; rmdir "$LOCKFILE" 2>/dev/null
    ;;
  status)
    PID=$(_running_pid)
    if [[ -n "$PID" ]]; then
      echo "运行中 (PID $PID)"
      [[ -f "$PIDFILE" ]] || echo "  注意: PID 文件缺失，但进程在跑（下一轮会自动重建）"
      echo "--- 最近同步记录 ---"
      grep -E "开始同步|同步结束|rsync error|跳过本轮" "$LOGFILE" 2>/dev/null | tail -6
    else
      echo "未运行"
    fi
    ;;
  once)
    bash scripts/sync_to_oss.sh
    ;;
  *)
    echo "用法: bash scripts/autosync_daemon.sh {start [秒]|stop|status|once}"; exit 1
    ;;
esac
