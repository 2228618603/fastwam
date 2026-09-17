#!/bin/bash
# ============================================================================
#  FastWAM 真机部署 — 控制环客户端
# ============================================================================
#  用法: bash scripts/start_client.sh [任意 fastwam_client.py 的参数]
#
#  ⚠️ 本脚本只提供**你没在命令行给出**的参数作为默认值。任何你显式传的参数
#     都会覆盖默认值,且**不会**残留一个冲突的默认(见下面 §为什么这么写)。
#     所以你可以放心用命令行表达一切,脚本里的值不用改。
#
#  常用:
#    # 看一眼(不动手臂)
#    bash scripts/start_client.sh
#
#    # RTC 异步,跑 10 分钟
#    bash scripts/start_client.sh --mode async --rtc \
#        --confirm-safety --i-am-watching --max-duration 600
#
#    # 一直跑(直到 Ctrl-C);--max-steps 0 = 不限步数
#    bash scripts/start_client.sh --mode async --rtc \
#        --confirm-safety --i-am-watching --max-steps 0
#
#    # 非 RTC 对照(server 那边的 RTC= 也要改成 0 并重启)
#    bash scripts/start_client.sh --mode sync --confirm-safety --max-steps 300
#
#    # 全部可选参数
#    bash scripts/start_client.sh --help
#
#  运行前提(三个进程,按顺序起):
#    1) 机器人 PC: python scripts/deploy/start_robot_arm_service.py --port 9900  (openpi 仓库)
#    2) GPU 机  : bash scripts/start_server.sh
#    3) 本脚本
#
#  ── 为什么这么写 ────────────────────────────────────────────────────────────
#  之前这个脚本是「把脚本里的变量拼成参数,再把你的参数追加在后面」。argparse 里
#  后者覆盖前者,所以 --mode 之类**看起来**是好的。但对**互相冲突**的参数就不行了:
#  脚本默认写死 --max-steps 60,你在命令行加 --max-duration 600,两个都会传进去,
#  而 `_budget_done()` 是「任一到达就停」—— 于是 2 秒后就停,不是 10 分钟。
#  现在改成:只在你**没提**某个参数时才补默认值。
# ============================================================================
set -e
cd "$(dirname "$0")/.."
source env.sh

# ---- 默认值(仅在命令行未提供时生效)----------------------------------------
DEF_SERVER="tcp://127.0.0.1:8900"     # fastwam_server.py
DEF_ENDPOINT="tcp://127.0.0.1:9900"   # openpi RobotArmService
DEF_MODE="dry"                        # dry(不动手臂) | sync | async
DEF_MAX_STEPS="60"                    # 0 = 不限;与 --max-duration 是「先到先停」
DEF_RECORD_DIR="runs/deploy/$(date +%m%d_%H%M)"

# 只在用户没给某个 flag 时才补它的默认值
has() { for a in "$@"; do [ "$a" = "$WANT" ] && return 0; done; return 1; }
add_default() {           # add_default <flag> <value>
    WANT="$1"
    has "${USER_ARGS[@]}" || DEFAULTS+=("$1" "$2")
}

USER_ARGS=("$@")
DEFAULTS=()
add_default --server   "$DEF_SERVER"
add_default --endpoint "$DEF_ENDPOINT"
add_default --mode     "$DEF_MODE"
add_default --record-dir "$DEF_RECORD_DIR"
# --max-steps 只在你**既没给它、也没给 --max-duration / --run-forever** 时才补 ——
# 否则会和你的界限冲突(见上面 §为什么这么写:两个界限是「先到先停」,
# 补上的 60 步会让 --max-duration 600 或 --run-forever 在 2 秒后就停)。
WANT=--max-duration;  has "${USER_ARGS[@]}" && HAS_LIMIT=1 || HAS_LIMIT=0
WANT=--run-forever;   has "${USER_ARGS[@]}" && HAS_LIMIT=1
if [ "$HAS_LIMIT" = "0" ]; then
    add_default --max-steps "$DEF_MAX_STEPS"
fi

# ---- 会动手臂时提醒(不自动补安全 flag)--------------------------------------
# 刻意**不**自动加 --confirm-safety / --i-am-watching:它们的意义就是「人已经确认过
# 现场」,脚本替你加等于把这道闸门作废。缺了 fastwam_client.py 会明确拒绝并告诉你缺哪个。
MODE_EFF="$DEF_MODE"
prev=""
for a in "$@"; do [ "$prev" = "--mode" ] && MODE_EFF="$a"; prev="$a"; done
if [ "$MODE_EFF" != "dry" ]; then
    echo "========================================="
    echo " ⚠️  即将移动机械臂 (--mode $MODE_EFF)"
    echo "   - 急停按钮在手边?"
    echo "   - 工作区清空、包络内无人?"
    echo " preflight 之后还会等你输入 yes(--confirm)"
    echo "========================================="
fi

echo ">>> python scripts/fastwam_client.py ${DEFAULTS[*]} $*"
echo "    (前半是脚本补的默认值,后半是你在命令行给的)"
exec python scripts/fastwam_client.py "${DEFAULTS[@]}" "$@"
