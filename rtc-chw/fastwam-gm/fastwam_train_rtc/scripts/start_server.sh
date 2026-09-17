#!/bin/bash
# ============================================================================
#  FastWAM 真机部署 — GPU 侧推理服务
# ============================================================================
#  用法: bash scripts/start_server.sh
#  只改下面几个变量,其余不用动。
#
#  为什么用这个而不是 deploy_real.py:推理独占一个专属线程,所以
#  compile_mode=reduce-overhead(CUDA Graph Trees)可用 —— RTC 异步执行下实测比
#  default 快 2.4 倍。单进程的 deploy_real.py 做不到(预热池与推理池是两个线程)。
# ============================================================================

# ---- 基本配置 --------------------------------------------------------------
PORT=8900
DEVICE="cuda:0"                                # 显存需求约 13.4 GB

# ---- RTC 开关(必须与 scripts/start_client.sh 的 RTC= 配套)------------------
# RTC=1 -> agilex_real_rtc.yaml  task=agilex_rtc_...   step_008000.pt(带前缀条件训练)
# RTC=0 -> agilex_real.yaml      task=agilex_final_... step_032000.pt(基线权重)
#
# 配错了不会静默出错:client 会用 server 上报的 rtc_trained 交叉校验并**直接拒绝**
# (拿基线权重跑 RTC 前缀,离线实测关节误差 1.52->2.02 度、MSE 近翻倍)。
RTC=1

if [ "$RTC" = "1" ]; then
    CONFIG="configs/deploy/agilex_real_rtc.yaml"
    # delay=0 与 delay>0 是两张形状不同的图,各要编译一次。只热前者的话,控制环里
    # 第一次带前缀的请求会触发约 41s 的编译(INFER_LATENCY_DEBUG 坑 4)。
    WARMUP_RTC="--warmup-rtc"
else
    CONFIG="configs/deploy/agilex_real.yaml"
    WARMUP_RTC=""
fi

# ⚠️ **必须开 compile**(RTC 尤其):不开的话 30Hz 下 d 会远超训练覆盖的 d<=11,
#    前缀条件形同虚设(overrun 接近 100%)。本服务把推理钉在专属线程上,
#    所以这里可以用 reduce-overhead(CUDA Graphs),不必退到 default。
#    ⚠️ 这个参数同时会**强制打开** compile_action_infer —— 非 RTC 的 yaml 里它默认是
#       false,靠这里的覆盖才拿到 CUDA Graphs。
COMPILE_MODE="reduce-overhead"

# CPU 线程数。**server 与 client 同机时必须限制 infer** —— 不限的话推理占满全部核,
# 30Hz 控制环(TCP 往返 + 下发)只能抢 CPU。真机实测:不限时节拍超时 25%、有效 23.7 Hz;
# 给控制环留出核之后 0% 超时、27.5 Hz。20 核机器给 4~6,80 核可放宽。
THREADS_PRE=8
THREADS_INFER=4

# ============================================================================
#  以下不用改
# ============================================================================
set -e
cd "$(dirname "$0")/.."
source env.sh

# Inductor 磁盘缓存:否则每次启动都要重付编译(实测 torch_threads_infer 限小之后
# 编译从 16.8s 涨到 125s)。缓存命中后只要几秒。
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$(pwd)/.cache/inductor}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "========================================="
echo " FastWAM 推理服务"
echo " RTC          : $RTC   $([ "$RTC" = "1" ] && echo '(前缀条件权重)' || echo '(基线权重)')"
echo " config       : $CONFIG"
echo " port         : $PORT"
echo " device       : $DEVICE"
echo " compile_mode : $COMPILE_MODE   <- 靠它把 d 压进训练支撑"
echo " threads      : pre=$THREADS_PRE infer=$THREADS_INFER"
echo " inductor 缓存: $TORCHINDUCTOR_CACHE_DIR"
echo "========================================="
echo " ⚠️ client 那边的 RTC= 必须与这里一致(scripts/start_client.sh)"
echo "========================================="

exec python scripts/fastwam_server.py \
    --config "$CONFIG" \
    --host 0.0.0.0 --port "$PORT" \
    --device "$DEVICE" \
    --compile-mode "$COMPILE_MODE" \
    --torch-threads-pre "$THREADS_PRE" \
    --torch-threads-infer "$THREADS_INFER" \
    --warmup 2 $WARMUP_RTC \
    "$@"
