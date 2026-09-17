#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zzd/miniconda3/envs/giga/bin/python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
SELF_TEST="${SELF_TEST:-0}"
# 原 ENABLE_COMPILE="${ENABLE_COMPILE:-0}"：默认 eager 的开销无法充分隐藏在 RTC 窗口内。
ENABLE_COMPILE="${ENABLE_COMPILE:-1}"
COMPILE_PREFIX="${COMPILE_PREFIX:-1}"
COMPILE_SCOPE="${COMPILE_SCOPE:-action-blocks}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
INFER_THREADS="${INFER_THREADS:-$(( $(nproc) / 4 ))}"
[[ "${INFER_THREADS}" -ge 2 ]] || INFER_THREADS=2
export TOKENIZERS_PARALLELISM=false
export GIGA_MODELS_LIGHT_IMPORT=1

CHECKPOINT="${CHECKPOINT:-/mnt/data/models/zzd/checkpoint/giga_470_aligned_l6g_r6g_bs128_gpu8_b16g1_100k/models/checkpoint_epoch_13_step_100000/transformer_ema}"
BASE_MODEL="${BASE_MODEL:-/mnt/data/zzd/giga-world-policy/Wan2.2-TI2V-5B-Diffusers}"
NORM_STATS="${NORM_STATS:-/mnt/data/zzd/giga-world-policy/geek_data/norm_stats_aligned_l6g_r6g.json}"
TOKEN_FILE="${TOKEN_FILE:-${ROOT}/prompt_tokens/fixed_task_token.pt}"

for path in "${CHECKPOINT}" "${BASE_MODEL}" "${NORM_STATS}" "${TOKEN_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "required path is unavailable: ${path}" >&2
        exit 1
    fi
done

args=(
    --backend python
    --device "cuda:${GPU_ID}"
    --checkpoint "${CHECKPOINT}"
    --base-model "${BASE_MODEL}"
    --norm-stats "${NORM_STATS}"
    --token-file "${TOKEN_FILE}"
    --host 0.0.0.0
    --port "${PORT}"
    --torch-threads "${INFER_THREADS}"
    --warmup "${WARMUP:-3}"
    --warmup-rtc-delay "${WARMUP_RTC_DELAY:-8}"
    --benchmark-repeat "${BENCHMARK_REPEAT:-5}"
)

if [[ "${SELF_TEST}" == "1" ]]; then
    args+=(--self-test)
    [[ -z "${SELF_TEST_OUTPUT:-}" ]] || args+=(--self-test-output "${SELF_TEST_OUTPUT}")
fi
if [[ "${ENABLE_COMPILE}" == "1" ]]; then
    # 原 args+=(--torch-compile-action-stack --torch-compile-mode reduce-overhead)
    # 改用完整编译入口，默认采用 gaomeng 实测的 action-blocks + prefix 配置。
    args+=(--compile --compile-scope "${COMPILE_SCOPE}" --torch-compile-mode "${COMPILE_MODE}")
    [[ "${COMPILE_PREFIX}" != "1" ]] || args+=(--compile-prefix)
fi

echo "checkpoint : ${CHECKPOINT}"
echo "norm stats : ${NORM_STATS}"
echo "device     : cuda:${GPU_ID}"
echo "port       : ${PORT}"
echo "compile    : ${ENABLE_COMPILE}, prefix=${COMPILE_PREFIX}, scope=${COMPILE_SCOPE}, mode=${COMPILE_MODE}"
echo "threads    : ${INFER_THREADS}; warmup RTC delay=${WARMUP_RTC_DELAY:-8}"

cd "${ROOT}"
exec "${PYTHON}" deploy/robot_server.py "${args[@]}"
