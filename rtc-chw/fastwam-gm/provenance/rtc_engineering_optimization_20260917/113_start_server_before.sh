#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zzd/miniconda3/envs/giga/bin/python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
SELF_TEST="${SELF_TEST:-0}"
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"

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
)

if [[ "${SELF_TEST}" == "1" ]]; then
    args+=(--self-test)
fi
if [[ "${ENABLE_COMPILE}" == "1" ]]; then
    args+=(--torch-compile-action-stack --torch-compile-mode reduce-overhead)
fi

echo "checkpoint : ${CHECKPOINT}"
echo "norm stats : ${NORM_STATS}"
echo "device     : cuda:${GPU_ID}"
echo "port       : ${PORT}"

cd "${ROOT}"
exec "${PYTHON}" deploy/robot_server.py "${args[@]}"

