#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zzd/miniconda3/envs/giga/bin/python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
SELF_TEST="${SELF_TEST:-0}"

GGUF="${GGUF:-/mnt/data/zzd/giga-world-policy/weights/giga_470_aligned_l6g_r6g_step100000.gguf}"
TOKEN_FILE="${TOKEN_FILE:-${ROOT}/prompt_tokens/fixed_task_token.pt}"
WAM_LIB="${WAM_LIB:-/home/zzd/project/wam.cpp/build-cuda/libwam_c_api.so}"

for path in "${GGUF}" "${TOKEN_FILE}" "${WAM_LIB}"; do
    if [[ ! -e "${path}" ]]; then
        echo "required path is unavailable: ${path}" >&2
        exit 1
    fi
done

args=(
    --backend cpp
    --device "cuda:${GPU_ID}"
    --gguf "${GGUF}"
    --wam-lib "${WAM_LIB}"
    --token-file "${TOKEN_FILE}"
    --host 0.0.0.0
    --port "${PORT}"
)

if [[ "${SELF_TEST}" == "1" ]]; then
    args+=(--self-test)
fi

echo "gguf    : ${GGUF}"
echo "wam lib : ${WAM_LIB}"
echo "device  : cuda:${GPU_ID}"
echo "port    : ${PORT}"

cd "${ROOT}"
exec "${PYTHON}" deploy/robot_server.py "${args[@]}"
