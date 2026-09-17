#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="/media/geekplus/PortableSSD/zzd_test/weights/rtc/giga_470_aligned_l6g_r6g_rtc_step30000_transformer_bf16" \
BASE_MODEL="/media/geekplus/PortableSSD/zzd_test/weights/base/Wan2.2-TI2V-5B-Diffusers" \
NORM_STATS="/media/geekplus/PortableSSD/zzd_test/weights/norm/norm_stats_aligned_l6g_r6g.json" \
TOKEN_FILE="${ROOT}/prompt_tokens/fixed_task_token.pt" \
PYTHON="${PYTHON:-$(which python)}" GPU_ID="${GPU_ID:-0}" PORT="${PORT:-8000}" SELF_TEST="${SELF_TEST:-0}" \
bash "${ROOT}/deploy/start_server_aligned_python.sh"
