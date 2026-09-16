#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${PYTHON_BIN:-}" && -x /home/chw/miniconda3/envs/fastwam/bin/python ]]; then
  PYTHON_BIN=/home/chw/miniconda3/envs/fastwam/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m app.local_robot_runner \
  --mode step \
  --max-steps "${MAX_STEPS:-1}" \
  --confirm-safety \
  --goto-start \
  --speed-percent "${SPEED_PERCENT:-10}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-4}" \
  --cuda-memory-fraction "${CUDA_MEMORY_FRACTION:-0.45}"
