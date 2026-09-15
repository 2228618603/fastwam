#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chw/code/packages/FastWAM}
echo "Deprecated wrapper: current requested training config is bs8 on 8 GPUs for 100000 steps."
echo "Launching: ${ROOT}/load-giga/code/06_start_bs8_8gpu_training_tmux.sh"
exec bash "${ROOT}/load-giga/code/06_start_bs8_8gpu_training_tmux.sh"
