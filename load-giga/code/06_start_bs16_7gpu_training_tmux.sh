#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chw/code/packages/FastWAM}
echo "Deprecated wrapper: launching bs14 stable config instead of bs16."
exec bash "${ROOT}/load-giga/code/06_start_bs14_7gpu_training_tmux.sh"
