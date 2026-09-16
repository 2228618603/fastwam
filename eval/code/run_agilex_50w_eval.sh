#!/usr/bin/env bash
set -euo pipefail

cd /home/chw/code/packages/FastWAM

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/data/chw/fastwam/checkpoints}"
export PYTHONPATH="/home/chw/code/packages/FastWAM:${PYTHONPATH:-}"

python eval/code/eval_agilex_fastwam.py \
  --stage eval \
  --device cuda:0 \
  --rand-device cpu \
  --max-windows 50 \
  --num-inference-steps 4 \
  --window-strategy sequential \
  "$@"
