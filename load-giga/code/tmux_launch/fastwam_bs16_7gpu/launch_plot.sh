#!/usr/bin/env bash
set -euo pipefail
cd "/home/chw/code/packages/FastWAM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
python load-giga/code/07_live_loss_plot.py \
  --log "/mnt/data/chw/fastwam/logs/giga_to_fastwam/agilex_empty_box_giga_init_bs16_5k.log" \
  --out-dir "/home/chw/code/packages/FastWAM/load-giga/code/live" \
  --interval 5 \
  --gif \
  --gif-tail 300
