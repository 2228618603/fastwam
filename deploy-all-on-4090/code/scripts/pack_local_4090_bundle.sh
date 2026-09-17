#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p weights \
  model_cache/Wan-AI \
  model_cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors

rsync -ah --info=progress2 \
  /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_035000.pt \
  weights/step_035000.pt

rsync -ah --info=progress2 \
  /mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  weights/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

rsync -ah --info=progress2 \
  /mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B/ \
  model_cache/Wan-AI/Wan2.2-TI2V-5B/

rsync -ah --info=progress2 \
  /mnt/data/chw/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
  model_cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors

python scripts/verify_bundle_integrity.py --root .
