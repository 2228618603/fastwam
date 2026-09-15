#!/usr/bin/env bash
set -euo pipefail
cd "/home/chw/code/packages/FastWAM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/chw/fastwam/checkpoints
export PYTHONPATH=/home/chw/code/packages/FastWAM:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COLUMNS=240
export NO_COLOR=1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 stdbuf -oL -eL accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 7 \
  scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs14_5k \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.val.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  model.action_dit_pretrained_path=/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  batch_size=14 \
  num_workers=2 \
  max_steps=5000 \
  num_epochs=100 \
  gradient_accumulation_steps=1 \
  save_every=500 \
  eval_every=0 \
  log_every=1 \
  eval_num_inference_steps=4 \
  mixed_precision=bf16 \
  learning_rate=1.0e-5 \
  weight_decay=1.0e-2 \
  wandb.enabled=false \
  2>&1 | tee -a "/mnt/data/chw/fastwam/logs/giga_to_fastwam/agilex_empty_box_giga_init_bs14_5k.log"
