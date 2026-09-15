#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chw/code/packages/FastWAM}
SESSION=${SESSION:-fastwam_bs14_7gpu}
GPUS=${GPUS:-1,2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-7}
BATCH_SIZE=${BATCH_SIZE:-14}
RUN_DIR=${RUN_DIR:-/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs14_5k}
LIVE_DIR=${LIVE_DIR:-${ROOT}/load-giga/code/live}
LAUNCH_DIR=${LAUNCH_DIR:-${ROOT}/load-giga/code/tmux_launch/${SESSION}}
LOG_DIR=${LOG_DIR:-/mnt/data/chw/fastwam/logs/giga_to_fastwam}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/agilex_empty_box_giga_init_bs14_5k.log}

mkdir -p "${RUN_DIR}" "${LIVE_DIR}" "${LAUNCH_DIR}" "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 0
fi

cat > "${LAUNCH_DIR}/launch_train.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/chw/fastwam/checkpoints
export PYTHONPATH=${ROOT}:\${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COLUMNS=240
export NO_COLOR=1
CUDA_VISIBLE_DEVICES=${GPUS} stdbuf -oL -eL accelerate launch \\
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \\
  --num_processes ${NUM_PROCESSES} \\
  scripts/train.py \\
  task=agilex_empty_box_uncond_3cam384 \\
  resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \\
  output_dir=${RUN_DIR} \\
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \\
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \\
  data.train.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \\
  data.val.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \\
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \\
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \\
  data.train.use_text_embed_cache=true \\
  data.val.use_text_embed_cache=true \\
  model.load_text_encoder=false \\
  model.action_dit_pretrained_path=/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \\
  batch_size=${BATCH_SIZE} \\
  num_workers=2 \\
  max_steps=5000 \\
  num_epochs=100 \\
  gradient_accumulation_steps=1 \\
  save_every=500 \\
  eval_every=0 \\
  log_every=1 \\
  eval_num_inference_steps=4 \\
  mixed_precision=bf16 \\
  learning_rate=1.0e-5 \\
  weight_decay=1.0e-2 \\
  wandb.enabled=false \\
  2>&1 | tee -a "${LOG_FILE}"
EOF

cat > "${LAUNCH_DIR}/launch_plot.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
python load-giga/code/07_live_loss_plot.py \\
  --log "${LOG_FILE}" \\
  --out-dir "${LIVE_DIR}" \\
  --interval 5 \\
  --gif \\
  --gif-tail 300 \\
  --tmux-pane "${SESSION}:train" \\
  --tmux-lines 5000
EOF

tmux new-session -d -s "${SESSION}" -n train "bash '${LAUNCH_DIR}/launch_train.sh'; echo; echo '[train exited]'; exec bash"
tmux new-window -t "${SESSION}" -n plot "bash '${LAUNCH_DIR}/launch_plot.sh'; echo; echo '[plot exited]'; exec bash"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Train log: ${LOG_FILE}"
echo "Run dir: ${RUN_DIR}"
echo "Launch scripts: ${LAUNCH_DIR}"
echo "Live monitor dir: ${LIVE_DIR}"
echo "Open files directly: ${LIVE_DIR}/loss_curve.gif or ${LIVE_DIR}/index.html"
