#!/bin/bash
# ============================================================================
#  把 FastWAM 部署所需文件同步到真机(同机部署:4090 + ROS2 + Piper 在一台)
# ============================================================================
#  用法:
#    bash scripts/sync_to_robot.sh user@192.168.1.10:/path/to/FastWAM          # 全量
#    bash scripts/sync_to_robot.sh user@192.168.1.10:/path/to/FastWAM --code   # 只同步代码
#    DRY=1 bash scripts/sync_to_robot.sh user@...                              # 只看要传什么
#
#  为什么不是"把整个仓库 rsync 过去":仓库 100 GB+,其中 86 GB 部署时用不到 ——
#  底座 DiT(18.8G)+ ActionDiT(2.0G)被 skip_base_dit_load 跳过、T5(11G)走 embedding
#  缓存绕开、数据集(30G)运行时完全不读(已实测:把数据集移走后 server+client 照常跑)。
#  详见 DEPLOY_DESIGN.md §10 与 SPLIT_DEPLOY_DESIGN.md。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DST="${1:-}"
if [ -z "$DST" ]; then
    echo "用法: bash scripts/sync_to_robot.sh user@robot-ip:/path/to/FastWAM [--code]"
    exit 2
fi
ONLY_CODE="${2:-}"
RS=(rsync -av --human-readable)
[ -n "${DRY:-}" ] && RS+=(--dry-run) && echo ">>> DRY RUN(只列出,不实际传输)"

# ---- 1) 代码(约 2 MB,每次改完都要重新同步)---------------------------------
echo ""
echo ">>> [1/3] 代码与配置"
"${RS[@]}" \
    --exclude '__pycache__' --exclude '*.pyc' \
    src/ "$DST/src/"
"${RS[@]}" configs/ "$DST/configs/"
"${RS[@]}" \
    scripts/fastwam_server.py \
    scripts/fastwam_client.py \
    scripts/deploy_robot_client.py \
    scripts/deploy_real.py \
    scripts/deploy_fake_service.py \
    scripts/start_server.sh \
    scripts/start_client.sh \
    scripts/sync_to_robot.sh \
    "$DST/scripts/"
"${RS[@]}" env.sh pyproject.toml \
    DEPLOY_DESIGN.md SPLIT_DEPLOY_DESIGN.md INFER_LATENCY_DEBUG.md RTC_TRAIN_DESIGN.md \
    "$DST/"

if [ "$ONLY_CODE" = "--code" ]; then
    echo ""
    echo ">>> 只同步代码,完成。(权重/统计量未动)"
    exit 0
fi

# ---- 2) 统计量与文本缓存(约 1.2 MB,几乎不变)-------------------------------
# ⚠️ 这两样错了/缺了都是**静默失败**:
#   dataset_stats.json  用错一份 -> 动作完全失真且不报错
#   text embeds 缓存    缺了     -> 启动时直接退出(报缺失路径),算好事
echo ""
echo ">>> [2/3] 归一化统计量 + T5 embedding 缓存 + 起始位姿分布"
"${RS[@]}" -R \
    runs/_shared/dataset_stats.json \
    runs/_shared/start_pose_stats.json \
    data/text_embeds_cache/agilex/ \
    "$DST/"

# ---- 3) 权重(13.4 GB / 25.4 GB,只需传一次)---------------------------------
# VAE 必需:它**不在 checkpoint 里**(训练时冻结,trainer 不存),而相机图必须经它编码成 latent。
# 这是"要搬好几个文件"的唯一实质原因。
echo ""
echo ">>> [3/3] 权重(大文件,--partial 支持断点续传)"
"${RS[@]}" -R --partial --progress \
    checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
    runs/agilex_rtc_3cam_384_1e-4/rtc_paper8k/checkpoints/weights/step_008000.pt \
    runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_032000.pt \
    "$DST/"

cat <<'EOF'

============================================================================
 同步完成。真机上接着做:
============================================================================
 1) 改 env.sh 里的 FASTWAM_ENV(指向真机的 conda 环境),其余用 $(pwd) 自动跟随
 2) source env.sh  &&  python -c "import torch;print(torch.cuda.get_device_name(0))"
 3) 实测延迟定 d(不需要硬件、不需要数据集,搬完立刻能跑):
      python scripts/fastwam_server.py --config configs/deploy/agilex_real_rtc.yaml \
          --benchmark --compile-mode both --steps-list 5,10
 4) 三个进程按顺序起,详见 SPLIT_DEPLOY_DESIGN.md §9
============================================================================
EOF
