#!/usr/bin/env bash
# FastWAM + Agilex 训练/部署环境变量。
# 用法: source env.sh   (在仓库任意位置 source 都行,会自己 cd 到仓库根)
#
# 约束 6：所有下载/缓存/输出都落在项目目录内，不写到 $HOME 或其他位置。

# 切到本文件所在目录 = 仓库根。不写死路径,这样搬到别的机器/别的路径不用改。
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)" || return 1

# 把 fastwam conda 环境放到 PATH 最前面。
# 必须有这一步：scripts/train_zero1.sh 直接调用 `accelerate`（第 110 行），
# 只 source 环境变量而不改 PATH 会得到 "accelerate: command not found" (exit 127)。
#
# 换机器时不用改这里：先 `export FASTWAM_ENV=/你的/conda/env` 再 source 即可。
export FASTWAM_ENV="${FASTWAM_ENV:-/root/miniforge3/envs/fastwam}"
if [[ -d "$FASTWAM_ENV/bin" ]]; then
  export PATH="$FASTWAM_ENV/bin:$PATH"
else
  echo "[env] ⚠️ FASTWAM_ENV=$FASTWAM_ENV 不存在,PATH 未前置。" >&2
  echo "[env]    换机器请先: export FASTWAM_ENV=/你的/conda/env  再 source env.sh" >&2
fi

# deepspeed 需要 CUDA_HOME 才能 JIT 编译算子（**只训练用**，部署不需要）
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# 容器镜像默认设了 NCCL_DEBUG=INFO，会把训练日志刷成几千行 NCCL 噪声，
# 真正的 loss/报错被埋掉（实测 20 步的冒烟日志 1243 行里绝大部分是 NCCL）。
# 降到 WARN：仍保留真正的通信告警，日志可读。
export NCCL_DEBUG=WARN

# FastWAM 自身的 Wan 权重目录（helpers/io.py:49-51 读这个变量）
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

# 把 HF / ModelScope 的缓存也关在项目内
export HF_HOME="$(pwd)/.cache/huggingface"
export MODELSCOPE_CACHE="$(pwd)/.cache/modelscope"

# 下载源：modelscope 为 FastWAM 默认（io.py:33-38）；国内更快
# 需要走 HuggingFace 时取消下面这行的注释（HF_ENDPOINT 已指向 hf-mirror）
# export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH" "$HF_HOME" "$MODELSCOPE_CACHE"

# wandb 凭证。存在 .wandb_key（chmod 600，已加入 .gitignore 且不同步到 OSS）时
# 导出 WANDB_API_KEY —— 它的优先级高于 ~/.netrc，所以不需要改动 netrc，
# 也不会影响已经在跑、已用旧凭证认证过的训练进程。
#   对应账号: menggao6073 / entity: menggao6073-geek-
if [[ -f .wandb_key ]]; then
  WANDB_API_KEY="$(tr -d ' \t\n\r' < .wandb_key)"
  export WANDB_API_KEY
  _wandb_state="已设置 (menggao6073-geek-)"
else
  _wandb_state="未设置（回退到 ~/.netrc）"
fi

echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] PATH 前置=$FASTWAM_ENV/bin  (accelerate: $(command -v accelerate || echo 未找到))"
echo "[env] DIFFSYNTH_MODEL_BASE_PATH=$DIFFSYNTH_MODEL_BASE_PATH"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
echo "[env] HF_ENDPOINT=${HF_ENDPOINT:-<unset>}"
echo "[env] WANDB_API_KEY=$_wandb_state"
