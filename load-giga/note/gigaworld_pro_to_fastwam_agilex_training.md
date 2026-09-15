# GigaWorld-1 Pro 到 FastWAM 的 AgileX Empty The Box 训练实施手册

本文档用于在不修改 FastWAM 原始源码和原始配置文件的前提下，将 GigaWorld-1 Pro Stage-1 世界模型知识迁移到 FastWAM，并在 AgileX `empty the box` 数据集上继续训练、保存、恢复和评估。

固定工作目录：

```bash
cd /home/chw/code/packages/FastWAM
```

固定输入路径：

```text
FastWAM 源码: /home/chw/code/packages/FastWAM
GigaWorld-1 源码: /home/chw/code/packages/FastWAM/giga-world-1/giga-world-1
GigaWorld-1 权重: /mnt/data/chw/giga-world-1
FastWAM 权重/输出: /mnt/data/chw/fastwam
AgileX 数据集: /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711
新增脚本目录: /home/chw/code/packages/FastWAM/load-giga/code
实施文档目录: /home/chw/code/packages/FastWAM/load-giga/note
```

硬约束：

- 不修改 FastWAM 原始源码。
- 不修改 FastWAM 原始 `configs/` 文件。
- 新脚本只放到 `/home/chw/code/packages/FastWAM/load-giga/code`。
- 新 checkpoint、训练输出、日志、缓存只放到 `/mnt/data/chw/fastwam`。
- 不使用 `hustvl/FasterWAM` 的 LIBERO 或 RoboTwin checkpoint 作为初始化。
- 不使用 GigaWorld Nano 作为主迁移来源。
- 不把 GigaWorld scene LoRA 当成完整模型权重。

## 总体方案

FastWAM 当前默认模型由以下部分组成：

```text
video expert: Wan2.2-TI2V-5B DiT
action expert: ActionDiT
MoT: video/action 两个 expert 的 mixed attention 封装
VAE: Wan2.2 VAE
text encoder: Wan/T5 text encoder, 可预计算后训练时不加载
tokenizer: Wan2.1 T2V 1.3B 里的 UMT5 tokenizer
proprio encoder: Linear(proprio_dim=14, text_dim=4096)
```

本次 FastWAM 初始化必须使用：

```text
model_id: Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
action_dit_pretrained_path: /mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

FastWAM 默认基础权重来源和用途：

```text
/mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B/
  用途: FastWAM video expert 的默认 Wan2.2-TI2V-5B DiT 初始化。

/mnt/data/chw/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
  用途: 视频帧与 latent 之间编码/解码。训练中冻结。

/mnt/data/chw/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
  用途: 文本 prompt 编码。建议预计算 text embeddings 后训练时不加载。

/mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
  用途: UMT5 tokenizer。

/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  用途: FastWAM action expert 默认 backbone 初始化。
```

`resume` 的含义必须区分清楚：

```text
resume=null
  含义: 不加载任何 FastWAM 训练 checkpoint。
  实际初始化:
    video expert 从 Wan-AI/Wan2.2-TI2V-5B 加载。
    action expert 从 ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt 加载。
    VAE 从 Wan2.2_VAE.safetensors 加载。
    text encoder/tokenizer 仅在 load_text_encoder=true 或预计算 text embeds 时使用。
    MoT 由 video/action expert 现场组合。
    proprio encoder 随机初始化。
  是否正确: 这是“禁用 RoboTwin/LIBERO checkpoint 后”的正确 FastWAM 默认初始化。

resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
  含义: 加载本文档 Stage 5 生成的自定义 FastWAM 初始化 checkpoint。
  实际初始化:
    video expert 已包含 Giga Pro 可迁移权重覆盖。
    action expert 仍来自默认 ActionDiT。
    proprio encoder 来自 Stage 5 初始化。
  是否正确: 这是完成 Giga 迁移后用于 AgileX 训练的推荐入口。

resume=/mnt/data/chw/fastwam/checkpoints/robotwin_release/robotwin_uncond_3cam_384.pt
  含义: 加载 RoboTwin 训练后的 FastWAM checkpoint。
  是否正确: 本任务禁止使用，必须覆盖掉。

resume=libero/step_021700.pt 或 robotwin/step_029355.pt
  含义: 加载 FasterWAM 发布的 LIBERO/RoboTwin 训练状态。
  是否正确: 本任务禁止使用。
```

GigaWorld-1 Pro 迁移来源：

```text
/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/
/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer/
```

GigaWorld Pro 与 FastWAM 公共结构：

```text
num_layers: 30
hidden_dim / inner_dim: 3072
ffn_dim: 14336
num_heads: 24
attn_head_dim: 128
text_dim: 4096
freq_dim: 256
patch_size: [1, 2, 2]
out_channels / out_dim: 48
```

不可整体直接加载的原因：

```text
GigaWorld Pro transformer config:
  _class_name: HeliosTransformer3DModelFunCtrl
  in_channels: 148
  out_channels: 48
  has_multi_term_memory_patch: true
  guidance_cross_attn: true

FastWAM video expert config:
  class: WanVideoDiT
  in_dim: 48
  out_dim: 48
  has_image_input: false
  action_conditioned: false
  fuse_vae_embedding_in_latents: true
```

因此：

- 可以迁移 transformer blocks、self/cross attention、FFN、time/text embedding、output projection 中 shape 与语义都匹配的权重。
- 不能直接迁移 Giga `patch_embedding.weight`，因为输入通道 `148 != 48`。
- 不能直接迁移 `patch_short`、`patch_mid`、`patch_long`，这些是 Giga FunControl/history memory 分支，FastWAM 没有对应模块。
- 不能直接迁移 scene LoRA，它不是完整 transformer。
- 不能把 `load_state_dict(strict=False)` 的成功返回当成迁移成功，必须逐 key 比较 shape、语义和最终覆盖比例。

推荐使用方式：

```text
主方案: Giga Pro 直接初始化 FastWAM video expert 的可映射主干参数。
可选增强: 训练稳定后，再把 Giga Pro 作为 teacher 做视频噪声预测蒸馏。
不推荐: 直接把整个 Giga FunControl transformer 塞进 FastWAM。
```

## Stage 0：目录、分支和产物边界确认

本阶段目标：确认当前仓库、分支、输入输出目录和“不污染原仓库”的边界。

### 输入

- 源码目录：
  - `/home/chw/code/packages/FastWAM`
  - `/home/chw/code/packages/FastWAM/giga-world-1/giga-world-1`
- 权重目录：
  - `/mnt/data/chw/giga-world-1`
  - `/mnt/data/chw/fastwam`
- 数据集目录：
  - `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- 环境变量：
  - `HF_HOME=/mnt/data/chw/fastwam/hf_cache`
  - `MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache`
  - `PYTHONPATH=/home/chw/code/packages/FastWAM:/home/chw/code/packages/FastWAM/giga-world-1/giga-world-1`
- GPU 要求：
  - 最低验证：1 张 40GB 级别 GPU，batch size 1。
  - 推荐训练：8 张 A100/H20 级别 GPU，bf16，ZeRO-2。
- 磁盘要求：
  - Giga Pro Stage-1 diffusers 约 20GB transformer，加上 VAE/T5 后建议预留 80GB。
  - FastWAM 训练输出建议预留 200GB 以上。

### 操作步骤

```bash
cd /home/chw/code/packages/FastWAM

mkdir -p /home/chw/code/packages/FastWAM/load-giga/code
mkdir -p /home/chw/code/packages/FastWAM/load-giga/note
mkdir -p /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam
mkdir -p /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init
mkdir -p /mnt/data/chw/fastwam/logs/giga_to_fastwam
mkdir -p /mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711
mkdir -p /mnt/data/chw/fastwam/dataset_stats/agilex_empty_box_542_0711

git status --short --branch
test -d /home/chw/code/packages/FastWAM/load-giga/code
test -d /home/chw/code/packages/FastWAM/load-giga/note
test -d /mnt/data/chw/giga-world-1
test -d /mnt/data/chw/fastwam
test -d /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711
```

建议训练 shell 统一设置：

```bash
export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export PYTHONPATH=/home/chw/code/packages/FastWAM:/home/chw/code/packages/FastWAM/giga-world-1/giga-world-1:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
```

### 预期输出

```text
git branch: dev/giga-to-fast
新增目录:
  /home/chw/code/packages/FastWAM/load-giga/code
  /home/chw/code/packages/FastWAM/load-giga/note
  /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam
  /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init
```

### 检查点 Checkpoint

```text
Checkpoint 0.1:
[ ] 当前目录是 /home/chw/code/packages/FastWAM
[ ] 当前分支是 dev/giga-to-fast
[ ] /home/chw/code/packages/FastWAM/load-giga/code 存在
[ ] /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam 存在
[ ] /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711 存在
```

## Stage 1：GigaWorld Pro 权重完整性检查

本阶段目标：确认 GigaWorld Pro Stage-1 完整 diffusers 权重已经下载完毕，并确认不是 Nano、不是 scene LoRA。

### 输入

- Giga Pro transformer：
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer`
- Giga Pro VAE/T5：
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/vae`
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/text_encoder`
- Giga 官方源码：
  - `/home/chw/code/packages/FastWAM/giga-world-1/giga-world-1`
- GPU：
  - 本阶段不需要 GPU。
- 磁盘：
  - 需要能读取 `/mnt/data/chw/giga-world-1`。

### 操作步骤

```bash
cd /home/chw/code/packages/FastWAM

GIGA_PRO=/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers

test -f ${GIGA_PRO}/model_index.json
test -f ${GIGA_PRO}/transformer/config.json
test -f ${GIGA_PRO}/transformer/diffusion_pytorch_model.safetensors.index.json
test -f ${GIGA_PRO}/transformer/diffusion_pytorch_model-00001-of-00003.safetensors
test -f ${GIGA_PRO}/transformer/diffusion_pytorch_model-00002-of-00003.safetensors
test -f ${GIGA_PRO}/transformer/diffusion_pytorch_model-00003-of-00003.safetensors
test -f ${GIGA_PRO}/vae/diffusion_pytorch_model.safetensors

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers")
cfg = json.loads((root / "transformer/config.json").read_text())
idx = json.loads((root / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())

print("class:", cfg.get("_class_name"))
print("model_type:", cfg.get("model_type"))
print("in_channels:", cfg.get("in_channels"))
print("out_channels:", cfg.get("out_channels"))
print("num_layers:", cfg.get("num_layers"))
print("num_heads:", cfg.get("num_attention_heads"))
print("head_dim:", cfg.get("attention_head_dim"))
print("ffn_dim:", cfg.get("ffn_dim"))
print("text_dim:", cfg.get("text_dim"))
print("patch_size:", cfg.get("patch_size"))
print("weight_keys:", len(idx["weight_map"]))
print("total_size:", idx["metadata"].get("total_size"))

assert cfg["_class_name"] == "HeliosTransformer3DModelFunCtrl"
assert cfg["model_type"] == "wan2.2_5b"
assert cfg["num_layers"] == 30
assert cfg["num_attention_heads"] == 24
assert cfg["attention_head_dim"] == 128
assert cfg["ffn_dim"] == 14336
assert cfg["text_dim"] == 4096
assert cfg["out_channels"] == 48
assert cfg["in_channels"] == 148
PY
```

如果下载来自 Hugging Face，可用以下命令重新拉取或补全缺失文件：

```bash
cd /mnt/data/chw
huggingface-cli download open-gigaai/Giga-World-1 \
  --local-dir /mnt/data/chw/giga-world-1 \
  --resume-download
```

如果使用 ModelScope，需要按 Giga 官方下载工具：

```bash
cd /home/chw/code/packages/FastWAM/giga-world-1/giga-world-1
bash tools/download_tool/download_giga_world.sh \
  --platform modelscope \
  --target model \
  --output-dir /mnt/data/chw
```

### 预期输出

```text
class: HeliosTransformer3DModelFunCtrl
model_type: wan2.2_5b
in_channels: 148
out_channels: 48
num_layers: 30
num_heads: 24
head_dim: 128
ffn_dim: 14336
text_dim: 4096
patch_size: [1, 2, 2]
weight_keys: 831 左右
total_size: 20090198784 左右
```

### 检查点 Checkpoint

```text
Checkpoint 1.1:
[ ] 使用的是 stage1/pro/Giga-World-1-pro-stage1_final-diffusers
[ ] transformer/config.json 的 model_type 是 wan2.2_5b
[ ] transformer/config.json 的 in_channels 是 148
[ ] transformer/config.json 的 out_channels 是 48
[ ] 3 个 transformer safetensors shard 都存在
[ ] 没有把 stage1/pro/Giga-World-1-pro-stage1_scene_lora 当成完整模型
```

## Stage 2：FastWAM 默认初始化和禁用 RoboTwin/LIBERO checkpoint

本阶段目标：确认 FastWAM 使用默认 Wan2.2-TI2V-5B、默认 ActionDiT，不加载 LIBERO/RoboTwin checkpoint。

### 输入

- FastWAM 模型配置：
  - `/home/chw/code/packages/FastWAM/configs/model/fastwam.yaml`
- FastWAM 训练主配置：
  - `/home/chw/code/packages/FastWAM/configs/train.yaml`
- 当前 AgileX task 配置：
  - `/home/chw/code/packages/FastWAM/configs/task/agilex_empty_box_uncond_3cam384.yaml`
- 默认 ActionDiT：
  - `/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- GPU：
  - 仅检查配置不需要 GPU。
  - 构造模型建议 1 张 40GB 级别 GPU。

### 操作步骤

先检查原配置中的默认项和冲突项：

```bash
cd /home/chw/code/packages/FastWAM

python - <<'PY'
from omegaconf import OmegaConf
from pathlib import Path

model = OmegaConf.load("/home/chw/code/packages/FastWAM/configs/model/fastwam.yaml")
task = OmegaConf.load("/home/chw/code/packages/FastWAM/configs/task/agilex_empty_box_uncond_3cam384.yaml")

print("model_id:", model.model_id)
print("tokenizer_model_id:", model.tokenizer_model_id)
print("action_dit_pretrained_path:", model.action_dit_pretrained_path)
print("load_text_encoder:", model.load_text_encoder)
print("task.resume:", task.get("resume"))

assert model.model_id == "Wan-AI/Wan2.2-TI2V-5B"
assert model.tokenizer_model_id == "Wan-AI/Wan2.1-T2V-1.3B"
assert "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" in model.action_dit_pretrained_path
PY
```

训练时必须覆盖掉 task 里的 RoboTwin resume：

```bash
cd /home/chw/code/packages/FastWAM

python scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=null \
  output_dir=/mnt/data/chw/fastwam/runs/config_check_no_robotwin \
  max_steps=0
```

注意：上面命令只用于验证 Hydra override 是否接受 `resume=null`。如果训练器不接受 `max_steps=0`，不用继续执行此命令，后续以 Stage 6 的 1-step dry run 为准。

`resume=null` 后，训练器的行为是：

```text
1. create_fastwam() 先构建 FastWAM。
2. video expert 加载 Wan-AI/Wan2.2-TI2V-5B。
3. action expert 加载 checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt。
4. VAE 加载 Wan2.2_VAE.safetensors。
5. 因为 load_text_encoder=false，训练样本必须提供 context/context_mask text embedding cache。
6. proprio encoder 随机初始化。
7. Wan22Trainer._resume_or_load_checkpoint() 看到 resume=null 后直接返回，不再覆盖模型权重。
```

因此，`resume=null` 不是“完全随机初始化”。它表示“不加载训练 checkpoint”，但仍然会按 `configs/model/fastwam.yaml` 加载 FastWAM 默认基础权重。这正是本任务要求的默认初始化。

### 预期输出

```text
model_id: Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
load_text_encoder: false
task.resume: /mnt/data/chw/fastwam/checkpoints/robotwin_release/robotwin_uncond_3cam_384.pt
```

解释：

```text
task.resume 显示 RoboTwin 是当前原始配置状态。
正式训练必须通过 override 写 resume=null，不能使用这个 RoboTwin checkpoint。
写 resume=null 后，实际使用 Wan2.2-TI2V-5B + 默认 ActionDiT + Wan2.2 VAE + 随机 proprio encoder。
完成 Stage 5 后，正式训练应改用 resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt。
```

### 检查点 Checkpoint

```text
Checkpoint 2.1:
[ ] FastWAM model_id 是 Wan-AI/Wan2.2-TI2V-5B
[ ] tokenizer_model_id 是 Wan-AI/Wan2.1-T2V-1.3B
[ ] ActionDiT 使用 /mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
[ ] 迁移前 baseline/dry run 命令包含 resume=null
[ ] 完成 Stage 5 后正式训练命令包含 resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
[ ] 正式训练命令不包含 libero/step_021700.pt
[ ] 正式训练命令不包含 robotwin/step_029355.pt
[ ] 正式训练命令不包含 robotwin_release/robotwin_uncond_3cam_384.pt
[ ] 如果 model.load_text_encoder=false，则 data.train.use_text_embed_cache=true 且 data.val.use_text_embed_cache=true
```

## Stage 3：权重映射审计脚本

本阶段目标：创建审计脚本，逐 key 对比 Giga Pro transformer 与 FastWAM video expert，输出可迁移、不可迁移、shape mismatch 和语义拒绝列表。

### 输入

- 新脚本：
  - `/home/chw/code/packages/FastWAM/load-giga/code/01_inspect_giga_fastwam.py`
- Giga Pro transformer：
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer`
- FastWAM 默认 Wan2.2：
  - `/mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B`
- ActionDiT：
  - `/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- 输出报告：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.json`
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.md`
- GPU：
  - 建议先 CPU 读取 safetensors metadata。
  - 如果构造 FastWAM 实例，需要 1 张 GPU。

### 操作步骤

安装或确认依赖：

```bash
cd /home/chw/code/packages/FastWAM
python - <<'PY'
import torch
print("torch:", torch.__version__)
try:
    import safetensors
    print("safetensors:", safetensors.__version__)
except Exception as exc:
    raise SystemExit("safetensors is required in the training env: " + repr(exc))
PY
```

如果缺少 `safetensors`：

```bash
python -m pip install safetensors
```

创建脚本文件：

```text
/home/chw/code/packages/FastWAM/load-giga/code/01_inspect_giga_fastwam.py
```

脚本核心逻辑：

```text
1. 读取 Giga transformer/config.json。
2. 读取 Giga transformer/diffusion_pytorch_model.safetensors.index.json。
3. 构造 FastWAM video expert 的 expected key/shape：
   - 用 configs/model/fastwam.yaml 的 video_dit_config。
   - 实例化 WanVideoDiT，不需要加载完整训练器。
4. 定义明确 key 映射：
   - condition_embedder.text_embedder.linear_1 -> text_embedding.0
   - condition_embedder.text_embedder.linear_2 -> text_embedding.2
   - condition_embedder.time_embedder.linear_1 -> time_embedding.0
   - condition_embedder.time_embedder.linear_2 -> time_embedding.2
   - condition_embedder.time_proj -> time_projection.1
   - blocks.N.attn1.to_q -> blocks.N.self_attn.q
   - blocks.N.attn1.to_k -> blocks.N.self_attn.k
   - blocks.N.attn1.to_v -> blocks.N.self_attn.v
   - blocks.N.attn1.to_out.0 -> blocks.N.self_attn.o
   - blocks.N.attn1.norm_q -> blocks.N.self_attn.norm_q
   - blocks.N.attn1.norm_k -> blocks.N.self_attn.norm_k
   - blocks.N.attn2.to_q -> blocks.N.cross_attn.q
   - blocks.N.attn2.to_k -> blocks.N.cross_attn.k
   - blocks.N.attn2.to_v -> blocks.N.cross_attn.v
   - blocks.N.attn2.to_out.0 -> blocks.N.cross_attn.o
   - blocks.N.attn2.norm_q -> blocks.N.cross_attn.norm_q
   - blocks.N.attn2.norm_k -> blocks.N.cross_attn.norm_k
   - blocks.N.ffn.net.0.proj -> blocks.N.ffn.0
   - blocks.N.ffn.net.2 -> blocks.N.ffn.2
   - blocks.N.scale_shift_table -> blocks.N.modulation
   - norm_out.scale_shift_table -> head.modulation
   - proj_out -> head.head
5. 明确拒绝：
   - patch_embedding.weight if shape is [3072,148,1,2,2]
   - patch_short.*
   - patch_mid.*
   - patch_long.*
   - any LoRA keys
   - any GAN hooks
6. 输出 JSON/Markdown 报告。
```

运行审计：

```bash
cd /home/chw/code/packages/FastWAM

python /home/chw/code/packages/FastWAM/load-giga/code/01_inspect_giga_fastwam.py \
  --fastwam-root /home/chw/code/packages/FastWAM \
  --giga-transformer-dir /mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer \
  --output-json /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.json \
  --output-md /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.md
```

### 预期输出

```text
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.json
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.md
```

报告中必须包含：

```text
Giga transformer config:
  in_channels=148
  out_channels=48
  num_layers=30
  hidden_dim=3072

FastWAM video expert config:
  in_dim=48
  out_dim=48
  num_layers=30
  hidden_dim=3072

direct_transfer:
  keys: 需要列出数量
  params: 需要列出参数量

shape_mismatch:
  patch_embedding.weight: Giga [3072,148,1,2,2] vs FastWAM [3072,48,1,2,2]

semantic_reject:
  patch_short.*
  patch_mid.*
  patch_long.*
```

### 检查点 Checkpoint

```text
Checkpoint 3.1:
[ ] inspect_report.json 存在
[ ] inspect_report.md 存在
[ ] 报告明确显示 Giga in_channels=148
[ ] 报告明确显示 FastWAM in_dim=48
[ ] patch_embedding.weight 被标为 shape_mismatch 或 special_handling
[ ] patch_short/mid/long 被标为 semantic_reject
[ ] 没有使用 strict=False 掩盖迁移失败
```

## Stage 4：转换 Giga Pro 可迁移 video expert 权重

本阶段目标：生成一个只包含 FastWAM video expert 可加载参数的中间权重文件。

### 输入

- 新脚本：
  - `/home/chw/code/packages/FastWAM/load-giga/code/02_convert_giga_pro_video_expert.py`
- 审计报告：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.json`
- Giga Pro transformer：
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer`
- 输出：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt`
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped_report.json`
- GPU：
  - 不需要 GPU。
- 内存：
  - 建议 CPU 内存 64GB 以上，因为 Pro transformer shard 总体约 20GB。

### 操作步骤

创建脚本文件：

```text
/home/chw/code/packages/FastWAM/load-giga/code/02_convert_giga_pro_video_expert.py
```

脚本核心逻辑：

```text
1. 用 safetensors 按 shard 读取 Giga Pro transformer tensor。
2. 按 Stage 3 的映射表改名为 FastWAM video_expert key。
3. 只保存目标 FastWAM 中存在且 shape 完全一致的 tensor。
4. 对 patch_embedding 做保守处理：
   - 默认不迁移 patch_embedding.weight。
   - patch_embedding.bias 如 shape 等于 FastWAM patch_embedding.bias，可迁移。
5. 不迁移 patch_short/mid/long。
6. 输出:
   {
     "video_expert_state_dict": {...},
     "meta": {
       "source": "...Giga-World-1-pro-stage1_final-diffusers/transformer",
       "target": "FastWAM video_expert",
       "direct_load": false,
       "skipped_reason": {...}
     }
   }
```

运行转换：

```bash
cd /home/chw/code/packages/FastWAM

python /home/chw/code/packages/FastWAM/load-giga/code/02_convert_giga_pro_video_expert.py \
  --fastwam-root /home/chw/code/packages/FastWAM \
  --giga-transformer-dir /mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers/transformer \
  --inspect-report /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/inspect_report.json \
  --output-pt /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt \
  --output-report /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped_report.json
```

验证输出：

```bash
python - <<'PY'
import torch
p = "/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt"
payload = torch.load(p, map_location="cpu")
sd = payload["video_expert_state_dict"]
print("mapped_keys:", len(sd))
print("mapped_params:", sum(v.numel() for v in sd.values()))
print("has_patch_weight:", "patch_embedding.weight" in sd)
print("has_patch_bias:", "patch_embedding.bias" in sd)
print("meta:", payload["meta"])
assert "video_expert_state_dict" in payload
assert "patch_short.weight" not in sd
assert "patch_mid.weight" not in sd
assert "patch_long.weight" not in sd
PY
```

### 预期输出

```text
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped_report.json
```

预期报告字段：

```text
mapped_keys: 大量 transformer block key
mapped_params: 应覆盖 video expert 的主体参数
skipped:
  patch_embedding.weight
  patch_short.*
  patch_mid.*
  patch_long.*
shape_mismatch:
  patch_embedding.weight
unexpected: 必须可解释
missing: 必须可解释，不能包含大量 blocks.* 主干权重
```

### 检查点 Checkpoint

```text
Checkpoint 4.1:
[ ] giga_pro_video_expert_mapped.pt 存在
[ ] payload 只有 video_expert_state_dict 和 meta，不包含 optimizer
[ ] patch_embedding.weight 没有被盲目迁移
[ ] patch_short/mid/long 没有被迁移
[ ] blocks.0 到 blocks.29 的 attention/ffn/modulation 主体权重已映射
[ ] 转换报告列出 mapped/missing/unexpected/shape_mismatch
```

## Stage 5：构建 FastWAM + Giga video init checkpoint

本阶段目标：构建一个 FastWAM 可通过 `resume=<pt>` 加载的初始化 checkpoint。这个 checkpoint 包含 MoT 权重，其中 video expert 部分已融合 Giga 可迁移权重，action expert 使用 FastWAM 默认 ActionDiT，proprio encoder 随机初始化。

### 输入

- 新脚本：
  - `/home/chw/code/packages/FastWAM/load-giga/code/03_build_fastwam_giga_init_ckpt.py`
- Giga video expert mapped 权重：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt`
- FastWAM 默认权重：
  - `/mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B`
  - `/mnt/data/chw/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors`
  - `/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- 输出：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt`
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init_report.json`
- GPU：
  - 推荐 1 张 80GB GPU。
  - 如果 CPU 构造可行，也可 CPU，但 Wan2.2 5B 加载较慢。

### 操作步骤

创建脚本文件：

```text
/home/chw/code/packages/FastWAM/load-giga/code/03_build_fastwam_giga_init_ckpt.py
```

脚本核心逻辑：

```text
1. 通过 fastwam.runtime.create_fastwam 构建 FastWAM。
2. 使用默认配置：
   model_id=Wan-AI/Wan2.2-TI2V-5B
   tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B
   video_dit_config 与 configs/model/fastwam.yaml 保持一致
   action_dit_pretrained_path=/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
   proprio_dim=14
   load_text_encoder=false
3. 加载 giga_pro_video_expert_mapped.pt。
4. 只对 model.video_expert.load_state_dict(mapped, strict=False) 做受控加载。
5. 检查 missing/unexpected：
   - missing 允许 patch_embedding.weight 和 FastWAM 特有 buffer。
   - unexpected 必须为 0。
   - blocks 主体 missing 必须为 0。
6. 保存 FastWAM checkpoint:
   {
     "mot": model.mot.state_dict(),
     "proprio_encoder": model.proprio_encoder.state_dict(),
     "step": 0,
     "torch_dtype": "torch.bfloat16",
     "meta": {...}
   }
```

运行构建：

```bash
cd /home/chw/code/packages/FastWAM

export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export PYTHONPATH=/home/chw/code/packages/FastWAM:/home/chw/code/packages/FastWAM/giga-world-1/giga-world-1:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python /home/chw/code/packages/FastWAM/load-giga/code/03_build_fastwam_giga_init_ckpt.py \
  --fastwam-root /home/chw/code/packages/FastWAM \
  --mapped-video-pt /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/giga_pro_video_expert_mapped.pt \
  --action-dit-pt /mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --output-pt /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  --output-report /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init_report.json \
  --device cuda \
  --dtype bf16
```

验证 checkpoint 结构：

```bash
python - <<'PY'
import torch
p = "/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt"
payload = torch.load(p, map_location="cpu")
print(payload.keys())
print("step:", payload.get("step"))
print("mot keys:", len(payload["mot"]))
print("has proprio:", "proprio_encoder" in payload)
print("dtype:", payload.get("torch_dtype"))
assert "mot" in payload
assert "proprio_encoder" in payload
assert "optimizer" not in payload
PY
```

### 预期输出

```text
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init_report.json
```

checkpoint 结构：

```text
mot:
  mixtures.video.*
  mixtures.action.*
proprio_encoder:
  weight: [4096, 14]
  bias: [4096]
step: 0
torch_dtype: torch.bfloat16
```

关键预期：

```text
video expert:
  来自 FastWAM 默认 Wan2.2，再被 Giga Pro 可映射主干覆盖。

action expert:
  来自 ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt。

MoT:
  由 video/action expert 组合，不从 Giga 单独加载。

proprio encoder:
  随机初始化，训练中更新。
```

### 检查点 Checkpoint

```text
Checkpoint 5.1:
[ ] fastwam_giga_pro_video_init.pt 存在
[ ] checkpoint 顶层包含 mot
[ ] checkpoint 顶层包含 proprio_encoder
[ ] checkpoint 顶层不包含 optimizer
[ ] report 中 unexpected_keys 为 0
[ ] report 中 blocks.* 主体 missing_keys 为 0
[ ] action expert 来源是默认 ActionDiT，不是 Giga、LIBERO 或 RoboTwin
```

## Stage 6：AgileX 542 数据与文本缓存准备

本阶段目标：确认 AgileX 542 数据集可读，并准备 text embedding cache，避免训练时加载 T5 text encoder。

### 输入

- 数据集：
  - `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- FastWAM 数据适配：
  - `/home/chw/code/packages/FastWAM/src/fastwam/datasets/lerobot/agilex_dataset.py`
- 文本缓存脚本：
  - `/home/chw/code/packages/FastWAM/scripts/precompute_text_embeds.py`
- 输出：
  - `/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711`
  - `/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box_542_0711/train_stats.json`
- GPU：
  - 预计算 text embeds 建议 1 张 GPU。
  - 只做数据读取检查不需要 GPU。
- 磁盘：
  - text embeds 很小，通常小于 1GB。

### 操作步骤

检查数据基本结构：

```bash
cd /home/chw/code/packages/FastWAM

DATASET=/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711

test -d ${DATASET}/data
test -d ${DATASET}/meta
test -f ${DATASET}/meta/tasks.jsonl
find ${DATASET}/data -name 'episode_*.parquet' | wc -l
head -5 ${DATASET}/meta/tasks.jsonl
```

预计算 text embeds：

```bash
cd /home/chw/code/packages/FastWAM

export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export PYTHONPATH=/home/chw/code/packages/FastWAM:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_text_embeds.py \
  task=agilex_empty_box_uncond_3cam384 \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.context_len=128 \
  data.val.context_len=128 \
  overwrite=true
```

如果需要多卡预计算：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py \
  task=agilex_empty_box_uncond_3cam384 \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.context_len=128 \
  data.val.context_len=128 \
  overwrite=true
```

检查缓存：

```bash
find /mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 -name '*.pt' | wc -l

python - <<'PY'
from pathlib import Path
import torch
cache = Path("/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711")
files = sorted(cache.glob("*.pt"))
print("cache_files:", len(files))
assert files, "no text embedding cache files"
p = files[0]
payload = torch.load(p, map_location="cpu")
print("sample:", p.name)
print("context:", tuple(payload["context"].shape), payload["context"].dtype)
print("mask:", tuple(payload["mask"].shape), payload["mask"].dtype)
assert tuple(payload["context"].shape) == (128, 4096)
assert tuple(payload["mask"].shape) == (128,)
PY
```

### 预期输出

```text
episode parquet 数量: 约 542 个 episode
text cache:
  /mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711/*.pt

单个 cache 文件:
  context: [128, 4096], dtype=torch.bfloat16
  mask: [128], dtype=torch.bool
```

### 检查点 Checkpoint

```text
Checkpoint 6.1:
[ ] DATASET/data 存在
[ ] DATASET/meta/tasks.jsonl 存在
[ ] episode_*.parquet 数量符合预期
[ ] text embedding cache 至少有 1 个 .pt 文件
[ ] cache context shape 是 [128, 4096]
[ ] cache mask shape 是 [128]
```

## Stage 7：1-step 训练、保存和评估 dry run

本阶段目标：用 Giga 初始化后的 FastWAM checkpoint，在 AgileX 542 数据集上跑 1 step，确认 forward、loss、backward、save、eval 全链路可用。

### 输入

- 初始化 checkpoint：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt`
- 数据集：
  - `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- text embeds：
  - `/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711`
- 输出：
  - `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step`
- GPU：
  - 单卡：`CUDA_VISIBLE_DEVICES=0`
  - 推荐显存：80GB。
  - 如果单卡 OOM，用 Stage 8 的多卡 ZeRO-2。

### 操作步骤

单卡 1-step：

```bash
cd /home/chw/code/packages/FastWAM

export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export PYTHONPATH=/home/chw/code/packages/FastWAM:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.val.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  batch_size=1 \
  num_workers=0 \
  max_steps=1 \
  num_epochs=1 \
  save_every=1 \
  eval_every=1 \
  log_every=1 \
  eval_num_inference_steps=4 \
  mixed_precision=bf16 \
  learning_rate=1.0e-5 \
  weight_decay=1.0e-2 \
  wandb.enabled=false
```

如果单卡 OOM，改用 8 卡 ZeRO-2：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step_8gpu \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.val.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  batch_size=1 \
  num_workers=2 \
  max_steps=1 \
  num_epochs=1 \
  save_every=1 \
  eval_every=1 \
  log_every=1 \
  eval_num_inference_steps=4 \
  mixed_precision=bf16 \
  learning_rate=1.0e-5 \
  weight_decay=1.0e-2 \
  wandb.enabled=false
```

验证输出：

```bash
find /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step -maxdepth 4 -type f | sort

python - <<'PY'
import torch
from pathlib import Path
run = Path("/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step")
ckpts = sorted((run / "checkpoints/weights").glob("step_*.pt"))
print("weight_ckpts:", [str(p) for p in ckpts])
assert ckpts, "no weight checkpoint saved"
payload = torch.load(ckpts[-1], map_location="cpu")
print("keys:", payload.keys())
print("step:", payload.get("step"))
print("mot_keys:", len(payload["mot"]))
assert "mot" in payload
assert payload.get("step") == 1
PY
```

### 预期输出

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step/config.yaml
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step/checkpoints/weights/step_000001.pt
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step/checkpoints/state/step_000001/
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_1step/eval/step_000001_rank_000.mp4
```

日志中应出现：

```text
Loading weight checkpoint only: /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
Train/val dataset size: ...
step=1
loss=...
val_loss=...
psnr_rg=...
ssim_rg=...
action_l1=...
action_l2=...
```

### 检查点 Checkpoint

```text
Checkpoint 7.1:
[ ] 训练命令包含 resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
[ ] 训练命令不包含 RoboTwin/LIBERO checkpoint
[ ] 训练日志中有 loss
[ ] step_000001.pt 已保存
[ ] state/step_000001 已保存
[ ] eval mp4 已保存
[ ] checkpoint 可用 torch.load 读取
```

## Stage 8：正式多卡训练

本阶段目标：在 AgileX 542 数据集上正式继续训练 FastWAM，并周期性保存权重、训练状态和评估视频。

### 输入

- 初始化 checkpoint：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt`
- 数据集：
  - `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- text embeds：
  - `/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711`
- 输出：
  - `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full`
- GPU：
  - 推荐 8 卡。
  - 启动方式：`accelerate launch --num_processes 8`。
- 显存：
  - 5B video expert + action expert + MoT 训练建议 8x80GB。
  - 如 OOM，优先降低 batch size、增大 gradient_accumulation_steps、关闭 eval 或降低 eval steps。

### 操作步骤

推荐 8 卡 ZeRO-2 正式训练：

```bash
cd /home/chw/code/packages/FastWAM

export HF_HOME=/mnt/data/chw/fastwam/hf_cache
export MODELSCOPE_CACHE=/mnt/data/chw/fastwam/modelscope_cache
export PYTHONPATH=/home/chw/code/packages/FastWAM:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.val.pretrained_norm_stats=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  model.mot_checkpoint_mixed_attn=false \
  model.compile_training_denoise=false \
  batch_size=1 \
  num_workers=2 \
  gradient_accumulation_steps=2 \
  num_epochs=10 \
  max_steps=null \
  save_every=500 \
  eval_every=500 \
  log_every=10 \
  eval_num_inference_steps=10 \
  mixed_precision=bf16 \
  learning_rate=1.0e-5 \
  weight_decay=1.0e-2 \
  wandb.enabled=false
```

如需从完整训练状态恢复：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/state/step_000500 \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  batch_size=1 \
  num_workers=2 \
  gradient_accumulation_steps=2 \
  num_epochs=10 \
  save_every=500 \
  eval_every=500 \
  log_every=10 \
  mixed_precision=bf16 \
  wandb.enabled=false
```

如只从权重恢复，不恢复 optimizer/scheduler：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights/step_000500.pt \
  output_dir=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full_resume_weights \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  batch_size=1 \
  num_workers=2 \
  gradient_accumulation_steps=2 \
  num_epochs=10 \
  mixed_precision=bf16 \
  wandb.enabled=false
```

### 预期输出

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/config.yaml
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/dataset_stats.json
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights/step_000500.pt
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/state/step_000500/
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/eval/step_000500_rank_000.mp4
```

checkpoint 结构：

```text
weights/*.pt:
  mot
  proprio_encoder
  step
  torch_dtype

state/step_xxxxxx:
  accelerator/deepspeed state
  optimizer state
  scheduler state
  trainer_state.json
```

### 检查点 Checkpoint

```text
Checkpoint 8.1:
[ ] 训练输出全部在 /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full
[ ] 每 save_every 步保存 weights/step_xxxxxx.pt
[ ] 每 save_every 步保存 state/step_xxxxxx
[ ] 每 eval_every 步保存 eval mp4
[ ] trainer_state.json 记录 global_step/epoch/batch_in_epoch
[ ] loss 没有 NaN/Inf
[ ] resume state 能恢复 optimizer/scheduler/global_step
```

## Stage 9：评估、模型选择和验收

本阶段目标：对训练出的 checkpoint 做可重复评估，选择可用于 AgileX empty the box 的最终 FastWAM 模型。

### 输入

- 候选 checkpoint：
  - `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights/step_xxxxxx.pt`
- 数据集：
  - `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- 输出：
  - `/mnt/data/chw/fastwam/evaluate_results/agilex_empty_box_giga_init`
- GPU：
  - 单卡可评估。
  - eval rollout 建议使用 1 张 80GB GPU。

### 操作步骤

先用训练器内置 eval 方式跑短评估。将下面的 `step_XXXXXX.pt` 替换为真实 checkpoint 文件名。

```bash
cd /home/chw/code/packages/FastWAM

CKPT=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights/step_XXXXXX.pt

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=agilex_empty_box_uncond_3cam384 \
  resume=${CKPT} \
  output_dir=/mnt/data/chw/fastwam/evaluate_results/agilex_empty_box_giga_init/eval_step_XXXXXX \
  data.train.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.val.dataset_dirs=[/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711] \
  data.train.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.val.text_embedding_cache_dir=/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711 \
  data.train.use_text_embed_cache=true \
  data.val.use_text_embed_cache=true \
  model.load_text_encoder=false \
  batch_size=1 \
  num_workers=0 \
  max_steps=1 \
  num_epochs=1 \
  save_every=0 \
  eval_every=1 \
  log_every=1 \
  eval_num_inference_steps=10 \
  mixed_precision=bf16 \
  learning_rate=1.0e-8 \
  wandb.enabled=false
```

检查评估视频：

```bash
find /mnt/data/chw/fastwam/evaluate_results/agilex_empty_box_giga_init -name '*.mp4' | sort
```

检查 checkpoint：

```bash
python - <<'PY'
import torch
from pathlib import Path

ckpt = Path("/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights")
files = sorted(ckpt.glob("step_*.pt"))
print("num_ckpts:", len(files))
for p in files[-5:]:
    payload = torch.load(p, map_location="cpu")
    print(p.name, "step=", payload.get("step"), "mot_keys=", len(payload["mot"]), "has_proprio=", "proprio_encoder" in payload)
PY
```

### 预期输出

```text
eval mp4:
  pred rollout | VAE reconstruction | GT video 拼接结果

日志指标:
  val_loss
  psnr_rg
  ssim_rg
  psnr_rd
  ssim_rd
  psnr_dg
  ssim_dg
  action_l1
  action_l2
```

最终模型建议保存路径：

```text
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/final_agilex_empty_box_fastwam.pt
```

选择最终 checkpoint 后复制：

```bash
cp /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_full/checkpoints/weights/step_XXXXXX.pt \
   /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/final_agilex_empty_box_fastwam.pt
```

### 检查点 Checkpoint

```text
Checkpoint 9.1:
[ ] final_agilex_empty_box_fastwam.pt 存在
[ ] checkpoint 包含 mot/proprio_encoder/step/torch_dtype
[ ] eval mp4 可正常播放
[ ] rollout 视频没有明显全黑、全白、静止崩坏
[ ] action_l1/action_l2 被记录
[ ] 能用 final checkpoint 作为 resume 权重重新启动训练
```

## Stage 10：如果直接迁移效果不好，启用 teacher distillation

本阶段目标：当 Stage 7/8 证明直接初始化不稳定或收益不明显时，使用 GigaWorld Pro 作为 teacher 做视频分支蒸馏，而不是继续盲目加载不匹配权重。

### 输入

- Giga Pro teacher：
  - `/mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers`
- FastWAM student：
  - `/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt`
- 新脚本：
  - `/home/chw/code/packages/FastWAM/load-giga/code/04_distill_giga_teacher_to_fastwam.py`
- 输出：
  - `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill`
- GPU：
  - teacher + student 同时加载显存很高。
  - 推荐 8x80GB，或 teacher CPU/offload，或只离线缓存 teacher 预测。

### 操作步骤

优先做离线 teacher 预测缓存，避免训练时同时跑 Giga 和 FastWAM：

```text
/home/chw/code/packages/FastWAM/load-giga/code/04_distill_giga_teacher_to_fastwam.py
```

脚本核心逻辑：

```text
1. 加载 Giga Pro pipeline/transformer 作为 teacher，冻结。
2. 用 AgileX 视频经过 FastWAM/Wan VAE 得到 latent。
3. 对相同 timestep/noise 计算 teacher video noise prediction。
4. 保存 teacher_pred 到 /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill/teacher_cache。
5. FastWAM student 训练时加入:
   loss = FastWAM 原训练 loss + lambda_distill * mse(student_video_pred, teacher_video_pred)
6. action loss 仍由 AgileX action supervision 提供。
```

离线缓存命令模板：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0 python /home/chw/code/packages/FastWAM/load-giga/code/04_distill_giga_teacher_to_fastwam.py \
  --mode cache_teacher \
  --giga-pipeline-dir /mnt/data/chw/giga-world-1/stage1/pro/Giga-World-1-pro-stage1_final-diffusers \
  --dataset-dir /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711 \
  --cache-dir /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill/teacher_cache \
  --num-frames 33 \
  --height 384 \
  --width 320 \
  --dtype bf16
```

蒸馏训练命令模板：

```bash
cd /home/chw/code/packages/FastWAM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  /home/chw/code/packages/FastWAM/load-giga/code/04_distill_giga_teacher_to_fastwam.py \
  --mode train_student \
  --student-init /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt \
  --teacher-cache /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill/teacher_cache \
  --dataset-dir /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711 \
  --output-dir /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill \
  --lambda-distill 0.1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --mixed-precision bf16
```

### 预期输出

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill/teacher_cache/*.pt
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_teacher_distill/checkpoints/weights/step_xxxxxx.pt
```

日志中应同时出现：

```text
loss_video
loss_action
loss_distill
loss_total
```

### 检查点 Checkpoint

```text
Checkpoint 10.1:
[ ] teacher_cache 存在且能读取
[ ] teacher_pred tensor shape 与 student video pred shape 一致
[ ] loss_distill 非 NaN/Inf
[ ] student checkpoint 仍是 FastWAM checkpoint 结构
[ ] 不把 Giga FunControl transformer 直接保存为 FastWAM 模型
```

## 风险清单

1. `patch_embedding.weight` 不匹配是确定风险。
   - Giga: `[3072, 148, 1, 2, 2]`
   - FastWAM: `[3072, 48, 1, 2, 2]`
   - 默认处理：不迁移 weight，只保留 FastWAM 默认 Wan2.2 patch embedding。

2. Giga `guidance_cross_attn=true` 与 FastWAM block 语义不完全相同。
   - 即使 tensor shape 一致，也必须在报告中列出映射理由。
   - 如果 blocks 加载后 loss 爆炸，回退到 teacher distillation。

3. 当前 AgileX task 原配置含 RoboTwin resume。
   - 所有训练命令必须显式 `resume=...fastwam_giga_pro_video_init.pt` 或 `resume=null`。

4. `model.load_text_encoder=false` 与 `use_text_embed_cache=false` 不能同时用于训练。
   - 推荐：先 Stage 6 预计算 cache，再训练时 `use_text_embed_cache=true`。

5. scene LoRA 不是完整模型。
   - 不参与主迁移。
   - 后续如需使用，只能在 Giga teacher 侧做 domain adapter 实验，不写入 FastWAM 初始化主流程。

## 最终验收标准

```text
Final Acceptance:
[ ] /mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/final_agilex_empty_box_fastwam.pt 存在
[ ] final checkpoint 可被 FastWAM load_checkpoint 读取
[ ] final checkpoint 不依赖 LIBERO/RoboTwin checkpoint
[ ] final checkpoint 的 video expert 至少部分来自 Giga Pro 可验证映射
[ ] action expert 来自默认 ActionDiT 并在 AgileX 上继续训练
[ ] proprio encoder 存在，shape 为 weight=[4096,14], bias=[4096]
[ ] AgileX 542 数据集可训练
[ ] 训练可保存 weights checkpoint
[ ] 训练可保存 full state checkpoint
[ ] 能从 weights checkpoint 恢复
[ ] 能从 full state checkpoint 恢复
[ ] eval mp4 可生成
[ ] eval 日志包含 val_loss、PSNR/SSIM、action_l1/action_l2
```

## 当前长跑配置：8 卡 bs8 100k

本轮长跑使用 8 卡、单卡 batch size 8、不做梯度累积：

```text
GPUS=0,1,2,3,4,5,6,7
NUM_PROCESSES=8
batch_size=8
global_batch_size=64
gradient_accumulation_steps=1
max_steps=100000
save_every=5000
eval_every=0
model.load_text_encoder=false
data.train.use_text_embed_cache=true
data.val.use_text_embed_cache=true
```

一键启动：

```bash
cd /home/chw/code/packages/FastWAM
bash load-giga/code/06_start_bs8_8gpu_training_tmux.sh
```

tmux session：

```text
fastwam_bs8_8gpu_100k
```

训练输出：

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k
```

训练日志：

```text
/mnt/data/chw/fastwam/logs/giga_to_fastwam/agilex_empty_box_giga_init_bs8_8gpu_100k.log
```

实时曲线：

```text
/home/chw/code/packages/FastWAM/load-giga/code/live/loss_curve.gif
/home/chw/code/packages/FastWAM/load-giga/code/live/loss_curve.png
/home/chw/code/packages/FastWAM/load-giga/code/live/index.html
```
