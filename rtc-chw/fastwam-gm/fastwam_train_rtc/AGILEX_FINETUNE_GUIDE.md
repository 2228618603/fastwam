# FastWAM 训练自有数据（Agilex 双臂）完整指南

> 目标数据：`data/agilex_empty_the_box_all_470`
> 代码版本：FastWAM @ `7faa711`（Optimize IDM action-only inference）
> 硬件：8 × NVIDIA A800-SXM4-80GB / 80 CPU / 1.4T RAM / 剩余磁盘 1.4T
> 文档生成日期：2026-08-26

---

## 目录

- [0. 结论速览](#0-结论速览)
- [1. 数据现状盘点](#1-数据现状盘点)
- [2. README 提供了什么 / 没提供什么](#2-readme-提供了什么--没提供什么)
- [3. 环境安装](#3-环境安装)
- [4. 数据适配（核心工作）](#4-数据适配核心工作)
- [5. 需要下载 / 生成的 checkpoint](#5-需要下载--生成的-checkpoint)
- [6. 需要新增的配置文件](#6-需要新增的配置文件)
- [7. 启动训练](#7-启动训练)
- [8. 注意事项与调参要点](#8-注意事项与调参要点)
- [9. 执行清单](#9-执行清单)
- [附录 A：数据转换脚本](#附录-a数据转换脚本)
- [附录 B：关键代码位置索引](#附录-b关键代码位置索引)
- [附录 C：RoboTwin 权重热启动的可迁移性评估与 A/B 验证流程](#附录-crobotwin-权重热启动的可迁移性评估与-ab-验证流程)

---

## 0. 结论速览

你的数据形态（3 相机 / 480×640 / 双臂 14 维绝对关节 action）和 FastWAM 的 **RoboTwin** 配置几乎完全一致，因此：

- **模板用 `configs/data/robotwin.yaml`，不要用 LIBERO 的**（LIBERO 是 2 相机 / 7 维 delta-eef action）。
- 官方 `robotwin_uncond_3cam_384.pt` 权重可以**直接热启动**（维度全对得上）。

但有一个 README 完全没提、且**必然导致启动失败**的问题：

> **你的数据字段名不符合 FastWAM 的 key 推导规则，必须先做一次字段改名。**

这是本次适配的主要工作量，详见 [第 4 节](#4-数据适配核心工作)。

> **⚠️ 别把两个转换搞混**：全文提到两个数据转换 —— ①**字段改名**（必做，几分钟，零磁盘）
> 和 ②**LeRobot v2.1 → v3.0**（可选的性能优化，数小时 + 28GB）。两者正交，做了 ② 之后
> ① 照样要做。区别与执行顺序见 [4.0](#40-先搞清楚本文档涉及两个不同的转换)。

---

## 1. 数据现状盘点

来自 `meta/info.json`、`meta/episodes.jsonl`、parquet schema 与 `ffprobe`：

| 项目 | 值 |
|---|---|
| LeRobot 版本 | **v2.1** |
| robot_type | `Agilex_Cobot_Magic` |
| episodes | 471（`splits.train = 0:471`） |
| 总帧数 | 1,003,984（平均 2131 帧/episode ≈ **71 秒**） |
| fps | 30 |
| task 数 | 1（"Alternately use the left arm and the right arm to pick up the goods from the nearby box and place them in the distant box, until the nearby box is empty."） |
| 相机 | 3 路，均 480×640，**codec = AV1**，yuv420p |
| chunks | 1（`chunk-000`，`chunks_size=1000`） |
| parquet 体积 | 169 MB |
| 视频体积 | 28 GB |

### 字段清单

| 字段 | dtype | 维度 | 说明 |
|---|---|---|---|
| `observation.image.top` | video | 480×640×3 | 顶部相机 |
| `observation.image.left_wrist` | video | 480×640×3 | 左腕相机 |
| `observation.image.right_wrist` | video | 480×640×3 | 右腕相机 |
| `observation.state.joint` | float32 | 12 | 左 6 关节 + 右 6 关节 |
| `observation.gripper_position` | float32 | 2 | 左/右夹爪**百分比** |
| `observation.state.end` | float32 | 12 | 双臂末端位姿（本次不用） |
| `actions` | float32 | 14 | `[左6关节, 左夹爪宽度, 右6关节, 右夹爪宽度]` |

`meta/modality.json`、`meta/relative_stats.json` 是 GR00T / Isaac 流水线的产物，FastWAM 不读取，保留无害。

---

## 2. README 提供了什么 / 没提供什么

### 提供了

| README 章节 | 内容 |
|---|---|
| Environment Setup | conda py3.10 + torch 2.7.1+cu128 + `pip install -e .` |
| Model Preparation | 设 `DIFFSYNTH_MODEL_BASE_PATH`；用 `preprocess_action_dit_backbone.py` 生成 ActionDiT backbone |
| Dataset Download | 仅 LIBERO / RoboTwin 官方数据的下载与解压 |
| Inference with Released Checkpoints | HF 上 6 个发布权重 + dataset_stats 的下载方式 |
| Training | ① `precompute_text_embeds.py` 预计算 T5 缓存 → ② `train_zero1.sh` 启动；**首次运行把 `pretrained_norm_stats` 设为 `null`**，跑完后改为生成的 `dataset_stats.json` |
| What's New | LeRobot 2.1/3.0 双支持；Optional IDM（一模型两推理模式）；`compile_training_denoise` 训练加速 ~10%；推理加速 ~2×；action scheduler shift 默认改为 1.0（评测旧权重需 `EVALUATION.sigma_shift=5.0`） |
| 自定义数据 | **仅一句话**："复制 `configs/data/libero_2cam_lerobot_v30.yaml`，改 `train.dataset_dirs`，用 `data=<config_name>` 选中" |

### 没提供（需自己解决）

1. **数据字段命名规范** —— 硬性要求，不满足直接 crash（[第 4 节](#4-数据适配核心工作)）。
2. **非 LIBERO/RoboTwin 数据没有评测入口** —— `experiments/` 下只有两个依赖仿真器的 manager（[8.6](#86-没有仿真评测入口)）。
3. **v2.1 与 v3.0 loader 的视频解码开销差异** —— 相差 3.7 倍（[8.2](#82-av1--v21-loader--每样本解码-99-帧最大性能陷阱)）。
4. **数据集长度按帧计** —— 你这份数据 1 epoch ≈ 7844 步（bs16×8卡），照抄 RoboTwin 的 `num_epochs: 5` 会跑到几万步（[8.1](#81-数据量1-epoch--7844-步必须设-max_steps)）。

---

## 3. 环境安装

当前 `base` 环境无 torch，需新建独立环境：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
cd /home/gaomeng/FastWAM && pip install -e .
```

### 注意

- `pyproject.toml` 中依赖**全部钉死版本**（`accelerate==1.12.0`、`deepspeed==0.18.7`、`transformers==4.49.0`、`datasets==4.8.5`、`torchcodec==0.4.0`、`pyarrow==24.0.0` 等），不要自行升级。
- **不需要**安装 LIBERO / mujoco / RoboTwin / SAPIEN —— 那些仅用于仿真评测，真机数据训练不涉及。
- 系统 `ffmpeg` 已带 `libdav1d`（AV1 解码器），满足你的视频格式要求。
- 环境变量 `HF_ENDPOINT=https://hf-mirror.com` 已设置，HuggingFace 下载会走镜像。

---

## 4. 数据适配（核心工作）

### 4.0 先搞清楚：本文档涉及两个不同的转换

全文一共提到两个数据转换，**性质完全不同，且不是二选一**：

| | ① 字段改名 | ② LeRobot v2.1 → v3.0 |
|---|---|---|
| **性质** | 改**字段名**，存储格式仍是 v2.1 | 改**存储格式**（分块 parquet / 分块视频） |
| **必要性** | **必做**，不改直接跑不起来 | **可选**，纯性能优化 |
| **解决什么问题** | key 推导规则不匹配 → crash | 每样本解码 99 帧 → 27 帧（见 [8.2](#82-av1--v21-loader--每样本解码-99-帧最大性能陷阱)） |
| **动到什么** | parquet 列名 + `info.json` + 视频目录名 | 整个目录布局；meta 从 jsonl 变 parquet |
| **代价** | 几分钟，零额外磁盘（视频软链接） | 数小时 + 约 28 GB 新磁盘 |
| **本文位置** | 本节 4.1–4.3 + [附录 A](#附录-a数据转换脚本) | [8.2](#82-av1--v21-loader--每样本解码-99-帧最大性能陷阱) |

#### 关键点：做了 ② 之后，① 照样要做

`src/fastwam/datasets/lerobot3/base_lerobot_dataset.py` 全文只有 12 行：

```python
class BaseLerobotDataset(_BaseLerobotDataset):   # 继承 v2.1 的基类
    metadata_cls = LeRobotDatasetMetadata
    multi_dataset_cls = MultiLeRobotDataset
    presample_images = True                      # 唯一的行为差异
```

`__init__`（包含下面 4.1 讲的 key 拼接规则）**完全继承自 v2.1 基类，一行没重写**。
所以字段命名要求对 v2.1 和 v3.0 两个 loader 是**同一套**。

#### 因此执行顺序是：先改名，再（可选）转 v3.0

```
原始数据 (v2.1, 字段名不合规)
    │
    ├─ ① 字段改名  ← 必做（附录 A 脚本）
    ↓
v2.1 + 正确字段名   ←── 到这里就已经可以开始训练了
    │
    ├─ ② v21 → v30 转换  ← 可选；实测确认 dataloader 是瓶颈后再做
    ↓
v3.0 + 正确字段名
```

反过来（先转 v3.0 再改名）会麻烦得多：v3.0 的 meta 是 `tasks.parquet` +
`episodes/*.parquet`，改名比改 jsonl 复杂；而且上游转换器可能对 `actions`、
`observation.gripper_position` 这类非标准命名做校验，先规范化更稳。

### 4.1 FastWAM 的 key 推导规则

`src/fastwam/datasets/lerobot/base_lerobot_dataset.py:79-96` 从配置 `shape_meta` 里的 key **拼接**出 LeRobot 字段名，**不接受任意命名**：

```python
# images
meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
# state
meta["lerobot_key"] = f"observation.state.{key}"  if key != "default" else "observation.state"
# action
meta["lerobot_key"] = f"action.{key}"             if key != "default" else "action"
```

拼出来的名字会被用于三个地方，所以必须与磁盘上的实际命名严格一致：
1. `delta_timestamps` 的 key → 查询 parquet 列；
2. `info.json` 的 `features` 键 → 判定是否为 video 字段；
3. `video_path` 模板中的 `{video_key}` → **视频子目录名**。

### 4.2 你的数据 vs 要求

| 现有字段 | 维度 | 合规 | 需改为 |
|---|---|:--:|---|
| `observation.image.top` | 480×640 | ❌ `image` 应为 `images` | `observation.images.cam_high` |
| `observation.image.left_wrist` | 480×640 | ❌ | `observation.images.cam_left_wrist` |
| `observation.image.right_wrist` | 480×640 | ❌ | `observation.images.cam_right_wrist` |
| `observation.state.joint` | 12 | ✅ 天然匹配 `key: joint` | 不动 |
| `observation.gripper_position` | 2 | ❌ 规则无法拼出此名 | `observation.state.gripper_position` |
| `observation.state.end` | 12 | ✅（本次不用） | 不动 |
| `actions` | 14 | ❌ 多一个 `s` | `action` |

> 相机名沿用 RoboTwin 的 `cam_high` / `cam_left_wrist` / `cam_right_wrist`，好处是可以直接复用 RoboTwin 配置，也便于对照官方权重。

### 4.3 转换方案

视频占 28 GB、parquet 仅 169 MB，因此：**parquet 重写 + meta 改写 + 视频目录软链接**（零额外磁盘、原数据完全不动）。

目标结构：

```
data/agilex_empty_the_box_fastwam/
├── meta/
│   ├── info.json            # features 中 4 个 key 改名
│   ├── tasks.jsonl          # 原样复制
│   ├── episodes.jsonl       # 原样复制
│   └── episodes_stats.jsonl # stats 键改名（可选，见下）
├── data/chunk-000/
│   └── episode_0000*.parquet   # 471 个，重写列名
└── videos/chunk-000/
    ├── observation.images.cam_high        -> symlink → 原 observation.image.top/
    ├── observation.images.cam_left_wrist  -> symlink → 原 observation.image.left_wrist/
    └── observation.images.cam_right_wrist -> symlink → 原 observation.image.right_wrist/
```

需要改动的三处：

1. **parquet 列名**（471 个文件）：`actions` → `action`，`observation.gripper_position` → `observation.state.gripper_position`。用 `pyarrow` 的 `table.rename_columns()` 重写，总量 169 MB，几分钟完成。
2. **`meta/info.json`** 的 `features` 字典键（4 个）。
3. **`meta/episodes_stats.jsonl`** 的 `stats` 键。
   > 这一步**技术上可选**：`aggregate_stats()`（`lerobot/datasets/compute_stats.py:158`）只对 key 求并集、不校验 features；而 FastWAM 的归一化用的是它自己生成的 `dataset_stats.json`，不用 LeRobot 的 stats。但改了更一致，建议一并处理。

软链接可行性已确认：`LeRobotDataset.__init__` 用 `(self.root / fpath).is_file()` 校验，`is_file()` 会跟随软链接。

**可直接使用的转换脚本见 [附录 A](#附录-a数据转换脚本)。**

---

## 5. 需要下载 / 生成的 checkpoint

### 5.0 先搞清楚：一共有三层权重，别混淆

```
┌─ 层 1: Wan2.2 原始权重（通用视频生成底座，不懂机器人）   ← 必需，自动下载
│   ├── DiT (5B)  ──────────┐
│   ├── VAE                 │
│   ├── T5 text encoder     │  线性插值降维
│   └── umt5 tokenizer      │  hidden 3072→1024, ffn 14336→4096
│                           │
├─ 层 2: ActionDiT backbone ┘                          ← 必需，本地生成（非下载）
│
└─ 层 3: robotwam 训练好的 .pt（如 robotwin_uncond_3cam_384.pt）  ← 可选，热启动
```

| | 层 1：Wan 权重 | 层 3：热启动权重 |
|---|---|---|
| **是什么** | 通用视频生成**底座**，海量普通视频预训练 | FastWAM 在**机器人数据**上训练好的成果 |
| **懂机器人吗** | 完全不懂，没见过 action | 懂，已学会 RoboTwin 双臂操作 |
| **必要性** | **必需**，没它模型建不起来 | **可选**，只是让收敛更快 |
| **怎么获得** | 从 ModelScope / HF 自动下载（5.1） | 手动 `huggingface-cli download`（5.3） |
| **包含什么** | DiT + VAE + T5 + tokenizer | 只有 `mot` + `proprio_encoder` |
| **训练时角色** | `video_expert` 初始权重 + **VAE 全程编码视频** | 覆盖 `mot`/`proprio_encoder` 后即退场 |

#### 关键：热启动权重**不能替代** Wan 权重下载

两个硬性原因：

**a) `.pt` 里没有 VAE，而训练时 VAE 是必需的。**
`fastwam.py:1199-1209` 保存时只存三样：

```python
payload = {
    "mot": self.mot.state_dict(),          # = video_expert + action_expert
    "step": step,
    "torch_dtype": str(self.torch_dtype),
}
if self.proprio_encoder is not None:
    payload["proprio_encoder"] = self.proprio_encoder.state_dict()
```

没有 VAE、没有 T5、没有 tokenizer。而训练时必须用 VAE 把视频编成 latent
（`_encode_video_latents`，`fastwam.py:248`）。

**b) 加载顺序上绕不过去。**
`runtime.py` → `FastWAM.from_wan22_pretrained()` 先用 Wan 权重把网络**建出来**；
`trainer.py:315` 的 `_resume_or_load_checkpoint()` 才在**之后**用 `.pt` 覆盖 `mot`。
没有第一步就没有第二步。

#### 层 2 的 ActionDiT backbone 是从层 1 派生的，不是下载的

`preprocess_action_dit_backbone.py:199-208` 把 Wan2.2 DiT 的权重**线性插值降维**到
ActionDiT 的尺寸（video expert hidden 3072 / ffn 14336 → action expert hidden 1024 /
ffn 4096），并做 `alpha = sqrt(d_video / d_action)` 缩放：

```python
value = _resize_tensor_to_shape(src, tuple(target.shape))          # 3072 -> 1024
if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
    alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
    value = value.to(torch.float32) * alpha
```

这解释了两件事：① 必须先能下载 Wan 权重才能生成它；② 换模型变体
（`fastwam_optional_idm.yaml` 等 action_dit_config 不同）必须**重新生成**一份。

#### 各阶段实际需要哪些组件

| 阶段 | Wan DiT | VAE | T5 + tokenizer | ActionDiT backbone | 热启动 `.pt` |
|---|:--:|:--:|:--:|:--:|:--:|
| `preprocess_action_dit_backbone.py` | ✅ 读它插值 | 加载但不用 | 加载但不用 | **生成它** | — |
| `precompute_text_embeds.py` | ❌ | ❌ | ✅ 只加载这两个 | ❌ | ❌ |
| 训练 | ✅ 作初始权重 | ✅ 全程编码视频 | ❌ 用缓存 | ✅ | 可选 |

> `precompute_text_embeds.py:228-245` 只 resolve 并加载 `text_config` + `tokenizer_config`，
> 不碰 DiT / VAE，所以这一步很轻。
> 训练时不加载 T5 是因为 `configs/model/fastwam.yaml:5` 设了 `load_text_encoder: false`
> —— 文本 embedding 已被 precompute 缓存，这也是 precompute 成为训练前必需步骤的原因。

#### 模型结构与权重的对应关系

```
FastWAM
├── mot  (MoT = Mixture of Transformers)   ← 唯一参与训练的部分，也是 .pt 里存的
│   ├── mixtures["video"]  = video_expert  ← 初始权重来自 Wan2.2 DiT（层 1）
│   └── mixtures["action"] = action_expert ← 初始权重来自 ActionDiT backbone（层 2）
├── proprio_encoder = nn.Linear(14, 4096)  ← 参与训练，也存在 .pt 里
├── vae          (Wan2.2 VAE)              ← 冻结；训练全程用于编码视频；不在 .pt 里
├── text_encoder (T5)                      ← 训练时不加载；不在 .pt 里
└── tokenizer                              ← 训练时不加载；不在 .pt 里
```

可训练参数仅 `mot.dit` + `proprio_encoder`（`trainer.py:95-101`），VAE / T5 全程冻结。

### 5.1 自动下载（首次运行时触发）

由 `src/fastwam/models/wan22/helpers/loader.py:125-138` 与 `helpers/io.py` 驱动，**默认下载源是 ModelScope**：

| 组件 | 来源 repo | 文件 pattern |
|---|---|---|
| Wan2.2 DiT | `Wan-AI/Wan2.2-TI2V-5B` | `diffusion_pytorch_model*.safetensors` |
| T5 text encoder | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `models_t5_umt5-xxl-enc-bf16.safetensors` |
| Wan2.2 VAE | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `Wan2.2_VAE.safetensors` |
| umt5 tokenizer | `Wan-AI/Wan2.1-T2V-1.3B` | `google/umt5-xxl/` |

> T5 与 VAE 的重定向由 `configs/model/fastwam.yaml` 的 `redirect_common_files: true` 控制（把 `.pth` 换成社区转换好的 `.safetensors`）。

```bash
mkdir -p /home/gaomeng/FastWAM/checkpoints
export DIFFSYNTH_MODEL_BASE_PATH=/home/gaomeng/FastWAM/checkpoints

# 默认走 modelscope；如需切到 HuggingFace（会走你已设的 hf-mirror）：
# export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
```

相关环境变量（`helpers/io.py:33-51`）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `DIFFSYNTH_MODEL_BASE_PATH` | 权重根目录 | `./checkpoints/` |
| `DIFFSYNTH_DOWNLOAD_SOURCE` | `modelscope` 或 `huggingface` | `modelscope` |
| `DIFFSYNTH_SKIP_DOWNLOAD` | `true` 则不联网下载 | `false` |

磁盘预留 **≥ 60 GB**。

### 5.2 必须自己生成：ActionDiT backbone

这是[层 2](#50-先搞清楚一共有三层权重别混淆)。**不是下载的**，是从层 1 的 Wan DiT 插值降维得来，所以必须先能下载到 Wan 权重。

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

输出路径写死在 `configs/model/fastwam.yaml:11`（`action_dit_pretrained_path`），不改配置就用这个文件名。

> 如果改用 Optional IDM / IDM / Joint 变体，需用对应的 `configs/model/fastwam_optional_idm.yaml` 等**重新生成一份** backbone。

### 5.3 强烈建议：用 RoboTwin 权重热启动

这是[层 3](#50-先搞清楚一共有三层权重别混淆)。**它不能替代 5.1 的 Wan 下载**（原因见 5.0），只是在 Wan + ActionDiT 初始化完成之后，把 `mot` / `proprio_encoder` 覆盖成已经学会机器人操作的权重。

`trainer.py:315-327` 的 `_resume_or_load_checkpoint()`：`resume` 指向 `.pt` **文件**时走 "Loading weight checkpoint only"（不恢复 optimizer/step）；指向**目录**时才恢复完整训练状态。

`fastwam.py:1211` 的 `load_checkpoint()`：`mot` 用 `strict=False`，`proprio_encoder` 用 `strict=True`。

你的 `action_dim=14` / `proprio_dim=14` / 视频尺寸 `384×320` **与 RoboTwin 完全一致**，所以官方权重可以直接热启动。471 条数据从头训一个 5B 模型收敛会很吃力，建议务必热启动：

```bash
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  --local-dir ./checkpoints/fastwam_release
```

训练时加 `resume=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`。

> `robotwin_uncond_3cam_384_dataset_stats.json` **不要**用于你的数据 —— 那是 RoboTwin 的归一化统计量，与你的关节范围无关。归一化统计量必须由你的数据自己算（见 [7.2](#72-归一化统计量的两轮流程)）。

---

## 6. 需要新增的配置文件

### 6.1 `configs/data/agilex_3cam.yaml`

基于 `configs/data/robotwin.yaml` 复制。注意 **`train:` 和 `val:` 两段都要改**，除了 `is_training_set` 外内容完全相同。

```yaml
train:
  _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - ./data/agilex_empty_the_box_fastwam
  shape_meta:
    images:                          # 顺序必须是 顶部 → 左腕 → 右腕，见下方说明
      - key: cam_high
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_left_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_right_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
    action:
      - key: default                 # -> "action"
        raw_shape: 14
        shape: 14
    state:                           # 拼接顺序 = 此处顺序 -> 12 + 2 = 14
      - key: joint                   # -> "observation.state.joint"
        raw_shape: 12
        shape: 12
      - key: gripper_position        # -> "observation.state.gripper_position"
        raw_shape: 2
        shape: 2
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4         # 32 action, 9 video frames
  video_size: [384, 320]             # [H, W]，robotwin 拼图后的尺寸
  camera_key: null
  val_set_proportion: 0.01           # ~5 个 episode 留作验证
  is_training_set: true
  pretrained_norm_stats: null        # 首次运行用 null，见 7.2
  skip_padding_as_possible: false
  concat_multi_camera: "robotwin"
  processor:
    _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 3
    action_output_dim: 14
    proprio_output_dim: 14

    action_state_transforms: null
    use_stepwise_action_norm: False
    norm_default_mode: "z-score"
    norm_exception_mode: null

    action_state_merger:
      _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign

    train_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
    val_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
  text_embedding_cache_dir: ./data/text_embeds_cache/agilex
  context_len: 128

val:
  _target_: fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - ./data/agilex_empty_the_box_fastwam
  shape_meta:
    images:
      - key: cam_high
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_left_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_right_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
    action:
      - key: default
        raw_shape: 14
        shape: 14
    state:
      - key: joint
        raw_shape: 12
        shape: 12
      - key: gripper_position
        raw_shape: 2
        shape: 2
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4
  video_size: [384, 320]
  camera_key: null
  val_set_proportion: 0.01
  is_training_set: false            # <- 唯一与 train 不同之处
  pretrained_norm_stats: null
  skip_padding_as_possible: false
  concat_multi_camera: "robotwin"
  processor:
    _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 3
    action_output_dim: 14
    proprio_output_dim: 14
    action_state_transforms: null
    use_stepwise_action_norm: False
    norm_default_mode: "z-score"
    norm_exception_mode: null
    action_state_merger:
      _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
    train_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
    val_transforms:
      - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
  text_embedding_cache_dir: ./data/text_embeds_cache/agilex
  context_len: 128
```

#### 三个必须注意的点

1. **相机顺序不能错。** `concat_multi_camera: "robotwin"`（`robot_video_dataset.py:168-192`）硬编码了拼图逻辑：
   - `video[0]` → resize 到 `256×320`，作为上半屏
   - `video[1]` / `video[2]` → 各 resize 到 `128×160`，左右并排作为下半屏
   - 竖向拼接得到 `384×320`

   `video[i]` 的顺序就是 `shape_meta.images` 的书写顺序。顺序错了画面布局就错。

2. **不要加 `delta_action_dim_mask`。** 那是 LIBERO 专用（其 action 是 delta-eef，掩码标记哪些维度是增量）。你的 action 是绝对关节位置，与 RoboTwin 一致，保持不设置。

3. **`raw_shape` 与实际一致。** processor 会断言变换后的 shape 等于 `[num_image_steps] + shape`（`fastwam_processor.py:229-231`）。480×640 经 `Resize [240,320]` 得 240×320，与 `shape: [3, 240, 320]` 匹配。

### 6.2 `configs/task/agilex_uncond_3cam_384_1e-4.yaml`

```yaml
# @package _global_

defaults:
  - override /data: agilex_3cam
  - override /model: fastwam
  - _self_

# dataloading
batch_size: 8            # A800 80G 先保守起步，见 8.3
num_workers: 10          # 80 CPU / 8 GPU

model:
  mot_checkpoint_mixed_attn: false   # 必须 false：设 true 会被 mot.py:28-31 抛错（不支持梯度检查点）

# scheduler
lr_scheduler_type: "cosine"
learning_rate: 1e-4
num_epochs: 5
max_steps: 20000         # 重要！见 8.1
log_every: 10
save_every: 2000
eval_every: 1000

# training
gradient_accumulation_steps: 1
weight_decay: 1e-2
resume: null
```

> `max_steps` 会覆盖 `num_epochs` 推算的步数（`trainer.py:254-256`），并同时决定 cosine 调度的总长和 5% warmup。设了 `max_steps` 后 `num_epochs` 只起兜底作用。

---

## 7. 启动训练

### 7.1 完整流程

```bash
conda activate fastwam
cd /home/gaomeng/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints

# ---- Step 0: 数据转换（见附录 A）----
python scripts/convert_agilex_to_fastwam.py \
  --src ./data/agilex_empty_the_box_all_470 \
  --dst ./data/agilex_empty_the_box_fastwam

# ---- Step 1: 生成 ActionDiT backbone（首次会自动下载 Wan2.2 全套权重）----
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# ---- Step 2: 预计算 T5 文本缓存（你只有 1 个 task，秒级完成，单卡足够）----
python scripts/precompute_text_embeds.py task=agilex_uncond_3cam_384_1e-4

# ---- Step 3: 小规模试跑，验证 pipeline 通了 ----
bash scripts/train_zero1.sh 8 task=agilex_uncond_3cam_384_1e-4 \
  max_steps=50 batch_size=4 eval_every=40 save_every=50

# ---- Step 4: 正式训练（热启动 RoboTwin 权重）----
bash scripts/train_zero1.sh 8 task=agilex_uncond_3cam_384_1e-4 \
  resume=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

`precompute_text_embeds.py` 只读 `meta/tasks.jsonl`（`scripts/precompute_text_embeds.py:114-145`），不实例化 dataset、不需要 stats，所以它可以在数据转换之后、任何时候单独跑。多卡版本：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=agilex_uncond_3cam_384_1e-4
```

### 7.2 归一化统计量的两轮流程

- **第一次运行**：`pretrained_norm_stats: null` → 遍历 471 个 episode 计算统计量，写到 `runs/<task>/<run_id>/dataset_stats.json`（`robot_video_dataset.py:99-118`）。
- **之后**：把 `configs/data/agilex_3cam.yaml` 里 **train 和 val 两段**的 `pretrained_norm_stats` 都改成那个路径，省掉每次重算的开销。

> val 段写 `null` 不会报错：`runtime.py:430-441` 的 `build_datasets()` 会让 val 回落到 `train.pretrained_norm_stats` 或 `{output_dir}/dataset_stats.json`（train 刚写出的那份）。

### 7.3 输出目录结构

```
runs/agilex_uncond_3cam_384_1e-4/<YYYY-MM-DD_HH-MM-SS>/
├── config.yaml                    # 本次运行的完整解析后配置
├── dataset_stats.json             # 归一化统计量（下次运行复用）
├── checkpoints/
│   ├── weights/step_XXXXXX.pt     # 纯权重，用于 resume=<file> 或推理
│   └── state/step_XXXXXX/         # accelerate 完整状态，用于 resume=<dir> 断点续训
└── eval/                          # 验证期生成的可视化视频
```

`train_zero1.sh` 自动设置 `output_dir=./runs/${TASK_BASENAME}/${RUN_ID}` 和 `wandb.name=${TASK_BASENAME}`。

### 7.4 断点续训 vs 热启动

| 目的 | 写法 |
|---|---|
| 热启动（只加载权重，step 从 0 开始） | `resume=/path/to/xxx.pt` |
| 断点续训（恢复 optimizer/scheduler/step/dataloader 进度） | `resume=/path/to/checkpoints/state/step_XXXXXX` |

### 7.5 开启 wandb（可选）

```bash
bash scripts/train_zero1.sh 8 task=agilex_uncond_3cam_384_1e-4 \
  wandb.enabled=true wandb.project=fastwam-agilex
```

---

## 8. 注意事项与调参要点

按重要性排序。

### 8.1 数据量：1 epoch ≈ 7844 步，必须设 `max_steps`

dataset 的 `__len__` 返回的是**帧数**（`base_lerobot_dataset.py:186-187` → `multi_dataset.num_frames`），不是 episode 数。窗口逐帧滑动、高度重叠。

| 配置 | 每步样本数 | 1 epoch 步数 |
|---|---|---|
| 8 卡 × bs 8 | 64 | ~15,700 |
| 8 卡 × bs 16 | 128 | ~7,844 |

RoboTwin 那份配置的 `num_epochs: 5` 放到你这儿是**几万步**的量级（官方 RoboTwin 用了 64 卡）。**先设 `max_steps=20000` 之类的上限**，看 loss 曲线和 val 指标再决定是否延长。

### 8.2 AV1 + v2.1 loader = 每样本解码 99 帧（最大性能陷阱）

两个 loader 的关键差异：

| loader | `presample_images` | 每样本解码帧数 |
|---|---|---|
| `fastwam.datasets.lerobot`（v2.1，**你的数据**） | `False` | 33 × 3 相机 = **99** |
| `fastwam.datasets.lerobot3`（v3.0） | `True` | 9 × 3 相机 = **27** |

v2.1 路径会把 33 帧**全部**解码、送进 processor，**之后**才按 `stride=4` 抽成 9 帧（`robot_video_dataset.py:156-165`）。等于 **3.7 倍无用解码**，而且你的视频是 AV1（随机 seek 成本显著高于 H.264）。80 CPU / 8 GPU 只有约 10 workers/GPU，很可能被 dataloader 卡住。

三个选项，按性价比排序：

1. **先测量**：按 v2.1 跑，看 `log_every=10` 的 step 时间，确认是否真是瓶颈（对比 `nvidia-smi` 的 GPU 利用率）。
2. **推荐：转 LeRobot v3.0**，改用 `_target_: fastwam.datasets.lerobot3.robot_video_dataset.RobotVideoDataset`（照抄 `configs/data/libero_2cam_lerobot_v30.yaml` 的写法，其余字段不变）。README 也明确推荐 v3.0 用于大数据集。你已有 `lerobot-convert` conda 环境，可用上游 LeRobot 的 `convert_dataset_v21_to_v30.py`。
3. **可叠加：AV1 转 H.264** 并缩小关键帧间隔（`ffmpeg -c:v libx264 -g 30 -crf 20`），随机访问解码会快很多。代价是几小时 CPU 时间和一份新的 28 GB 数据。

#### 转 v3.0 的三个前置条件（务必先读 [4.0](#40-先搞清楚本文档涉及两个不同的转换)）

**a) 字段改名仍然必须先做。** v3.0 loader 只是个 12 行子类，key 拼接规则完全继承自
v2.1 基类。顺序：先做附录 A 的改名，再转 v3.0。

**b) `precompute_text_embeds.py` 会在 v3.0 数据集上报错。** 该脚本硬编码读
`meta/tasks.jsonl`（`scripts/precompute_text_embeds.py:120`，无 fallback）：

```python
tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
if not tasks_path.exists():
    raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
```

但 v3.0 的 metadata 用的是 `meta/tasks.parquet`（`lerobot3/lerobot_dataset.py:74`）。

绕过办法（任选其一）：
- **推荐**：文本缓存文件名是 prompt 的 SHA256（`{hash}.t5_len128.wan22ti2v5b.pt`，
  见 `robot_video_dataset.py:254-255`），**与数据集版本无关**。只要 v2.1 config 和
  v3.0 config 的 `text_embedding_cache_dir` 指向**同一个目录**，在改名后的 v2.1
  数据集上跑一次 precompute，v3.0 训练即可直接复用。
- 或者在 v3.0 数据集目录里额外保留一份 `meta/tasks.jsonl`。

**c) `info.json` 的 `codebase_version` 必须正好是 `"3.0"`。**
`lerobot3/lerobot_dataset.py:71-73` 做严格相等判断，不等就抛
`ValueError: Expected LeRobot codebase_version v3.0, got ...`。

#### v3.0 与 v2.1 的目录差异（改配置时对照用）

| | v2.1（你现在的数据） | v3.0 |
|---|---|---|
| task 列表 | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| episode 列表 | `meta/episodes.jsonl` | `meta/episodes/*/*.parquet` |
| episode 统计量 | `meta/episodes_stats.jsonl` | 并入 `meta/episodes/` 的 `stats/*` 列 |
| 帧数据 | `data/chunk-000/episode_XXXXXX.parquet`（1 episode / 文件） | `data/*/*.parquet`（多 episode / 文件） |
| 视频 | `videos/chunk-000/{video_key}/episode_XXXXXX.mp4` | 分块，仍含 `{video_key}` 目录层 |
| loader `_target_` | `fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset` | `fastwam.datasets.lerobot3.robot_video_dataset.RobotVideoDataset` |

### 8.3 显存与 batch size

8 × A800 80GB。官方 RoboTwin 用 **64 卡** bs16。5B 模型 + DeepSpeed ZeRO-1（只分片 optimizer state，参数不分片）+ 384×320×9帧 latent，bs16 在 80G 上可能偏紧。

- 建议 `batch_size: 8` 起步。
- **⚠️ 梯度检查点不可用**（2026-08-28 实测更正）：`mot.py:28-31` 对
  `mot_checkpoint_mixed_attn=true` 直接 `raise ValueError`，`mot.py:55-62` 还会对任何
  `use_gradient_checkpointing=True` 的 expert 再抛一次。设成 true 只会报错，不会省显存。
- **ZeRO-2 也无帮助**：实测 bs=16 在 zero1/zero2 下均 OOM —— 瓶颈是**激活值**
  （9 帧 384×320 过 30 层 5B 模型），不是优化器状态/梯度。
- OOM 的实际回退顺序：降 `batch_size`（8→6→4）→ 加 `gradient_accumulation_steps` 保持等效 batch。
- 显存有余可试编译加速（README 称 H20 上快约 10%）：
  ```bash
  ... model.compile_training_denoise=true
  ```
- 还有 `scripts/train_zero2.sh`（ZeRO-2，额外分片梯度）可作为进一步省显存的选项。
- 只有 DiT + proprio_encoder 参与训练（`trainer.py:95-101`），VAE / T5 冻结。

### 8.4 时序跨度只有 1.1 秒

`fps=30`、`num_frames=33` → 窗口 33/30 ≈ **1.1 秒**，包含 32 步 action + 9 帧视频（`action_video_freq_ratio=4`）。

而你的 episode 平均 **71 秒**，任务是"反复搬运直到箱子清空"这类长时程行为。1.1 秒的未来想象对该任务偏短。

想拉长时序：调 `global_sample_stride: 2` → 窗口变 2.2 秒。**但注意**这会让 action 也变成隔帧采样（控制频率从 30Hz 变 15Hz），必须与你真机部署时的执行频率对齐。这是个真实的调参决策点，不是 bug。

约束条件（`robot_video_dataset.py:67-70`）：`(num_frames-1) % action_video_freq_ratio == 0` 且 `((num_frames-1) / ratio) % 4 == 0`。当前 32/4=8、8%4=0 ✓。

### 8.5 proprio 的夹爪与 action 的夹爪不是同一个量

- `observation.gripper_position` → `left_gripper_percent` / `right_gripper_percent`（**百分比**）
- `actions[6]` / `actions[13]` → `gripper_width_left` / `gripper_width_right`（**宽度**）

单位不同。z-score 归一化后能训，但这是语义不一致 —— **真机部署时务必确认推理时喂入的 proprio 用的是训练时同一套百分比定义**。

另外布局也不对齐：proprio 是 `[12 关节, 2 夹爪]`，action 是 `[6 左关节, 1 左夹爪, 6 右关节, 1 右夹爪]`。这**没问题** —— LIBERO 也是 proprio 8 / action 7，两者各自独立归一化、独立编码，不要求逐维对应。

### 8.6 没有仿真评测入口

`experiments/` 下只有 `libero/run_libero_manager.py` 和 `robotwin/run_robotwin_manager.py`，都依赖各自仿真器。你的 Agilex 真机数据两者都用不上。

训练期间可用的指标（`eval_every` 触发，`trainer.py:587-617`）：

- `val_loss`
- 视频质量：`psnr_rg` / `ssim_rg` / `psnr_rd` / `ssim_rd` / `psnr_dg` / `ssim_dg`（real-gt / real-denoised / denoised-gt 三组对比）
- 动作精度：`action_l2` / `action_l1`
- `runs/.../eval/` 下的可视化视频

要上真机推理，参照 `experiments/robotwin/fastwam_policy/deploy_policy.py` 自己写一个 policy wrapper。注意 `infer_action` 现在直接返回 action + video latents，只有 `infer_joint` 才做 VAE 解码（action-only 部署更省算力）。

### 8.7 模型变体选择

`configs/task/` 下有 4 类（× 2 个 benchmark）：

| 变体 | model config | 特点 |
|---|---|---|
| `uncond` | `fastwam.yaml` | Fast-WAM：跳过测试时未来想象，直接出 action，**推理最快** |
| `idm` | `fastwam_idm.yaml` | 先想象未来视频，再反推 action |
| `joint` | `fastwam_joint.yaml` | 联合建模 |
| `optional_idm` | `fastwam_optional_idm.yaml` | **一个模型两种推理模式**，评测时用 `+EVALUATION.action_infer_mode=idm\|first_frame` 切换，无需重训 |

- 只求推理快 → `uncond`（本文档默认）
- 想研究"未来想象到底有没有用" → `optional_idm`（需重新生成对应的 ActionDiT backbone）

### 8.8 其他小项

- **action scheduler shift**：现在训练和评测都默认 `1.0`（`configs/model/fastwam.yaml` 的 `action_scheduler.train_shift/infer_shift`）。README 说 1.0~3.0 表现接近。若要评测**旧版**发布权重需设 `EVALUATION.sigma_shift=5.0`。
- **`val_set_proportion: 0.01`** → 471 × 1% ≈ 5 个 episode 留作验证，按 `seed=42` 打乱后切分（`base_lerobot_dataset.py:99-112`）。想完全不留验证集就设 `0.0`（此时 `build_datasets` 仍会建 val，但用的是同一批 episode）。
- **`skip_padding_as_possible`**：设 `true` 会在采到含 padding 的窗口时最多重试 3 次换随机样本。你的 episode 都很长（2000+ 帧），padding 比例极低，保持 `false` 即可。
- **`__getitem__` 静默兜底**：`robot_video_dataset.py:283-292` 捕获所有异常后**换随机样本继续**，只打印日志。所以配置错了不一定崩，可能表现为日志里刷 `Error processing sample idx ...`。**首次试跑一定要看日志**，别只看 loss 在降。
- **数据集统计量计算的内存**：`get_dataset_stats()` 用无界 `ThreadPoolExecutor` 提交全部 471 个 episode，每个构造 `(N, 32, 14)` 的滑窗张量（约 3.6 MB）。峰值约 1.7 GB，你有 1.4 TB 内存，无压力。

---

## 9. 执行清单

- [ ] **1.** 写 / 运行**字段改名**脚本（附录 A）：parquet 列改名 + info.json 改名 + episodes_stats 改名 + 视频目录软链接 —— **这是必做的转换 ①，与下面第 9 项的 v3.0 转换是两回事，见 [4.0](#40-先搞清楚本文档涉及两个不同的转换)**
- [ ] **2.** `conda create -n fastwam`，装 torch 2.7.1+cu128 与 `pip install -e .`
- [ ] **3.** 设 `DIFFSYNTH_MODEL_BASE_PATH`，生成 ActionDiT backbone（顺带自动下载 Wan2.2 全套权重，预留 60GB）
- [ ] **4.** 下载 `robotwin_uncond_3cam_384.pt` 用于热启动（**不要**用它的 dataset_stats）
- [ ] **5.** 新建 `configs/data/agilex_3cam.yaml`（第 6.1 节，train/val 两段）
- [ ] **6.** 新建 `configs/task/agilex_uncond_3cam_384_1e-4.yaml`（第 6.2 节）
- [ ] **7.** 跑 `precompute_text_embeds.py`
- [ ] **8.** 小规模试跑（`max_steps=50 batch_size=4`），**逐行看日志**确认无 `Error processing sample`，记录 step 时间
- [ ] **9.** 根据 step 时间决定是否做**可选的转换 ②**：转 LeRobot v3.0 / 转码 H.264（第 8.2 节，注意 v3.0 的三个前置条件）
- [ ] **10.** 把 `pretrained_norm_stats` 指向首次生成的 `dataset_stats.json`
- [ ] **11.** 正式训练（热启动 + `max_steps=20000`）
- [ ] **12.** 按需写真机部署 policy wrapper（参照 `experiments/robotwin/fastwam_policy/deploy_policy.py`）

---

## 附录 A：数据转换脚本

> 保存为 `scripts/convert_agilex_to_fastwam.py`。
> **说明：本脚本按第 4 节的分析编写，逻辑已对照代码校验，但尚未在你的数据上实际执行过。**
> 首次运行建议先加 `--limit 5` 只转 5 个 episode 验证，再全量跑。
> 需要 `pyarrow`（在 `fastwam` 环境里已有；也可用你的 `lerobot-convert` 环境跑）。

```python
#!/usr/bin/env python
"""把 Agilex/GR00T 风格的 LeRobot v2.1 数据集改名成 FastWAM 期望的字段命名。

FastWAM 从 shape_meta 的 key 拼出 LeRobot 字段名（见
src/fastwam/datasets/lerobot/base_lerobot_dataset.py:79-96）：
    images -> observation.images.{key}
    state  -> observation.state.{key}
    action -> action.{key}  /  "action" (key == "default")

因此需要做的改名：
    observation.image.top           -> observation.images.cam_high
    observation.image.left_wrist    -> observation.images.cam_left_wrist
    observation.image.right_wrist   -> observation.images.cam_right_wrist
    observation.gripper_position    -> observation.state.gripper_position
    actions                         -> action

视频不复制、不转码，只做目录软链接（28GB -> 0 额外磁盘）。
parquet 逐个重写列名（169MB，几分钟）。
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

# ---- 改名表 ----------------------------------------------------------------

VIDEO_RENAME = {
    "observation.image.top": "observation.images.cam_high",
    "observation.image.left_wrist": "observation.images.cam_left_wrist",
    "observation.image.right_wrist": "observation.images.cam_right_wrist",
}

COLUMN_RENAME = {
    "observation.gripper_position": "observation.state.gripper_position",
    "actions": "action",
}

ALL_RENAME = {**VIDEO_RENAME, **COLUMN_RENAME}


def convert_info(src_meta: Path, dst_meta: Path) -> dict:
    """改写 info.json 的 features 键（保持插入顺序）。"""
    info = json.loads((src_meta / "info.json").read_text())

    new_features = {}
    for key, value in info["features"].items():
        new_features[ALL_RENAME.get(key, key)] = value
    info["features"] = new_features

    (dst_meta / "info.json").write_text(json.dumps(info, indent=4))
    return info


def convert_episodes_stats(src_meta: Path, dst_meta: Path) -> None:
    """改写 episodes_stats.jsonl 的 stats 键。

    技术上可选（aggregate_stats 只对 key 求并集、不校验 features；FastWAM 的
    归一化用它自己生成的 dataset_stats.json），但保持一致更省心。
    """
    src = src_meta / "episodes_stats.jsonl"
    if not src.exists():
        print("[skip] episodes_stats.jsonl 不存在")
        return

    with src.open() as fin, (dst_meta / "episodes_stats.jsonl").open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["stats"] = {
                ALL_RENAME.get(k, k): v for k, v in record["stats"].items()
            }
            fout.write(json.dumps(record) + "\n")


def convert_parquet(src_root: Path, dst_root: Path, limit: int | None) -> int:
    """逐个重写 parquet 的列名。"""
    files = sorted((src_root / "data").rglob("*.parquet"))
    if limit is not None:
        files = files[:limit]

    for src_file in tqdm(files, desc="rewriting parquet"):
        table = pq.read_table(src_file)
        new_names = [COLUMN_RENAME.get(n, n) for n in table.column_names]
        table = table.rename_columns(new_names)

        dst_file = dst_root / src_file.relative_to(src_root)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_file)

    return len(files)


def link_videos(src_root: Path, dst_root: Path) -> None:
    """把每个相机目录软链接过去（不复制、不转码）。"""
    for chunk_dir in sorted((src_root / "videos").glob("chunk-*")):
        dst_chunk = dst_root / "videos" / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)

        for cam_dir in sorted(chunk_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            new_name = VIDEO_RENAME.get(cam_dir.name)
            if new_name is None:
                print(f"[warn] 未知相机目录，跳过：{cam_dir.name}")
                continue
            target = dst_chunk / new_name
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(cam_dir.resolve(), target_is_directory=True)
            print(f"[link] {new_name} -> {cam_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只转前 N 个 episode 的 parquet，用于快速验证",
    )
    args = ap.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()

    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # 1) meta
    info = convert_info(src_meta, dst_meta)
    print("[ok] info.json features:", list(info["features"].keys()))

    for name in ["tasks.jsonl", "episodes.jsonl"]:
        shutil.copy2(src_meta / name, dst_meta / name)
    print("[ok] 复制 tasks.jsonl / episodes.jsonl")

    convert_episodes_stats(src_meta, dst_meta)
    print("[ok] episodes_stats.jsonl")

    # 2) parquet
    n = convert_parquet(src_root, dst_root, args.limit)
    print(f"[ok] 重写 {n} 个 parquet")

    # 3) videos（软链接）
    link_videos(src_root, dst_root)

    print(f"\n完成。数据集根目录：{dst_root}")
    print("下一步：把 configs/data/agilex_3cam.yaml 的 dataset_dirs 指向它。")


if __name__ == "__main__":
    main()
```

### 转换后的自查

```bash
# 1) info.json 的 features 键
python -c "
import json
f=json.load(open('data/agilex_empty_the_box_fastwam/meta/info.json'))['features']
print(list(f.keys()))
"
# 期望包含：observation.images.cam_high / cam_left_wrist / cam_right_wrist,
#           observation.state.joint, observation.state.gripper_position,
#           observation.state.end, action

# 2) parquet 列名
python -c "
import pyarrow.parquet as pq
t=pq.read_table('data/agilex_empty_the_box_fastwam/data/chunk-000/episode_000000.parquet')
print(t.column_names)
"

# 3) 视频软链接可达
ls -l data/agilex_empty_the_box_fastwam/videos/chunk-000/
ls data/agilex_empty_the_box_fastwam/videos/chunk-000/observation.images.cam_high/ | head -3

# 4) parquet 数量应为 471
find data/agilex_empty_the_box_fastwam/data -name '*.parquet' | wc -l
```

---

## 附录 B：关键代码位置索引

排查问题时的定位表。

| 关注点 | 文件:行 |
|---|---|
| **LeRobot key 拼接规则** | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py:79-96` |
| train/val episode 切分（seed 42） | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py:99-112` |
| dataset 长度 = 帧数 | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py:186-187` |
| 归一化统计量计算 | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py:264-386` |
| `pretrained_norm_stats` 分支逻辑 | `src/fastwam/datasets/lerobot/robot_video_dataset.py:99-121` |
| **robotwin 三相机拼图（顺序敏感）** | `src/fastwam/datasets/lerobot/robot_video_dataset.py:168-192` |
| 视频帧下采样时机（v2.1 解码 33 帧） | `src/fastwam/datasets/lerobot/robot_video_dataset.py:156-165` |
| `num_frames` / `freq_ratio` 约束断言 | `src/fastwam/datasets/lerobot/robot_video_dataset.py:67-70` |
| 文本缓存路径与校验 | `src/fastwam/datasets/lerobot/robot_video_dataset.py:249-281` |
| prompt 模板 `DEFAULT_PROMPT` | `src/fastwam/datasets/lerobot/robot_video_dataset.py:23` |
| **异常静默兜底（换随机样本）** | `src/fastwam/datasets/lerobot/robot_video_dataset.py:283-292` |
| v3.0 loader（`presample_images=True`） | `src/fastwam/datasets/lerobot3/base_lerobot_dataset.py:11` |
| **v3.0 适配器全文仅 12 行（key 规则继承自 v2.1）** | `src/fastwam/datasets/lerobot3/base_lerobot_dataset.py:8-11` |
| v3.0 `codebase_version` 严格校验（须为 3.0） | `src/fastwam/datasets/lerobot3/lerobot_dataset.py:71-73` |
| v3.0 用 `meta/tasks.parquet`（非 jsonl） | `src/fastwam/datasets/lerobot3/lerobot_dataset.py:74-75` |
| processor shape 断言 | `src/fastwam/datasets/lerobot/processors/fastwam_processor.py:229-231` |
| state/action 拼接（ConcatLeftAlign） | `src/fastwam/datasets/lerobot/transforms/action_state_merger.py:63-66` |
| 模型权重下载源与重定向 | `src/fastwam/models/wan22/helpers/loader.py:125-138` |
| 下载相关环境变量 | `src/fastwam/models/wan22/helpers/io.py:33-51` |
| `load_checkpoint` 严格性（mot 松 / proprio 严） | `src/fastwam/models/wan22/fastwam.py:1211-1230` |
| **`.pt` 里只存 `mot`+`proprio_encoder`（无 VAE/T5）** | `src/fastwam/models/wan22/fastwam.py:1199-1209` |
| Wan 底座加载（dit/vae/text_encoder/tokenizer） | `src/fastwam/models/wan22/wan22.py:46-87` |
| video_expert / action_expert 的初始权重来源 | `src/fastwam/models/wan22/fastwam.py:136-158` |
| VAE 编码视频为 latent（训练全程需要） | `src/fastwam/models/wan22/fastwam.py:248` |
| **ActionDiT backbone 从 Wan DiT 插值降维** | `scripts/preprocess_action_dit_backbone.py:199-208` |
| precompute 只加载 T5 + tokenizer | `scripts/precompute_text_embeds.py:228-245` |
| val 数据集 stats 回落逻辑 | `src/fastwam/runtime.py:430-441` |
| `resume` 文件 vs 目录的语义 | `src/fastwam/trainer.py:315-327` |
| `max_steps` 优先级 | `src/fastwam/trainer.py:254-256` |
| 可训练参数范围（仅 DiT + proprio） | `src/fastwam/trainer.py:95-101` |
| 验证指标定义 | `src/fastwam/trainer.py:587-617` |
| checkpoint 保存结构 | `src/fastwam/trainer.py:618-647` |
| 启动脚本 / output_dir 命名 | `scripts/train_zero1.sh:104-116` |
| 文本缓存只读 tasks.jsonl | `scripts/precompute_text_embeds.py:114-145` |
| **文本缓存硬编码 `meta/tasks.jsonl`（v3.0 会报错）** | `scripts/precompute_text_embeds.py:120-122` |
| 文本缓存文件名 = prompt 的 SHA256（跨版本可复用） | `src/fastwam/datasets/lerobot/robot_video_dataset.py:254-255` |

---

## 附录 C：RoboTwin 权重热启动的可迁移性评估与 A/B 验证流程

> 本附录针对一个具体疑问补充：**"我的数据只是格式和 RoboTwin 类似，但相机位置、机械臂细节可能不一致，加载 RoboTwin 权重热启动会有问题吗？"**
>
> 结论：**值得试，且已确认没有阻塞性问题。** 但到底有多少收益是经验问题，用 C.4 的 A/B 流程实测。

### C.1 逐组件可迁移性评估

| 组件 | 语义是否对齐 | 风险 | 说明 |
|---|:--:|:--:|---|
| 三路相机的**角色** | ✅ 完全对齐 | 无 | 顶部俯视 / 左腕 / 右腕，`concat_multi_camera="robotwin"` 的拼图布局也一致 |
| **action 布局** | ✅ 完全对齐 | 无 | 双方都是 `[左6关节, 左夹爪, 右6关节, 右夹爪]` |
| **夹爪方向约定** | ✅ 已确认一致 | 无 | 均为"数值越大越开"（用户确认） |
| 关节绝对范围 / 运动学 | ❌ 不同 | **无** | action / proprio 都用**你自己的** `dataset_stats.json` 做 z-score，绝对范围不影响 |
| 相机外参 / FOV | ❌ 不同 | 低 | 正常 domain shift，微调可解 |
| **视觉域 sim vs real** | ❌ 不同 | **中** | RoboTwin 是 SAPIEN 仿真渲染，你的是真机 RGB。**这是唯一实质风险**，也是 A/B 要回答的问题 |
| proprio 布局 | ❌ 错位 | **低（自校正）** | 见 C.2、C.3 |

有利因素值得强调：三路相机的**语义角色完全一致**（顶部大图 + 两个腕部小图），所以拼图后模型"哪块区域是什么"的先验是对的，只是纹理 / 光照 / 视角细节不同。而 video expert 的底座 Wan2.2 本身是在真实视频上预训练的，RoboTwin 微调只跑了 5 epoch，不太可能有严重的向仿真域的灾难性遗忘。

### C.2 RoboTwin 的 proprio 布局（权威出处）

`third_party/RoboTwin/envs/_base_task.py:509-516`：

```python
left_jointstate  = self.robot.get_left_arm_jointState()   # 7 维 = 6 关节 + 1 夹爪
right_jointstate = self.robot.get_right_arm_jointState()  # 7 维

pkl_dic["joint_action"]["left_arm"]      = left_jointstate[:-1]   # 6
pkl_dic["joint_action"]["left_gripper"]  = left_jointstate[-1]    # 1
pkl_dic["joint_action"]["right_arm"]     = right_jointstate[:-1]  # 6
pkl_dic["joint_action"]["right_gripper"] = right_jointstate[-1]   # 1
pkl_dic["joint_action"]["vector"] = np.array(left_jointstate + right_jointstate)  # 14
```

`experiments/robotwin/fastwam_policy/deploy_policy.py:238` 证实 proprio 就是取这个 vector：

```python
state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
proprio = self._normalize_state(state_vector)
```

**所以 RoboTwin 的 proprio 布局 = `[左6关节, 左夹爪, 右6关节, 右夹爪]`**，与其 action 布局相同。

而 §6.1 给出的配置（`joint` 12 维 + `gripper_position` 2 维，按 `shape_meta.state` 顺序拼接）布局是 `[左6关节, 右6关节, 左夹爪, 右夹爪]`，二者错位：

| 维度 | RoboTwin proprio | §6.1 配置 | |
|:--:|---|---|:--:|
| 0–5 | 左臂 6 关节 | 左臂 6 关节 | ✅ |
| 6 | **左夹爪** | 右臂关节 0 | ❌ |
| 7–11 | 右臂关节 0–4 | 右臂关节 1–5 | ❌ |
| 12 | 右臂关节 5 | 左夹爪 | ❌ |
| 13 | 右夹爪 | 右夹爪 | ✅ |

即维度 6–12 共 7 维语义错位。

### C.3 但这个错位是低风险、可自校正的

`fastwam.py:370-378` + `fastwam.py:228-246` 显示 proprio 的完整路径：

```python
proprio = proprio[:, 0, :]                     # 33 帧里只取第 0 帧 -> [B, 14]
proprio_token = self.proprio_encoder(...)      # nn.Linear(14, 4096) -> [B, 1, 4096]
context = torch.cat([context, proprio_token], dim=1)   # append 到 128 个 text token 之后
```

三个原因说明影响有限：

1. **参数量极小**：`nn.Linear(14, 4096)` = 5.7 万参数，且直接接收梯度（`trainer.py:95-101` 里 `proprio_encoder` 是可训练的），几百步即自校正。
2. **信息占比小**：proprio 只是 129 个 context token 里的 **1 个**。
3. **不会报错**：`proprio_encoder` 虽然用 `strict=True` 加载（`fastwam.py:1222`），但校验的是 shape `[4096, 14]`，与**语义布局无关**。只要 `proprio_output_dim=14` 就能加载。

> ⚠️ 前文若有把此项标为"高风险"之处，以本节为准：**实际是低风险、自校正**。对齐 proprio 属于"顺手能拿的小收益"，不是必须修的阻塞问题。

**决策**：为了让 A/B 只有单一变量，**A/B 阶段保持数据格式不变**（即 §6.1 的两 key 配置、附录 A 的纯改名脚本），不做 proprio 重排。

### C.3.1 （可选）以后若想对齐 proprio

在附录 A 的转换脚本里额外合成一个 14 维 `observation.state` 列：

```
observation.state = concat(
    observation.state.joint[0:6],        # 左臂 6 关节
    observation.gripper_position[0:1],   # 左夹爪
    observation.state.joint[6:12],       # 右臂 6 关节
    observation.gripper_position[1:2],   # 右夹爪
)
```

然后把 §6.1 里 state 段改成与 RoboTwin 完全一致的单 key：

```yaml
    state:
      - key: default          # -> "observation.state"
        raw_shape: 14
        shape: 14
```

三个好处：① proprio_encoder 热启动生效；② proprio 与 action 布局一致；③ 配置与 RoboTwin 对齐，少一个出错点。
代价：转换脚本从"纯改名"变成"改名 + 合成一列"，且要同步更新 `info.json` 的 features。

### C.4 A/B 验证流程

只有一个变量：`resume`。数据、配置、随机种子全部相同。

```bash
conda activate fastwam
cd /home/gaomeng/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints

# ---------- A：纯 Wan 初始化（不加载 RoboTwin 权重）----------
bash scripts/train_zero1.sh 8 task=agilex_uncond_3cam_384_1e-4 \
  max_steps=2000 eval_every=250 save_every=1000

# 记下 A 生成的 stats 路径（首次运行 pretrained_norm_stats=null 时自动生成）
STATS=./runs/agilex_uncond_3cam_384_1e-4/<A的run_id>/dataset_stats.json

# ---------- B：热启动 RoboTwin 权重，其余全部相同 ----------
bash scripts/train_zero1.sh 8 task=agilex_uncond_3cam_384_1e-4 \
  max_steps=2000 eval_every=250 save_every=1000 \
  data.train.pretrained_norm_stats=$STATS \
  data.val.pretrained_norm_stats=$STATS \
  resume=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

把 A 的 `dataset_stats.json` pin 给 B，既保证归一化完全一致，又省掉 B 重新遍历 471 个 episode 的时间。

#### 必须保持一致的参数

| 参数 | 原因 |
|---|---|
| `max_steps` | warmup = `int(max_steps * 0.05)`、cosine 调度总长都由它决定（`trainer.py:112-131`），不同就没法比 |
| `batch_size` | 决定每步样本数 |
| `num_workers` | 影响吞吐，虽不影响精度但影响 step 时间对比 |
| `seed`（默认 42） | 决定 train/val 切分与数据顺序 |
| `dataset_stats.json` | 决定归一化，务必用同一份 |

#### 看哪些指标

| 指标 | 优先级 | 说明 |
|---|:--:|---|
| `action_l2` | **主** | 任务相关，直接反映动作预测精度 |
| `val_loss` | 辅 | 整体收敛速度 |
| `psnr_rg` / `ssim_rg` | 辅 | 视频重建质量，能侧面反映视觉域是否适配 |

val 集是同一批按 `seed=42` 切出的约 5 个 episode（471 × `val_set_proportion=0.01`），两次跑完全可比。

#### 结论判读

| 结果 | 判读 | 行动 |
|---|---|---|
| **B 明显优于 A** | 结论很稳（B 还带着 proprio 错位的让分仍然赢） | 正式训练用 B 方案 |
| **两者差距 < 5%** | 热启动无所谓 | 选 A，少一个外部依赖 |
| **B 明显差于 A** | sim→real 视觉域先验是负担 | 试 C.5 的"只加载 action expert" |

#### 一个已知局限

2000 步 cosine-to-zero 的 LR 轨迹与 20000 步的不同，所以 A/B 反映的是**早期收敛优势**。通常能预测全程，但不是必然。若两者非常接近，可以把对比延长到 5000 步再判。

### C.5 第三选项：只加载 action expert

若 A/B 显示全量热启动反而更差，说明 video expert 里的 SAPIEN 仿真视觉先验是负担。此时可以只保留 action expert（它学到的是"video + proprio → 双臂关节动作"的映射，结构上更可迁移），让 video expert 保持 Wan2.2 原始的真实视频先验：

```python
import torch

src = "./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
dst = "./checkpoints/fastwam_release/robotwin_action_only.pt"

p = torch.load(src, map_location="cpu")
p["mot"] = {k: v for k, v in p["mot"].items() if k.startswith("mixtures.action.")}
torch.save(p, dst)
print(f"kept {len(p['mot'])} action-expert tensors -> {dst}")
```

可行性依据：
- `mot` 是 `nn.ModuleDict({"video": ..., "action": ...})`（`mot.py:26`），state_dict 的 key 前缀就是 `mixtures.video.*` / `mixtures.action.*`。
- `load_checkpoint` 对 `mot` 用 `strict=False`（`fastwam.py:1214`），缺失的 `mixtures.video.*` 会被忽略，那部分保持 Wan2.2 原始权重。

反向操作（只保留 video expert）把前缀换成 `mixtures.video.` 即可。

> ⚠️ `strict=False` 只忽略 missing / unexpected key，**shape 不匹配仍会抛 RuntimeError**。当前维度全部对得上；但若以后改了 `proprio_output_dim`，`proprio_encoder` 那里是 `strict=True`，会直接 crash。

### C.6 本附录新增的代码位置索引

| 关注点 | 文件:行 |
|---|---|
| **RoboTwin proprio 布局 `[L6,Lg,R6,Rg]`** | `third_party/RoboTwin/envs/_base_task.py:509-516` |
| RoboTwin 部署时 proprio 取 `joint_action.vector` | `experiments/robotwin/fastwam_policy/deploy_policy.py:238-239` |
| proprio 只取第 0 帧 | `src/fastwam/models/wan22/fastwam.py:370-378` |
| proprio → 1 个 token 并 append 到 context | `src/fastwam/models/wan22/fastwam.py:228-246` |
| `proprio_encoder = nn.Linear(14, 4096)` | `src/fastwam/models/wan22/fastwam.py:60` |
| `proprio_encoder` 用 `strict=True` 加载 | `src/fastwam/models/wan22/fastwam.py:1222` |
| `mot` = ModuleDict（key 前缀 `mixtures.*`） | `src/fastwam/models/wan22/mot.py:26` |
| warmup / cosine 总长由 `max_steps` 决定 | `src/fastwam/trainer.py:112-131` |
