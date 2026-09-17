# Agilex 数据训练 FastWAM —— 执行计划与执行记录

> 项目根：`/home/gaomeng/FastWAM`　|　代码版本：`7faa711`
> 数据：`data/agilex_empty_the_box_all_470`（LeRobot v2.1，471 eps，1,003,984 帧，3×AV1 480×640，双臂 14 维 action）
> 硬件：8 × A800-80GB / 80 CPU / 1.4 TB RAM / 磁盘余 1.4 TB
> 配套文档：`AGILEX_FINETUNE_GUIDE.md`（原理与排查手册，本文件是它的执行版）
> 创建：2026-08-26

---

## 目标

1. 把 Agilex 数据适配到 FastWAM 可训练的形态
2. 跑 **A/B 短跑**对比"加载 RoboTwin 权重热启动是否有收益"
   - **A** = 纯 Wan2.2 初始化
   - **B** = 热启动 `robotwin_uncond_3cam_384.pt`
3. 以 **A 为主**做正式训练，全程 wandb 记录

## 执行约束（用户指定）

| # | 约束 | 落实 |
|---|---|---|
| 1 | conda 新环境，按 README 配置 | Phase 0 严格照 README `Environment Setup` |
| 2 | 需要 wandb 记录和显示 | 所有训练 `wandb.enabled=true`；A/B 同 `group=ab_warmstart` 便于叠图 |
| 3 | **中途无法确认，需自主执行** | 全流程无确认门；`max_steps`/`batch_size`/`num_workers`/A/B 判读均按实测自主决定并记录理由 |
| 4 | 遇问题自己解决 | 见「故障处理预案」；仅触及授权边界才停下 |
| 5 | 计划写成新 md 后执行 | 本文件 |
| 6 | **所有产物在当前目录** | `env.sh` 把 `DIFFSYNTH_MODEL_BASE_PATH`/`HF_HOME`/`MODELSCOPE_CACHE` 全指到项目内；wandb 借 `wandb.init(dir=output_dir)` 落在 `runs/.../wandb/` |

**未获授权**（触及则停下报告）：转 LeRobot v3.0、视频转码、改 proprio 布局、换模型变体。IO 优化仅限配置层参数。

---

## 关键前提：字段改名（不做直接 crash）

`src/fastwam/datasets/lerobot/base_lerobot_dataset.py:79-96` 从 `shape_meta` 的 key **拼**出 LeRobot 字段名，不接受任意命名：

```python
images: f"observation.images.{key}"
state:  f"observation.state.{key}"   /  "observation.state"  (key=="default")
action: f"action.{key}"              /  "action"             (key=="default")
```

拼出的名字用于三处：parquet 列名、`info.json` 的 `features` 键、`video_path` 模板的 `{video_key}`（= **视频子目录名**）。

| 原字段 | 新字段 | 载体 |
|---|---|---|
| `observation.image.top` | `observation.images.cam_high` | 视频目录（软链接） |
| `observation.image.left_wrist` | `observation.images.cam_left_wrist` | 视频目录（软链接） |
| `observation.image.right_wrist` | `observation.images.cam_right_wrist` | 视频目录（软链接） |
| `observation.gripper_position` | `observation.state.gripper_position` | parquet 列 |
| `actions` | `action` | parquet 列 |

`observation.state.joint`(12) / `observation.state.end`(12) 天然合规，不动。**原数据目录全程只读。**

---

## 环境探查结论（执行前已完成）

| 项目 | 结论 |
|---|---|
| pip | 已配阿里云镜像；`download.pytorch.org/whl/cu128` 可达 0.7s（`torch==2.7.1+cu128` 的 local tag 只在此，`--extra-index-url` 必需） |
| ModelScope | 可达 0.04s（FastWAM 默认源） |
| wandb | `~/.netrc` 已有 `api.wandb.ai` 凭证 → **online 可直接用** |
| 现有 Wan 缓存 | 仅 `Wan2.2_VAE.pth`(2.7G)；默认 `redirect_common_files: true` 走 `.safetensors`，**不复用**，仍需下 ~50GB |
| 现有 conda 环境 | `lawam` = torch 2.6.0/transformers 5.2.0，与钉死版本冲突，必须新建 |
| 编译器 | gcc 13.3 + nvcc 12.8；需 `CUDA_HOME=/usr/local/cuda` |
| 数据健康度 | action/state **无零方差维、无 NaN/Inf**，std 0.22–0.63；normalizer 有 `std_reg=1e-8` + clamp[-5,5]（`utils/normalizer.py:92-129`） |
| 下载落盘 | `io.py:73-97` 两源都用 `local_dir=./checkpoints/<model_id>`，天然满足约束 6 |

### 两项实测发现（改变了原方案）

**1. AV1 解码不是瓶颈** —— 推翻 `AGILEX_FINETUNE_GUIDE.md` §8.2 的判断
关键帧间隔 = **2 帧**（985 keyframes / 1970 frames），单线程 dav1d **411 fps**，seek 几乎免费。
每样本 33 帧 × 3 相机 = 99 帧 ÷ 411 ≈ **0.24 s**；每 GPU 8 workers 可供 ~33 样本/s，远超 5B 模型 step 需求。
→ 「只调 num_workers」的授权范围够用。

**2. eval 每次只采 1 样本/GPU** —— `trainer.py:428-437`
`eval_index = torch.randint(...)` 按 `global_step + process_index` 播种，每 rank 取 1 个 val 样本，共 8 个。
→ A/B 的 `global_step` 序列相同 → 抽到**同一批** eval 样本，可比；但 8 样本单点噪声大。
→ 对策：`eval_every=100` 拿 ~20 点，**主看 train loss 曲线**，辅看 `action_l2` 后 10 点均值。

---

## 阶段计划

### Phase 0：环境搭建
```bash
conda create -n fastwam python=3.10 -y && conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
cd /home/gaomeng/FastWAM && pip install -e .
source env.sh
```
验收：`import torch, deepspeed, accelerate, wandb, fastwam`；`cuda.device_count()==8`；`ds_report` 无致命缺失。

### Phase 1：数据字段改名
`scripts/convert_agilex_to_fastwam.py` → `data/agilex_empty_the_box_fastwam/`
先 `--limit 5` 验证，再全量（471 parquet）。视频软链接，0 额外磁盘。

### Phase 2：权重准备
```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16          # 顺带下载 Wan2.2 DiT+T5+VAE+tokenizer ~50GB
huggingface-cli download yuanty/fastwam robotwin_uncond_3cam_384.pt \
  --local-dir ./checkpoints/fastwam_release
```

### Phase 3：配置文件
`configs/data/agilex_3cam.yaml`（取自指南 §6.1）+ `configs/task/agilex_uncond_3cam_384_1e-4.yaml`（§6.2）

### Phase 4：T5 文本缓存
`python scripts/precompute_text_embeds.py task=agilex_uncond_3cam_384_1e-4`

### Phase 5：冒烟 + 三项探测
所有运行用 `RUN_ID=<名字>` 固定输出目录（`train_zero1.sh:59`），日志 `tee` 到 `runs/<task>/<RUN_ID>/train.log`（trainer 只输出 stdout）。
- **5a 冒烟**：`max_steps=20 batch_size=4 num_workers=4 eval_every=10 save_every=20`
  **关键验收**：`grep -c "Error processing sample" train.log` == **0**（`robot_video_dataset.py:283-292` 会静默换样本继续，配置错了不一定崩）
- **5b ckpt 体积**：量 `state/` 与 `weights/*.pt` 大小 → 反推 `save_every` + 旧 state 清理策略（不做可能吃掉 1TB）
- **5c batch_size**：bs ∈ {8,12,16}，选峰值 <~70GB 的最大值（给 eval 的 infer+VAE decode 留 ~10GB）
- **5d num_workers**：{6,8}，比 `steps_per_sec` 与 GPU 利用率（8 进程 × 8 = 64 + 8 主进程 = 72 < 80 核）

### Phase 6：A/B 短跑
`AB_STEPS` 按实测 step 时间取，使单跑 ≈ 2h。两跑除 `resume` 外全部相同，B 复用 A 的 `dataset_stats.json`。

**自主判读规则**：

| 结果 | 行动 |
|---|---|
| B 优于 A ≥10%（loss 与 action_l2 均） | 正式训练用 B |
| 差距 <5%（噪声内） | 用 **A** |
| B 明显差于 A | 用 **A**，报告并建议指南 C.5「只加载 action expert」（需另行授权） |

### Phase 7：最终 A 训练 + 监控
自主定 `max_steps`（目标 wall-clock ≤3 天，覆盖 ≥1 epoch）→ 后台启动 → ~2.5h 间隔 cron 健康检查 → 进度追加到本文件「执行记录」。

---

## 故障处理预案

| 症状 | 自主处置 |
|---|---|
| Wan 权重下载中断 | 重跑 Phase 2，`require_downloading()`(`io.py:56-71`) 天然续传 |
| pip 装依赖失败 | 阿里云 → 清华源（实测 0.99s）；torch 必须走 `download.pytorch.org` |
| deepspeed JIT 失败 | 查 `CUDA_HOME` + `ds_report`。**仓库三个 accelerate 配置全是 DEEPSPEED，无纯 DDP 回落**，只能就地修 |
| OOM | 降 bs（8→6→4）→ `gradient_accumulation_steps=2` 保持等效 batch。**梯度检查点不可用**（`mot.py:28-31` 直接抛错），**ZeRO-2 也无帮助**（实测 bs=16 仍 OOM，瓶颈是激活值） |
| `Error processing sample` 刷屏 | **立刻停**，定位字段/shape 问题，修完重跑 5a |
| GPU 利用率 <60% | 只调 `num_workers`/`batch_size`；仍低则记录继续，不动数据 |
| loss NaN | 回退最近 state + 降 lr 试一次；仍 NaN 则停下报告 |
| 进程崩溃 | `resume=runs/.../checkpoints/state/step_XXXXXX`（**目录**=完整状态恢复，`trainer.py:320-322`） |
| 磁盘将满 | 清旧 state，留最近 2–3 份 + 全部 weights `.pt` |
| wandb 掉线 | 不应中断训练；必要时 `wandb.mode=offline`，事后 `wandb sync` |

---

## 执行记录

> 按时间倒序追加。每个 Phase 完成后补实测数据。

### 2026-08-28 下午 — GPU 空出，Phase 5 执行中

vLLM 服务已被停止（每卡空闲 81154 MiB），守候 cron `f3315d67` 已删除，训练流程启动。

#### Phase 5a 冒烟测试 —— ✅ 通过

`RUN_ID=smoke`，8 卡 / bs4 / nw4 / 20 步 / eval_every=10。

| step | train loss | val_loss | infer_psnr | infer_ssim | action_l2 | action_l1 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.7016 | 1.7175 | 21.06 | 0.6064 | 0.1601 | 0.3132 |
| 20 | **1.3283** | **1.2606** | 21.74 | 0.6504 | **0.1071** | **0.2466** |

- 模型规模确认：**video expert 5.00 B + action expert 1.02 B = 6.02 B**（`mot.py:67` 日志）
- `Error processing sample` 计数 = **0** ✅（字段改名正确）
- eval 视频已生成（`eval/step_000010_rank_00*.mp4`，8 个 rank 各一个）
- wandb run 正常：`fastwam-agilex/smoke`
- loss 与 action_l2 两个 eval 点都在下降，管线健康

**踩到的两个坑（已修）**

1. `accelerate: command not found` (exit 127) —— `train_zero1.sh:110` 直接调用 `accelerate`，
   而 `source env.sh` 只设了变量没改 `PATH`。已在 `env.sh` 里加
   `export PATH="$FASTWAM_ENV/bin:$PATH"`。
2. 容器镜像默认 `NCCL_DEBUG=INFO`，20 步的日志被刷成 1243 行 NCCL 噪声，真正的 loss 被埋掉
   （我第一次 grep `[train]` 没匹配到，误判为失败）。已在 `env.sh` 里加 `NCCL_DEBUG=WARN`。

> 另外注意：rich 日志处理器会**折行并重排** `[train]`/`[eval]`/`[ckpt]` 前缀，
> 用 `grep "\[train\]"` 抓不到。解析日志要先 `sed 's/\x1b\[[0-9;]*m//g'` 去 ANSI，
> 再按 `step=N/` 和 `loss=` 之类的裸模式匹配。

#### Phase 5b checkpoint 体积 —— ⚠️ 计划中预判的风险已确认

| 产物 | 体积 |
|---|---|
| `checkpoints/weights/step_NNNNNN.pt` | **12 GB** |
| `checkpoints/state/step_NNNNNN/`（ZeRO 完整状态） | **80 GB** |
| **单次保存合计** | **92 GB** |

80 GB 的构成对得上 6.02 B 可训练参数：Adam 的 m+v 用 fp32 = 8 B/param ≈ 48 GB，
加 fp32 master weights ≈ 24 GB，再加 bf16 副本与通信 buffer（`reduce_bucket_size: 2e8`）。

**若照原配置 `save_every=2000` 跑 20000 步 → 10 次保存 → 920 GB**，磁盘只剩 1.2 T，
叠加 A/B 两轮就会撑爆。这正是计划里把 5b 列为必做项的原因。

**对策（已定）**
- 正式训练 `save_every` 放大，且**只保留最近 1 份 state**，新的存成功后立刻删旧的
- A/B 属一次性对比，跑完立即删 `state/`，只留 12 GB 的 weights
- 已删掉冒烟测试的 state，释放 80 GB

> 另一个发现：`trainer.py:822-828` 在 `global_step >= max_steps` 时**无条件**保存，
> 无法用配置关掉。所以探测脚本改用「`max_steps` 设成 999999，跑够步数后 kill 进程」，
> 完全避免每次探测写 92 GB。

#### Phase 5c/5d 结论：ZeRO-1 + bs=8 + nw=8

追加两组探测后的完整结果：

| 配置 | 峰值显存 | 占比 | 步时间 | 吞吐 | 结果 |
|---|---:|---:|---:|---:|:--|
| zero1 bs=8（无 eval） | 76,309 MiB | 93% | **1.44 s** | 44.4 样本/s | ok |
| zero1 bs=8（**开 eval**） | 76,217 MiB（瞬时观测到 78,329） | 96% | — | — | ok |
| zero1 bs=12（无 eval） | 78,617 MiB | 96% | 1.82 s | 52.6 样本/s | ok |
| zero1 bs=12（**开 eval**） | 79,685 MiB | **97.3%** | — | — | ok |
| zero1 bs=16 | 81,085 MiB | 99% | — | — | **OOM** |
| **zero2** bs=16 | 81,061 MiB | 99% | — | — | **OOM** |
| **zero2** bs=24 | 81,105 MiB | 99% | — | — | **OOM** |

**选定 `batch_size=8` / `num_workers=8` / ZeRO-1**，理由：

1. bs=12 开 eval 后只剩 **2.2 GB** 余量。多天训练里一次内存碎片或稍长的 eval 就会 OOM 把整个
   run 打死，18% 的吞吐收益不值这个风险。bs=8 留约 3.6 GB。
2. **ZeRO-2 并没有帮助** —— bs=16 在 zero1/zero2 下都 OOM，说明瓶颈是**激活值**
   （9 帧 384×320 过 30 层 5B 模型），不是优化器状态或梯度。所以 zero2 的额外通信开销白付。
3. dataloader 有 2.1× 余量（见下），bs=8 时 GPU util 实测 100%，算力已经吃满。

#### ⚠️ 文档更正：梯度检查点在本实现里**不可用**

我在 `AGILEX_FINETUNE_GUIDE.md` §8.3 和本文件早先的「故障处理预案」里，把
`model.mot_checkpoint_mixed_attn=true`（开梯度检查点）列为 OOM 的首选回退手段。**这是错的。**

`mot.py:28-31`：
```python
if mot_checkpoint_mixed_attn:
    raise ValueError("Wan MoT gradient checkpointing is not supported by the compiled tensor core.")
```
`mot.py:55-62` 还会对任何 `use_gradient_checkpointing=True` 的 expert 再抛一次错。
而 `configs/model/fastwam.yaml` 里两个 DiT 的 `use_gradient_checkpointing` 都插值自
`${model.mot_checkpoint_mixed_attn}` —— 所以设成 true **只会直接报错，不会省显存**。

**修正后的 OOM 回退顺序**：
1. 降 `batch_size`（8 → 6 → 4）
2. 加 `gradient_accumulation_steps` 保持等效 batch（吞吐会降）
3. ~~梯度检查点~~ ❌ 不可用
4. ~~ZeRO-2~~ ❌ 实测无帮助（瓶颈是激活值）


#### Phase 5d dataloader 是否瓶颈 —— ✅ 不是

用 5c 的实测值反算：

| | 值 |
|---|---|
| bs=8 时每 GPU 进程的**需求** | 8 样本 / 1.44 s = **5.55 样本/s** |
| 8 个 worker 的**供给**（675 ms/样本） | **11.9 样本/s** |
| 余量 | **2.1×** |

所以 `num_workers=8` 足够，训练不是 IO-bound（GPU util 实测 100%）。
这也印证了「AV1 解码不是瓶颈」以及「只调 num_workers 的授权范围足够」。



- [x] 环境探查完成（见上「环境探查结论」）
- [x] 本计划文档创建
- [x] `env.sh` 创建（统一环境变量，约束 6）
- [x] `scripts/convert_agilex_to_fastwam.py` 创建
- [x] `configs/data/agilex_3cam.yaml` 创建
- [x] `configs/task/agilex_uncond_3cam_384_1e-4.yaml` 创建
- [x] **Phase 1 数据改名 —— 完成并验证**
- [x] **Phase 0 环境搭建 —— 完成并验证**
- [x] **配置组装验证 —— 通过**（hydra compose，含维度断言）
- [x] **归一化统计量预计算 —— 完成**（`runs/_shared/dataset_stats.json`）
- [x] **Phase 2 权重准备 —— 完成**（Wan DiT/T5/VAE/tokenizer + ActionDiT backbone + RoboTwin 热启动权重）
- [x] **Phase 4 文本缓存 —— 完成**
- [x] **数据管线端到端验证 —— 通过**（真实取样 + collate）
- [x] **Phase 5 冒烟 + 探测 —— 完成**
- [x] **Phase 6 A/B 短跑 —— 完成，结论：用 A**
- [x] **Phase 7 最终 A 训练 —— 运行中**（无人值守守护 + 自动续训 + 自动清盘 + OSS 同步）

#### Phase 6 A/B 短跑结果 —— 结论：**热启动无收益，采用 A**

> ## ⛔ 2026-09-07 更正：**本小节的 A/B 结论无效,已作废。**
>
> `resume=<文件>` 当时是**空操作**:权重加载在 `accelerator.prepare()` 之后,
> DeepSpeed ZeRO-1 每个 step 用引擎初始化时的 fp32 master 副本覆盖回 bf16 参数,
> 热启动权重只活过第 1 次前向。所以 A 与 B 训的是**同一个模型**,曲线四位小数级吻合
> 是必然的,不是"负迁移"。
>
> 下面「已核实：热启动**确实生效**」那段只验证了 key 名匹配(1649/1649)与日志行,
> **没有验证权重活到第 1 个 step 之后**。
>
> 详情与修复见 **`WANPRETRAIN_TRANSFER_PLAN.md` §3**。
> **`final_A` 与本文件其余全部内容不受影响**(final_A 是 `resume=null`)。

两跑各 5000 步（比原计划的 2000–3000 更长，噪声更小），`eval_every=250` → 20 个 eval 点。
除 `resume` 外配置完全相同，共用 `runs/_shared/dataset_stats.json`。

| 指标 | A（纯 Wan 初始化） | B（热启动 RoboTwin） | 差异 |
|---|---:|---:|---:|
| step=10 train loss（几乎未学习，反映初始权重质量） | 2.3440 | 2.3507 | +0.29% |
| 末 5 个 log 点 train loss | 0.3433 / 0.3677 / 0.3692 / 0.3850 / 0.2646 | 0.3437 / 0.3671 / 0.3683 / 0.3868 / 0.2664 | 4 位小数级 |
| **action_l2 后 10 个 eval 点均值** | **0.02848** | **0.02856** | **+0.28%** |
| val_loss 后 10 点均值 | 0.39932 | 0.40017 | +0.21% |

**B 全面略差于 A，差异全部 <1%，远低于判读规则的 5% 阈值 → 采用 A。** 与「主要训练 A」的意图一致。

##### 已核实：热启动**确实生效**，不是静默失败（本次新增验证）

A 和 B 的曲线过于接近，需排除「`resume` 指定了但权重没真正加载」的可能
——`fastwam.py:1214` 对 `mot` 用 `strict=False`，key 名不匹配会**静默丢弃全部权重**。三重证据：

1. 日志有 `trainer.py:326` 的 `Loading weight checkpoint only: ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`；`ab_B/config.yaml` 里 `resume` 已设，`ab_A` 为 `null`
2. **key 名逐一比对**：官方权重与我们自己训练保存的权重（同一份代码写出）
   mot 张量数均 **1649**，**交集 1649，差集 0**（前缀均为 `mixtures.action.*` / `mixtures.video.*`）
3. `proprio_encoder` 形状一致：`weight (4096, 14)` / `bias (4096,)`

所以权重是真的载入了。**「无收益」是一个真实的负迁移结果，而非 bug。**

##### 为什么没有收益（机理推测）

1. **归一化基准不同**：RoboTwin 的 action 用它自己的关节范围做 z-score，我们用 Agilex 的。
   同样是「14 维标准化值」，同一个数值代表的物理量完全不同，学到的映射不可直接迁移。
2. **sim → real 视觉域差异**：RoboTwin 是 SAPIEN 渲染，我们是真机 RGB。
3. **机器人本体不同**：Cobot Magic vs RoboTwin 的 Aloha/ARX，运动学与关节范围都不同。

step=10（lr=4.38e-06，几乎没学习）B 的 loss 就已经和 A 持平，说明 RoboTwin 权重在 Agilex 数据上
的初始表现并不优于 Wan 初始化 —— 与上述机理一致。

> 附带的方法论提示：`trainer.py:428-437` 的 eval 每次只采 **1 样本/GPU = 8 个**，单点噪声很大
> （A 的 action_l2 在 0.0142~0.0534 之间跳）。所以判读必须看**后 10 点均值**与 train loss 曲线，
> 单看某一步会得出完全相反的结论。

#### Phase 7 最终 A 训练 —— 运行中

| 项 | 值 |
|---|---|
| run 目录 | `runs/agilex_final_3cam_384_1e-4/final_A` |
| task 配置 | `configs/task/agilex_final_3cam_384_1e-4.yaml` |
| 方案 | **A** —— `resume: null`，纯 Wan2.2 初始化 |
| 规格 | bs 8 × 8 卡 / nw 8 / ZeRO-1 / `num_epochs=5` → **77,690 步** |
| 启动时间 | 2026-08-28 19:38 |
| wandb | https://wandb.ai/menggao6073-geek-/fastwam-agilex/runs/8f4gqlmo |

**无人值守设计**：`scripts/run_final.sh _daemon` 用 `setsid` 脱离终端（PPID=1），
不依赖 Claude 会话在线 —— 纯机械的监控/续训/清盘不需要模型参与，零上下文开销。
配 `.run_final.pid` 互斥（读 `/proc/<pid>/cmdline` 校验防 PID 复用）。
`scripts/autosync_daemon.sh` 每 30 分钟增量 rsync 到 `/mnt/data/gaomeng/FastWAM`（OSS，排除 80GB 的 ZeRO state）。

**磁盘安全**（计划里预判的风险已实际生效）：单次保存 92 GB（weights 12 + state 80），
38 次 = 3.5 TB，而共享盘只剩约 850 G。每轮监控跑 `scripts/prune_ckpt.sh`，
只留最近 **1 份 state + 3 份 weights**，稳态约 **114 GB**。实测已生效（13 次保存后只剩 3 个 weights）。

##### 进度快照 @ 2026-08-29 12:51（step 27,800 / 77,690 = 35.8%）

各指标按 5000 步分段取均值（`trainer.py:428-437` 的 eval 每次只有 8 个样本，单点噪声极大，必须看分段均值）：

| 区间 | train loss | loss_action | loss_video | val_loss | psnr | ssim | action_l2 | action_l1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–5,000 | 0.5021 | 0.1893 | 0.3127 | 0.4237 | 23.73 | 0.7389 | 0.0456 | 0.1456 |
| 5,001–10,000 | 0.3977 | 0.1114 | 0.2863 | 0.4175 | 23.83 | 0.7363 | 0.0288 | 0.1089 |
| 10,001–15,000 | 0.3693 | 0.0892 | 0.2802 | 0.3946 | 23.78 | 0.7339 | 0.0310 | 0.1053 |
| 15,001–20,000 | 0.3473 | 0.0767 | 0.2706 | 0.4271 | 23.61 | 0.7398 | 0.0303 | 0.1018 |
| 20,001–25,000 | 0.3223 | 0.0649 | 0.2574 | 0.3917 | 23.78 | 0.7391 | 0.0226 | 0.0876 |
| 25,001– | 0.2991 | 0.0562 | 0.2430 | 0.3762 | 24.40 | 0.7485 | 0.0257 | 0.0861 |

**全部单调改善，未见平台期。** 关键观察：

- `loss_action` 降了 **77%**（0.238 → 0.056），而 `loss_video` 只降 22%。符合预期 ——
  Wan2.2 本身已有很强的视频先验，模型主要在学动作头；对以动作预测为目的的 uncond 变体是好现象。
- `action_l2` 首 10 点均值 0.0456 → 末 10 点 0.0234（**改善 49%**），历史最好 0.0099 @ step 24500。
- ⚠️ **待观察**：train loss 0.299 vs val_loss 0.376 已有间隙。当前只跑到 epoch 1/5，
  且 val_loss 每次只采 8 个样本、噪声很大，暂不构成过拟合结论，但后 3 个 epoch 需盯住
  val_loss 是否转升。

健康检查全绿：`Error processing sample` 0 / `nan` 0 / OOM 0 / Traceback 0；
8 卡 util 75–100%；磁盘余 805 G。

> 排查提示：`nvidia-smi` 单次采样偶尔会显示某张卡 0% util（实测 GPU 6 连采 6 次里有 3 次为 0）。
> 这是瞬时采样落在 kernel 间隙，不是掉卡 —— 真有 rank 卡死，NCCL 集合通信会让整个 job 停住，
> 而 step 一直稳定在 0.45 step/s。

---

### 2026-08-31 — ✅ final_A 训练完成

| 项 | 值 |
|---|---|
| 起止 | 2026-08-28 19:38:41 → **2026-08-31 00:37:27** |
| 总时长 | **53.0 小时**（2 天 5 小时），1466 步/小时 |
| 步数 | **77,690 / 77,690**（5 epoch）完成，日志有 `max_steps reached` |
| 健康 | `Error processing sample` 0 / nan 0 / OOM 0 / Traceback 0 —— **全程零异常** |
| 最终权重 | `runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_077690.pt`（12 GB） |
| 归一化统计 | `runs/agilex_final_3cam_384_1e-4/final_A/dataset_stats.json`（部署必需） |
| eval 视频 | 1240 个 mp4 |
| wandb | https://wandb.ai/menggao6073-geek-/fastwam-agilex/runs/8f4gqlmo |

#### 按 epoch 的指标演化（155 个 eval 点 / 1553 个 train 点）

> ⚠️ **指标口径更正**：本文件所有表格里的 `psnr` / `ssim` 列都取自 `[eval]` 日志行，
> 而日志打印的是 **`psnr_rd` / `ssim_rd`**（`trainer.py:789`）= **pred vs VAE重建**，
> **不是 vs 真值**。它衡量「扣掉 VAE 损失后的纯扩散质量」。
> 端到端质量是 `psnr_rg` / `ssim_rg`，VAE 天花板是 `psnr_dg` / `ssim_dg`，
> 这 6 个只在 wandb 里有（`trainer.py:800-810`）。

| | 区间 | train loss | loss_action | loss_video | val_loss | psnr | ssim | action_l2 | action_l1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ep1 | 1–15,538 | 0.4211 | 0.1285 | 0.2926 | 0.4115 | 23.79 | 0.7367 | 0.0351 | 0.1198 |
| ep2 | 15,539–31,076 | 0.3192 | 0.0634 | 0.2558 | **0.3978** | 23.98 | 0.7442 | 0.0250 | 0.0901 |
| ep3 | 31,077–46,614 | 0.2665 | 0.0369 | 0.2296 | 0.4268 | 24.39 | 0.7503 | 0.0232 | 0.0855 |
| ep4 | 46,615–62,152 | 0.2277 | 0.0205 | 0.2072 | 0.5250 | 24.07 | 0.7467 | 0.0232 | 0.0850 |
| ep5 | 62,153–77,690 | **0.2046** | **0.0124** | **0.1922** | 0.6101 | 24.19 | 0.7493 | 0.0226 | 0.0841 |

首尾对比：train loss −64%，**loss_action −95%**，loss_video −41%，action_l2 −46%，action_l1 −40%，
但 **val_loss +52%（变差）**。

#### 结论：已收敛，且从 epoch 3 起出现过拟合

1. **训练侧完全收敛**：末 10,000 步 train loss 0.2039/0.2058/0.2047/0.2015、
   loss_action 0.0125/0.0121/0.0116/0.0114 —— 完全平台，无下降空间。
2. **验证 loss 在 step≈32,000（epoch 2 末）触底 0.3731，随后单调上升到 0.675** ——
   train 继续降、val 持续升，是标准过拟合信号。
3. **但任务相关指标没有崩**：`action_l2` 在 step≈26,000–30,000 后进入 0.020–0.026 的噪声带，
   之后既没显著改善也没退化；psnr/ssim 也稳定在 24.0–24.4 / 0.745–0.759。
   所以过拟合主要发生在**视频分支**（train loss_video 一路降到 0.19 是在记忆训练视频），
   动作分支只是**停止进步**，不是变差。
4. **算力效率**：动作指标约在 **18 小时 / epoch 2** 就到位，剩下 35 小时（约 2/3 算力）
   对 action_l2 几乎没有贡献。下次同类实验 **2–3 epoch 就够**。

#### 各 checkpoint 的窗口均值（±2500 步，压掉 8 样本单点噪声）

| checkpoint | val_loss | action_l2 | action_l1 | psnr | ssim | 备注 |
|---|---:|---:|---:|---:|---:|---|
| 30,000 | 0.3789 | 0.0218 | 0.0822 | **24.76** | **0.7593** | **视频质量最好，val_loss 接近最优，动作在最优带内** |
| 32,000 | **0.3731** | 0.0238 | 0.0854 | 24.65 | 0.7567 | val_loss 最优 |
| 42,000 | 0.4214 | 0.0205 | 0.0824 | 24.47 | 0.7507 | 动作较优、val_loss 尚可 |
| 68,000 | 0.5755 | **0.0174** | **0.0764** | 24.25 | 0.7493 | action_l2 最低，但 val_loss 已明显恶化 |
| **77,690（最终）** | 0.6751 | 0.0256 | 0.0876 | 23.98 | 0.7434 | val_loss 比最优差 81%，action_l2 差 47% |

**推荐用 step_030000.pt 作为部署首选**（三项指标同时接近或达到最优）；
若纯做 action-only 推理、且愿意接受更嘈杂的证据，可对比 step_042000 / step_068000。

> ⚠️ 证据强度说明：验证集只有 **5 个 episode**（`val_set_proportion=0.01`），
> 且每个 eval 点只采 **8 个窗口**。上表的窗口均值已压掉大部分噪声，但仍属**弱证据**。
> 最终 checkpoint 选择应以真机 rollout（或扩大验证集重跑离线评测）为准。

#### 幸运之处：全部 39 个 checkpoint 都在 OSS 上

本地 `prune_ckpt.sh` 只留最近 3 份（step 74000/76000/77690，全在过拟合区间），
但 `autosync_daemon.sh` 每 30 分钟同步一次、**总在 prune 之前跑过**，所以
`/mnt/data/gaomeng/FastWAM/runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/`
完整保留了 **step_002000 ~ step_077690 共 39 个**（约 458 GB）。
推荐的 step_030000 可直接从 OSS 取回。

> 这是个意外之喜而非设计。若下次仍要做 checkpoint 选择，应显式保留若干「里程碑」权重，
> 不要只依赖 OSS 同步与 prune 的时序巧合。

#### Phase 2 验收结果

| 产物 | 体积 | 说明 |
|---|---|---|
| `checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` | 9.83 + 10.0 + 0.18 GB | Wan2.2 DiT |
| `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors` | 11.36 GB | T5 text encoder |
| `checkpoints/DiffSynth-Studio/.../Wan2.2_VAE.safetensors` | 1.41 GB | VAE |
| `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` | 小 | tokenizer（4 个文件） |
| `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | **2.04 GB** | ActionDiT backbone |
| `checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt` | **12.04 GB** | B 实验热启动权重 |

合计约 47 GB，全部落在项目内（约束 6）。

**backbone 生成改在 CPU 上完成**（GPU 被占满，见下）。这是合理的自主处置 —— 该步骤是纯权重插值，不需要 GPU：
```
[INFO] Saved ActionDiT backbone payload ... (copied=300, interpolated=520, skipped=4)
```
`skipped=4` 正是 `ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")` 那些与 action_dim 相关的层
（`action_dit.py:32`），所以脚本 `[WARN] action_dim ... defaulting to 7` 无害。CPU 耗时 94 秒。

#### Phase 4 验收结果

`data/text_embeds_cache/agilex/36a916de41f5...t5_len128.wan22ti2v5b.pt`（1.05 MB，`new=1`）。
hash 与数据集先前索取的完全一致。同样因 GPU 占满而**用 `CUDA_VISIBLE_DEVICES="" ` 走 CPU**，1 个 prompt 耗时 43 秒。

#### 数据管线端到端验证（无需 GPU，已通过）

```
train len = 994,426    val len = 9,558
video   (3, 9, 384, 320)  ∈ [-1, 1]      <- robotwin 三相机拼图正确
action  (32, 14)          z-score 后 ~[-2.6, +1.9]
proprio (32, 14)
context (128, 4096)                       <- T5 embedding
collate batch=4 -> video(4,3,9,384,320) / action(4,32,14) / context(4,128,4096)
无 NaN / Inf
```

**实测取样耗时：平均 675 ms/样本**（单进程串行，含 torchcodec 解码 + resize + 拼图 + 归一化 + parquet 查询），
比纯 ffmpeg 解码推算的 240 ms 高 2.8 倍。换算：8 workers → ~11.9 样本/s/进程 → bs=8 约 **0.67 s/step 的供给能力**。
对 5B 模型的 step 时间（预计 1.5–3 s）应仍有余量，但比原估计紧，Phase 5d 需实测确认。

---

### ⛔ 阻塞：8 张 GPU 被 vLLM 推理服务占满

**发现时间**：2026-08-28 10:55（Phase 2a 首次尝试时 CUDA OOM）

```
nvidia-smi: 每卡 74989 MiB / 81920 MiB 已用，仅剩 6165 MiB，utilization 0%
```

占用者是**本容器内**的一个 vLLM 服务：

```
/root/qwen38_27b/venv_v019/bin/vllm serve /root/qwen38_27b/model \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 18080 \
  --tensor-parallel-size 8 --kv-cache-memory-bytes 67108087296 \
  --max-num-seqs 320 --enable-prefix-caching --dtype bfloat16
```

- `--tensor-parallel-size 8` → 占用**全部 8 张卡**
- `--kv-cache-memory-bytes 67108087296` → 显式钉住 **62.5 GiB KV cache / 卡**，加上 27B 模型分片 ~6.8 GB ≈ 73 GiB，与 OOM 报告的 `73.22 GiB` 完全吻合
- 已运行 2 小时以上，`utilization 0%`（空载但显存不释放，vLLM 预分配特性）

另外容器内还有另一位用户的 FastWAM 任务：`/root/litianyu/FastWAM`（已运行 ~1 天 22 小时），
其 `runs/` 下已有 `agilex_giga_542_3cam_384_1e-4`、`agilex_uncond_3cam_384_1e-4` 等目录。

**为什么这是硬阻塞**：剩余 6.1 GB/卡 远不足以训练 5B 模型（MoT 权重 bf16 约 12 GB，再加 VAE 1.4 GB、
优化器状态、激活值）。即使 bs=1 + 梯度检查点也放不下。

**为什么我不自行处置**：`vllm serve` 是监听 `0.0.0.0:18080` 的**对外推理服务**，可能有其他系统依赖它；
`litianyu` 的任务属于另一位用户。终止它们属于破坏性且对外的操作，超出「自己解决问题」的范围，需要你决定。

#### 处置决定（用户已确认）

**保持守候，等 GPU 自然空出。不终止 vLLM，不动 litianyu 的任务。**

已建立 durable cron 守候任务 `f3315d67`（`.claude/scheduled_tasks.json`）：

| 项 | 值 |
|---|---|
| 频率 | 每小时 `:07` 和 `:37`（每天 48 次） |
| 触发条件 | **每张卡空闲显存 ≥ 70000 MiB** |
| 满足时的动作 | 自动继续 Phase 5 → 6 → 7，并自删该 cron |
| 不满足时 | noop，只报当前空闲显存，不做别的 |
| 有效期 | **7 天后自动过期**（会最后触发一次）；若届时仍未空出，我会重建 |

所以你在任意时刻停掉 vLLM，最迟 30 分钟内训练就会自动启动，无需再通知我。

#### 就绪状态清单（GPU 一空出即可直接开跑）

| 项 | 状态 |
|---|---|
| conda 环境 `fastwam` | ✅ 全部钉死版本，torch 见到 8 卡 |
| 数据（改名后） | ✅ `data/agilex_empty_the_box_fastwam`，471 eps |
| Wan 底座权重 | ✅ DiT + T5 + VAE + tokenizer（~35 GB） |
| ActionDiT backbone | ✅ 2.04 GB |
| RoboTwin 热启动权重（B 用） | ✅ 12.04 GB |
| data / task 配置 | ✅ hydra compose 验证通过 |
| T5 文本缓存 | ✅ 1 个 prompt |
| `dataset_stats.json`（A/B 共用） | ✅ `runs/_shared/` |
| 数据管线 | ✅ 真实取样 + collate 全通过 |
| OSS 持久镜像 | ✅ 72 GB，已校验 |
| **唯一缺口** | **GPU 显存** |

---

## 目录布局（混合方案）

| | 执行目录 | 持久镜像 |
|---|---|---|
| 路径 | `/home/gaomeng/FastWAM` | `/mnt/data/gaomeng/FastWAM` |
| 文件系统 | overlay（容器 rootfs） | **ossfs2**（阿里云 OSS 对象存储 FUSE） |
| 速度 | 快 | 顺序写 149 MB/s；随机读实测慢 1.3× |
| 软链接 / 硬链接 | ✅ 支持 | ❌ **都不支持**（对象存储无 inode） |
| 持久性 | 容器重建可能丢 | 持久 |
| 存放 | 全部（含数据、runs、缓存） | 代码/配置/文档/权重/原始数据备份 |

### 为什么执行留在 `/home` 而不是直接在 OSS 上跑

1. **数据集随机读** —— dataloader 每样本 99 次 seek 解码。OSS 实测只慢 1.3×，但那是**反复读同一文件**、命中 ossfs2 缓存的乐观值；真实训练随机读 1413 个视频文件（28 GB），命中率低得多。
2. **checkpoint 写入** —— weights 约 12 GB、ZeRO state 更大，8 个 rank 并发写 FUSE。
3. **文本 embedding 缓存** —— `robot_video_dataset.py:261` **每个样本**都 `torch.load` 一次该文件，不适合走 FUSE。
4. **软链接** —— 转换后数据集的 3 个相机目录是软链接，OSS 建不了（这正是 `cp -r` 报 `Operation not supported` 的原因）。

### 同步脚本

`scripts/sync_to_oss.sh`（rsync 增量，**刻意不加 `-l/-a`** 以避免尝试创建软链接）：

```bash
bash scripts/sync_to_oss.sh                  # 全量同步（不含 ZeRO state）
bash scripts/sync_to_oss.sh --changing-only  # 只同步会变的部分（定时任务用）
bash scripts/sync_to_oss.sh --code-only      # 只同步代码/配置/文档（秒级）
bash scripts/sync_to_oss.sh --with-state     # 连 ZeRO state 一起（80GB/份，很慢）
bash scripts/sync_to_oss.sh --dry-run        # 只看会传什么
```

排除项：`data/agilex_empty_the_box_fastwam/videos/`（全是软链接，可重建）、`.cache/`、
`__pycache__/`、`runs/**/wandb/`、默认还排除 `runs/**/checkpoints/state/`。

**`--changing-only` 为什么必要**：`checkpoints/`（44 GB）和原始数据集（28 GB）是**静态**的，
同步过一次就不再变。但 rsync 每轮仍要 stat 全部 72 GB，在 OSS FUSE 上单次要好几分钟。
定时任务用 `--changing-only` 跳过这两块后，单轮降到 ~2 分钟（且后续更快）。

### 自动同步守护（已启用）

```bash
bash scripts/autosync_daemon.sh start [间隔秒]   # 默认 1800s = 30 分钟
bash scripts/autosync_daemon.sh status
bash scripts/autosync_daemon.sh stop
bash scripts/autosync_daemon.sh once             # 立刻前台同步一次
```

设计取舍：用**脱离终端的后台循环**（`setsid nohup`，PPID=1）而不是 Claude 的定时任务 ——
纯机械的 rsync 不需要模型参与，这样零上下文开销，也不依赖会话是否在线。
带 `.autosync.lock` 目录锁，防止上一轮没跑完就叠加（OSS 慢时可能超过一个周期）。

- PID 文件：`.autosync.pid`
- 日志：`.autosync.log`（含每轮起止时间与传输量）
- 当前状态：**已启动，30 分钟一轮**

> ⚠️ 这个守护进程会一直跑，直到你显式 `stop` 或容器重启。它只做 rsync，不会动训练。

### 磁盘是多人共享的（重要运维风险）

2026-08-28 实测 `/`（overlay，4.2 T）上的占用：

| 路径 | 占用 |
|---|---|
| `/root/litianyu` | **1.9 TB**（另一位用户的 FastWAM，有活跃任务） |
| `/root/zzd` + `/home/zzd` | 418 GB + 228 GB |
| `/root/glm53_flash` | 294 GB |
| `/root/qwen38_27b` | 89 GB（vLLM 模型） |
| `/home/gaomeng`（本项目） | 84 GB |
| **可用** | **~970 GB，且别人还在增长**（几小时内观察到减少约 200 GB） |

叠加单次 checkpoint 92 GB，长训练必须主动清理：

```bash
bash scripts/prune_ckpt.sh runs/<task>/<run_id>          # 默认留最近 1 份 state + 3 份 weights
KEEP_STATE=2 KEEP_WEIGHTS=5 bash scripts/prune_ckpt.sh <run_dir>
DRY=1 bash scripts/prune_ckpt.sh <run_dir>               # 只看会删什么
```

按默认策略，长训练的 checkpoint 峰值footprint ≈ 80 + 3×12 = **116 GB**。


> **首次同步为什么传了全部 47 GB**：你之前的 `cp -r` 没有 `-p`，未保留 mtime，rsync 的
> size+mtime 判定认为需要重传。现在 mtime 已对齐，后续同步是真增量。

### 关于 `mv` 与 `cp`

跨设备（overlay → ossfs2）的 `mv` **不是** rename，会退化成"逐字节复制 + 删除源"：

| | `cp -r` | `mv` |
|---|---|---|
| 速度 | 复制全部字节 | **一样**，不会更快 |
| 软链接报错 | 会 | **同样会** |
| 失败后果 | 源完好，可重试 | **边复制边删源**，失败后源被部分删除 |

所以本项目一律用 `rsync`（可重入、可增量），不用 `mv`。

---

## 灾难恢复（容器重建后 `/home` 丢失）

OSS 镜像覆盖 `/home/gaomeng` 下的**全部**工作目录（FastWAM + 同级仓库），恢复只差
"重建 6 个软链接"这一步：

```bash
# 1) 从 OSS 拉回本地（约 510 GB：权重 65G + 原始数据 68G + runs weights ~370G）
#    --size-only 是必须的：ossfs2 不保存 mtime，不加会把全部文件当成"有差异"。
mkdir -p /home/gaomeng
rsync -rt --size-only --no-perms --no-owner --no-group \
  /mnt/data/gaomeng/ /home/gaomeng/

# 2) 重建 conda 环境（见 Phase 0）
conda create -n fastwam python=3.10 -y && conda activate fastwam
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
cd /home/gaomeng/FastWAM && pip install -e .

# 3) 重建两个转换后数据集的视频软链接（唯一未备份的东西，OSS 建不了软链接）
/root/miniforge3/envs/fastwam/bin/python scripts/convert_agilex_to_fastwam.py \
  --src ./data/agilex_empty_the_box_all_470 \
  --dst ./data/agilex_empty_the_box_fastwam
/root/miniforge3/envs/fastwam/bin/python scripts/convert_agilex_to_fastwam.py \
  --src ./data/agilex_empty_the_box_all_542_0711 \
  --dst ./data/agilex_empty_the_box_542_fastwam

# 4) 恢复 wandb 凭证（刻意不同步到 OSS）
#    写入 /home/gaomeng/FastWAM/.wandb_key

# 5) 校验
find data/agilex_empty_the_box_fastwam     -type l | wc -l   # 期望 3
find data/agilex_empty_the_box_542_fastwam -type l | wc -l   # 期望 3
ls data/agilex_empty_the_box_fastwam/videos/chunk-000/*/ | head  # 应能穿透到 mp4
ls /home/gaomeng   # 期望 FastWAM giga-world-policy wam.cpp geekrl openpi GigaWorldPolicy解读
```

**已在 OSS，无需重下/重算**：底座权重（65 GB）、原始数据集 470（28 GB）与
542_0711（40 GB）、`runs/` 下的 weights 与训练日志、`dataset_stats.json`、
T5 文本缓存、`.cache/`、wandb 本地运行目录，以及 5 个同级工作目录。

**不在 OSS，需重建**：ZeRO 续训 state（`runs/**/checkpoints/state/`，240 GB —— 只用于
断点续训，重建后无法从中断处续训，只能从最近的 weights 冷启动）、`.wandb_key`、
转换后数据集的视频软链接、conda 环境、`__pycache__`。


#### Phase 0 验收结果

`conda create -n fastwam python=3.10`（Python 3.10.21），按 README 装完，全部命中钉死版本：

| 包 | 版本 |
|---|---|
| torch | 2.7.1+cu128（`cuda.is_available()=True`，**8 GPUs**） |
| torchvision | 0.22.1+cu128 |
| accelerate / deepspeed | 1.12.0 / 0.18.7 |
| transformers / datasets | 4.49.0 / 4.8.5 |
| pyarrow / torchcodec | 24.0.0 / 0.4.0 |
| wandb | 0.23.1 |
| fastwam | 0.1.0（editable → `/home/gaomeng/FastWAM`） |

`ds_report`：我们需要的算子（`fused_adam`/`cpu_adam`/`transformer_inference` 等）全部 `[NO] ... [OKAY]`（未预编译但兼容，可 JIT）。
`async_io [NO]...[NO]` 无影响 —— 它只用于 NVMe offload，而 `ds_zero1_config.json` 里 `offload_optimizer.device=none`。
另外训练用的是 `torch.optim.AdamW`（`trainer.py:91`），并非 deepspeed 的 FusedAdam。

#### 配置组装验证（hydra compose）

| 项 | 值 |
|---|---|
| images key 顺序 | `cam_high` → `cam_left_wrist` → `cam_right_wrist` ✅（robotwin 拼图顺序敏感） |
| action | `default` 14 → 14 |
| state | `joint` 12 + `gripper_position` 2 = **14** |
| 注入模型 | `proprio_dim=14`、`video_dit.action_dim=14`、`action_dit.action_dim=14` ✅ |
| `delta_action_dim_mask` | **absent** ✅（LIBERO 专用，绝对关节 action 不该有） |
| 时序断言 | `num_frames=33`, `ratio=4` → 32 actions / 9 video frames ✅ |
| `load_text_encoder` | false（训练用缓存 embedding） |
| action scheduler shift | train/infer 均 1.0 |

#### 归一化统计量（A/B 共用，保证完全可控对比）

standalone 预计算到 **`runs/_shared/dataset_stats.json`**（90 KB）：

| 项 | 值 |
|---|---|
| num_episodes | 466（471 × 99%，val 切走 1%） |
| num_transition | **994,426** |
| action / state keys | `action.default` / `state.joint` + `state.gripper_position` |
| action global_std | `[0.213 0.641 0.539 0.431 0.361 0.464 0.332 0.244 0.604 0.547 0.437 0.280 0.381 0.336]` |
| 零方差维度 | **无** ✅ |
| 耗时 | **16.3 s**（比预期的数分钟快得多） |

> **对原计划的改进**：原计划是「先跑 A 生成 stats，再 pin 给 B」。既然 standalone 只需 16s，改为**预先算好、A 和 B 都 pin 同一份**，从第 0 步就完全一致，比原方案更严格。

#### 已验证的一个预期失败

首次拉样本报 `Missing text embedding cache: ...t5_len128.wan22ti2v5b.pt`（`robot_video_dataset.py:256-260`）—— Phase 4 尚未执行，属预期。
顺带实地确认了 `__getitem__` 的**静默兜底**机制（`robot_video_dataset.py:283-292`）：它 catch 异常后换随机样本重试，只有重试也失败才抛出。这正是 Phase 5a 必须 `grep -c "Error processing sample"` 的原因。

#### Phase 1 验收结果

`data/agilex_empty_the_box_fastwam/` 已生成：

| 检查项 | 结果 |
|---|---|
| `info.json` features 改名 | ✅ `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` / `observation.state.gripper_position` / `action` |
| parquet 列名 | ✅ `action`、`observation.state.gripper_position`（`observation.state.joint` / `.end` 原样保留） |
| parquet 数量 | ✅ 471 |
| 每 parquet 行数 | ✅ 1970（episode_000000，与源一致） |
| `episodes_stats.jsonl` | ✅ 471 行，stats 键已改名 |
| 视频软链接 | ✅ 3 个相机目录，`ls` 可穿透到 471 个 mp4 |
| **额外磁盘占用** | ✅ **172 MB**（视频走软链接，未复制 28GB） |
| **源目录只读性** | ✅ `data/agilex_empty_the_box_all_470/meta/info.json` 仍是原字段名 |

转换耗时约 4 秒（471 parquet @ 141 it/s）。

#### 实测数据表

| 项目 | 值 | 来源 |
|---|---|---|
| AV1 单线程解码 | 411 fps | ffmpeg + dav1d 实测 |
| AV1 关键帧间隔 | 2 帧（985 kf / 1970 帧） | ffprobe 实测 |
| 纯解码推算 | ~0.24 s（99 帧） | 推算 |
| **实际取样耗时** | **675 ms/样本**（单进程，含解码+resize+拼图+归一化+parquet） | 实测，比纯解码推算高 2.8× |
| 推算供给能力 | 8 workers → ~11.9 样本/s/进程 → bs=8 约 0.67 s/step | 推算，待 Phase 5d 确认 |
| train / val 长度 | 994,426 / 9,558 | 实测 |
| dataset_stats 计算耗时 | **16.3 s**（466 eps） | 实测 |
| ActionDiT backbone 生成 | 94 s（CPU）；copied=300 / interp=520 / skipped=4 | 实测 |
| 文本缓存生成 | 43 s（CPU，1 prompt） | 实测 |
| 权重总体积 | ~47 GB | 实测 |
| 选定 batch_size | **8**（bs12 开 eval 后只剩 2.2GB 余量，风险不值 18% 吞吐） | Phase 5c |
| 峰值显存 | 76,309 MiB / 81,920（93%），开 eval 瞬时 78,329 | Phase 5c |
| 选定 num_workers | **8**（供给 11.9 样本/s vs 需求 5.55，余量 2.1×） | Phase 5d |
| step 时间 | **1.44 s**（探测）→ 实际稳态 **0.45 step/s = 2.2 s**（含 eval 摊薄） | Phase 5c/7 |
| 模型规模 | video expert 5.00 B + action expert 1.02 B = **6.02 B** | `mot.py:67` |
| weights `.pt` 体积 | **12 GB** | Phase 5b 实测 |
| state 目录体积 | **80 GB**（单次保存合计 92 GB） | Phase 5b 实测 |
| `AB_STEPS` | **5000**（各跑约 2 小时） | Phase 6 |
| **A/B 结论** | **A ≈ B，差异 <1% → 采用 A**（热启动已核实生效，属真实负迁移） | Phase 6 |
| 最终 `max_steps` | **77,690**（= 5 epoch，`num_epochs=5` + `max_steps=null` 推算）≈ 42 h | Phase 7 |
