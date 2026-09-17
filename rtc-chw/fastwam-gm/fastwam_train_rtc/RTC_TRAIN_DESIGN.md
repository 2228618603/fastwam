# Training-Time RTC 实现记录

> 论文解读见 `RTCtrain_论文解读.md`（arXiv:2512.05964）。
> 训练与离线评测的既有结论见 `NEXT_SESSION.md`；真机部署见 `DEPLOY_DESIGN.md`。
> 本文件只记录 **RTC 加装的实现、实测数字与操作步骤**。
> 生成：2026-09-01

---

## 一句话状态

**训练侧与推理侧已实现并全部验证通过，尚未正式训练。**
`model.rtc.enabled=false`（默认）时训练与推理**逐比特不变**，已用
`scripts/rtc_regression_check.py` 证实。下一步是跑 `task=agilex_rtc_3cam_384_1e-4` 微调。

---

## 1. 两个必须先搞清的前提

### 1.1 本仓库的 flow matching 约定与论文相反

`WanContinuousFlowMatchScheduler`（`schedulers/scheduler_continuous.py`）：

```
sigma = timestep / num_train_timesteps
add_noise = (1-σ)·A + σ·ε          target = ε - A
推理：build_inference_schedule 让 σ 从 1 递减到 0，deltas 为负
```

> **σ = 0 是干净数据，σ = 1 是纯噪声。论文的「前缀 τ=1」在这里实现为 `timestep = 0`。**

好处：`add_noise` 在 σ=0 处自动给出 `1·A + 0·ε = A`，**干净前缀是自动涌现的，无需特判**——
与论文解读 §5.1 强调的自洽性一致。

### 1.2 per-token timestep 的下游支持本来就有

| 位置 | 已有机制 |
|---|---|
| `wan_video_dit.py` `DiTBlock.forward` | `has_seq = len(t_mod.shape) == 4` → 逐 token 切 adaLN 调制 |
| `mot.py` `MoT._split_modulation` | 同一个 `has_seq` 分支 |
| `wan_video_dit.py` `pre_dit` | video expert **已在用**：`token_timesteps[:, 0, :] = 0` 把第一帧标成干净条件帧 |

所以论文的「改动 1」只需要改 `ActionDiT`，**`DiTBlock` / `MoT` 一行未动**。
RTC **不增加任何参数**（`time_embedding` / `time_projection` 完全共享，只是 broadcast
维度从 `(B,6,C)` 变 `(B,H,6,C)`），因此可直接在既有 checkpoint 上微调加装。

---

## 2. 改了什么

| 文件 | 改动 |
|---|---|
| `schedulers/scheduler_continuous.py` | `add_noise` 广播改为**尾部补维**，同时支持 `[B]` 与 `[B,H]` timestep；`[B]` 路径与旧实现逐比特一致 |
| `models/wan22/action_dit.py` | 新增 `_embed_timestep`，`prepare()` / `pre_dit()` 接受 `[B]` 或 `[B,H]` timestep |
| `models/wan22/rtc.py` | **新建**：`RTCConfig` + 延迟采样 / 前缀掩码 / timestep 置零 / clamp / 校验 |
| `models/wan22/fastwam.py` | `rtc` 配置接入；抽出 `_build_action_flow_inputs` / `_compute_action_loss` / `_prepare_action_prefix`；`infer_action` / `infer_joint` 新增 `action_prefix` + `delay` |
| `models/wan22/fastwam_idm.py` | `training_loss` 改用同一组 helper（消掉与 `fastwam.py` 逐字重复的那段，IDM 顺带获得 RTC） |
| `runtime.py` | 四个 `create_fastwam*` 透传 `rtc` |
| `configs/model/fastwam.yaml` | 追加 `rtc:` 块，默认全关 |
| `configs/task/agilex_rtc_3cam_384_1e-4.yaml` | **新建**：微调配置 |
| `configs/sim_agilex.yaml` | **新建**：延迟基准用 |
| `scripts/dryrun_fastwam.py` | 补 p95/p99 + 换算成控制步的 `max_delay` 建议；补 `DRYRUN.delay` 走前缀路径 |
| `scripts/rtc_regression_check.py` | **新建**：零回归校验 |
| `scripts/rtc_selftest.py` | **新建**：纯函数 + 数值性质校验（无需 GPU） |
| `scripts/rtc_e2e_check.py` | **新建**：真实模型上的端到端校验 |
| `scripts/rtc_seam_eval.py` | **新建**：离线接缝评测 |

**未改**：`wan_video_dit.py`、`mot.py`、`trainer.py`、`offline_eval.py`、`deploy_real.py`。

### 关键实现细节（照抄容易踩的三点）

1. **loss 归一化用 `sum / sum(mask)` 而非 `mean`**，否则 d 大的样本因有效 token 少而被稀释。
   `rtc_selftest.py` 第 [7] 组用一个反例证明了这个选择是必要的（d=24 用 `sum/H` 会被稀释到
   d=0 的 25%）。
2. **loss 权重用逐样本的 `base_timestep`（`[B]`）而非逐 token 版本。** 后缀 token 共享同一个
   采样 τ，掩码后二者等价，但用 `[B]` 让 RTC 关闭时语义与原实现完全不变。
3. **推理时 clamp 要做两次**：循环内每步、模型调用**之前**；以及循环**结束后**再一次。
   最后一次 `step()` 也更新了前缀位置，不补这一下返回值的前缀就是「脏」的
   （论文解读 §5.2 第 3 点点名的经典坑）。

---

## 3. 实测数字

### 3.1 延迟基准（`scripts/dryrun_fastwam.py --config-name sim_agilex`）

A800-80G，10 步去噪，compiled，n=60：

| delay | mean | median | p95 | p99 | peak mem |
|---|---|---|---|---|---|
| 0 | 131.3 ms | 118.4 ms | 192.2 ms | 203.1 ms | 12.851 GiB |
| 5 | 128.6 ms | 118.3 ms | 198.9 ms | 213.0 ms | 12.851 GiB |

- **前缀路径零额外开销**（128.6 vs 131.3 ms，在噪声内；显存完全相同）。这正是论文的核心卖点：
  去掉了 pseudoinverse guidance 的每步 VJP，循环里只有 forward。
- 30 Hz 下 d(mean)=3.9、d(p99)=6.1 → 脚本建议 `max_delay = 11~14`，**当前默认 12 落在带内**。
- 12.85 GiB 峰值 → 4090 的 24 GB 装得下（前提 `load_text_encoder: false` + 预计算 text embeds）。

> ⚠️ **以上是 A800 的数字。部署机是 4090，必须在 4090 上重跑一遍再定 `max_delay`。**

### 3.2 naive async 基线（`scripts/rtc_seam_eval.py`）

非 RTC 的 `final_A/step_077690.pt`，val split，4 样本 x 4 步去噪（仅冒烟，样本量不足以下结论）：

| d | seam_accel | traj_accel_median | **seam_ratio** | postfix L1 | prefix_err |
|---|---|---|---|---|---|
| 0 | 50.51 | 3.46 | **15.70** | 0.131 | 0 |
| 4 | 40.41 | 3.47 | **12.15** | 0.157 | 0 |

- **`seam_ratio ≈ 15.7`**：naive async 拼接处的加速度是该轨迹正常 jerk 水平的约 **15 倍**。
  这就是 RTC 要解决的那个「chunk 间可见抖动」，现在有了可量化的基线。
- d=4 降到 12.1 —— 光靠 clamp 就有 C⁰ 连续的好处，但这个 checkpoint 没训过前缀条件，
  所以这只是参照，不是 RTC 的表现。
- `prefix_err = 0` 精确成立 → clamp 两处都生效。

### 3.3 RTC 训练冒烟（60 步，4xA800，bs=2）

从 `final_A/step_032000.pt` 热启动（确认走的是「只加载权重」分支）：

```
step   5  loss=2.764  action=2.2552  mean_delay=4.75
step  20  loss=1.565  action=1.2400  mean_delay=7.00
step  40  loss=2.168  action=1.8397  mean_delay=6.25
step  60  loss=1.223  action=0.7969  mean_delay=5.63
```

`rtc_mean_delay` 12 次记录的均值 **5.875**，uniform[0,12) 期望 5.5（每点仅 8 样本，
方差本就大；n=96 时 SEM≈0.35，差距约 1 个 SEM）。`loss_action` 2.26 → 0.80。

---

## 4. 怎么跑

```bash
source env.sh   # 必须

# ── 0. 校验（改完代码随时可跑，都很快）
python scripts/rtc_selftest.py                                    # 38 项，无需 GPU
CUDA_VISIBLE_DEVICES=0 python scripts/rtc_regression_check.py \
  task=agilex_final_3cam_384_1e-4 +REGRESSION.record=/tmp/rtc_baseline.json
CUDA_VISIBLE_DEVICES=0 python scripts/rtc_regression_check.py \
  task=agilex_final_3cam_384_1e-4 +REGRESSION.check=/tmp/rtc_baseline.json
CUDA_VISIBLE_DEVICES=0 python scripts/rtc_e2e_check.py \
  task=agilex_rtc_3cam_384_1e-4 resume=null output_dir=/tmp/rtc_e2e

# ── 1. 在 4090 上实测延迟，据此确认 max_delay
python scripts/dryrun_fastwam.py --config-name sim_agilex \
  +DRYRUN.use_random_context=true +DRYRUN.compile_action_infer=true \
  +DRYRUN.warmup=10 +DRYRUN.iters=200
# 若建议值与 configs/task/agilex_rtc_3cam_384_1e-4.yaml 里的 12 不符，改配置

# ── 2. 微调（4000 步 ≈ 2.2 小时 @ 8xA800）
RUN_ID=rtc_A bash scripts/train_zero1.sh 8 task=agilex_rtc_3cam_384_1e-4 \
  wandb.enabled=true wandb.project=fastwam-agilex wandb.name=rtc_A
# ⚠️ 每次保存 ≈92GB x 8 份，必须配 scripts/prune_ckpt.sh 清理

# ── 3. 接缝评测：RTC 各 ckpt vs naive async 基线
python scripts/rtc_seam_eval.py --task agilex_rtc_3cam_384_1e-4 \
  --ckpt runs/agilex_rtc_3cam_384_1e-4/rtc_A/checkpoints/weights/step_004000.pt \
  --delays 0,2,4,8 --num-samples 32 --out eval_offline/rtc_seam/rtc_A_step004000
python scripts/rtc_seam_eval.py --task agilex_final_3cam_384_1e-4 \
  --ckpt /mnt/data/gaomeng/FastWAM/runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_032000.pt \
  --delays 0 --num-samples 32 --out eval_offline/rtc_seam/final_A_baseline
```

---

## 5. 两个最需要事后调整的数

写在 `configs/task/agilex_rtc_3cam_384_1e-4.yaml` 里，理由也写在那儿：

- **`max_steps: 4000`**。论文仿真是「32 epoch 里最后 8 epoch」= 25% 算力，折算到 final_A 的
  77,690 步是约 19,400 步 —— 但 `NEXT_SESSION.md` 已确认 **final_A 后期是真实过拟合、
  动作分支也在退化**，从 32,000 再走 19,400 会落进退化区，所以取短程。
  **风险在另一侧：太短可能学不动这个新的条件分布。** 判据是 §4 步骤 3 的 d-sweep ——
  若 `seam_ratio` 与 naive async 基线（≈15.7）无明显差别，说明模型在忽略前缀，加到 8,000~12,000。
- **`learning_rate: 2e-5`**（= final_A 峰值的 0.2×）。理由同上：这是「加装一个新的条件分布」，
  不是继续拟合数据。

---

## 6. 已知遗留

- **exposure bias**（论文解读 §9.2，论文本身完全没提）：训练前缀取自数据集 GT，推理前缀来自
  模型自生成，且沿 chunk 链**递归累积**。`rtc.prefix_jitter_std`（默认 0）是最便宜的缓解手段。
  `rtc_seam_eval.py` 刻意用**模型自生成前缀**，就是为了如实暴露这个失配面。
- **丢失 soft masking**（论文 L1）：硬前缀只保证 C⁰ 连续，前缀之后的过渡是学出来的、非显式
  约束的。`seam_ratio` 是唯一判据。
- **`d = 0,1` 时可能极轻微落后**（论文 §6.3）：前缀位置被 mask 掉 loss，前几个动作拿到的监督
  总量少于普通训练。论文称 "very marginally worse"。
- **timestep 用 bf16**：`sample_training_t(dtype=action.dtype)` 而 `action` 是 bf16，
  bf16 在 [512,1024) 的间距是 4，timestep 本就被粗量化。这是**既有行为**，与 RTC 无关；
  前缀值 `0.0` 在 bf16 下精确可表示，不受影响。
- **`rtc_regression_check.py` 必须在建模型之前播种**：`ActionDiT.action_encoder` / `head` 属于
  `ACTION_BACKBONE_SKIP_PREFIXES`，不从 backbone 载入而是随机初始化，不播种两次运行权重不同。
  （真实训练里也是如此 —— `run_training` 建模型时还没播种，`Wan22Trainer.__init__` 才播。
  这不影响训练，但要知道有这回事。）

---

## 7. 还没做：把 RTC 接进 `deploy_real.py --mode async`

`scripts/deploy_real.py` 的 `run_async`（约 939-986 行）**已经是** RTC 那套异步执行框架：
提前发起推理、`elapsed = step_i - fired_at` 就是论文的 `d`、按实际经过步数对齐拼接
（`chunk, idx = filtered, min(elapsed, ...)`）。它缺的只是**前缀条件**——现在是直接切进
第 `elapsed` 步，即论文对照组里的 naive async。

接进来大约 15 行：

1. `FastWAMPolicy.infer_chunk(..., action_prefix=None, delay=0)` → 透传给 `infer_action`。
   `infer_chunk` 已经返回 `action_norm`，正好是前缀需要的归一化空间，不用额外转换。
2. `run_async` 里发起推理时用实测延迟预估一个 `d_pred`（如 p95/median 换算成步数），
   传 `action_prefix = prev_action_norm[idx : idx + d_pred]`、`delay = d_pred`；
   拼接时仍按 `elapsed` 切入。`elapsed <= d_pred` 时切入点落在已承诺的前缀内，完全自洽；
   `elapsed > d_pred` 说明预估偏小，按现有逻辑切入并计数告警。

> 论文相对 inference-time RTC 的一个主要增益正是：**`d` 是运行时输入，不必保守估计**，
> 因为训练时已经见过整个 `d` 分布。所以这里应该传实测值，而不是一个固定的保守常数。
