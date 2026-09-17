# 新会话交接：FastWAM Agilex 训练（已完成）+ 离线评测（已完成）

> 训练的完整过程见 `AGILEX_TRAINING_PLAN.md`（约 700 行，含所有实测数据与踩坑记录）。
> 原理与排查手册见 `AGILEX_FINETUNE_GUIDE.md`。
> **离线评测的设计与踩坑见 `OFFLINE_EVAL_DESIGN.md`**（本文件只放结论）。
> 本文件是**给新会话的最小必读交接**。
> 生成：2026-08-31　|　更新：2026-09-01（补入离线评测结论）

---

## 一句话状态

**训练与离线评测都已完成。** `final_A`（纯 Wan2.2 初始化）跑满 5 epoch / 77,690 步 / 53 小时，
零异常结束。全部 39 个 checkpoint 已在**两套验证集**上评完（见下）：

- **推荐 `step_032000.pt`**（epoch 2.06）。旧推荐 `step_030000` 也在并列区间内，但 032000
  在新验证集上各指标更一致地占优。
- **后期是真实过拟合，动作分支也在退化** —— 不只是视频分支。这一点与上一版交接的判断不同，
  见下方"收敛与 checkpoint 选择"。

下一步是真机部署验证。

wandb：https://wandb.ai/menggao6073-geek-/fastwam-agilex/runs/8f4gqlmo


---

## 环境怎么起

```bash
cd /home/gaomeng/FastWAM
source env.sh                                  # 必须；它会把 conda env 加到 PATH
# python 用 /root/miniforge3/envs/fastwam/bin/python（env.sh 已把它放到 PATH 最前）
```

`env.sh` 已处理四件事，别绕过它：`PATH` 前置（否则 `train_zero1.sh` 调 `accelerate` 会 127）、
`CUDA_HOME`（deepspeed JIT）、`NCCL_DEBUG=WARN`（否则日志被 NCCL 刷爆）、
以及把 HF/ModelScope 缓存关在项目内。

---

## 关键路径

| 用途 | 路径 |
|---|---|
| 训练数据（改名后，实际训练用的，471 episode） | `data/agilex_empty_the_box_fastwam/` |
| 原始数据（**只读，未改动**，列名未转） | `data/agilex_empty_the_box_all_470/` |
| **542 数据集（改名后）** —— 471 + 新增 71 条 | `data/agilex_empty_the_box_542_fastwam/`（视频软链到 `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`，零额外磁盘） |
| 最终 run 目录 | `runs/agilex_final_3cam_384_1e-4/final_A/` |
| 归一化统计（**部署必需**） | `runs/_shared/dataset_stats.json`（= `final_A/dataset_stats.json`，两套评测都用它） |
| 本地权重（只剩 3 份，全在过拟合区间） | `.../checkpoints/weights/step_0{74000,76000,77690}.pt` |
| **全部 39 份权重（OSS 持久化）** | `/mnt/data/gaomeng/FastWAM/runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/` |
| **离线评测结果（旧验证集，5 ep / 64 样本）** | `eval_offline/final_A_sweep/`（`figs/` 12 张图 + `compare.csv`） |
| **离线评测结果（新验证集，71 ep / 284 样本）** | `eval_offline/final_A_newval71/`　← **决策看这个** |
| A/B 对比 run | `runs/agilex_uncond_3cam_384_1e-4/ab_A/`、`ab_B/` |
| 配置 | `configs/data/agilex_3cam.yaml`、`configs/task/agilex_final_3cam_384_1e-4.yaml` |
| **离线评测配置** | `configs/eval/{smoke,final_A_sweep,final_A_newval71}.yaml` |
| Wan 底座 + ActionDiT backbone | `checkpoints/` |

---

## eval 生成了什么

### 1. 视频（`runs/.../final_A/eval/`，1240 个 mp4，共 164 MB）

命名 `step_{步数:06d}_rank_{卡号:03d}.mp4`。155 个 eval 点 × 8 张卡 = 1240 个。
每个 eval 点每张卡**只采 1 个验证样本**（`trainer.py:428-437`，按 `global_step + process_index` 播种）。

规格：**h264 / 320×1152 / 9 帧 / 8 fps**。

`1152 = 3 × 384` —— 是**三个画面竖向拼接**（`trainer.py:563-566`
`torch.cat([pred, vae, gt], dim=2)`）：

| 位置 | 内容 | 作用 |
|---|---|---|
| **上** 1/3 | `pred` —— 模型生成的视频 | 要评价的对象 |
| **中** 1/3 | `vae` —— GT 经 VAE 编码再解码 | **质量天花板**（模型再好也不可能超过它） |
| **下** 1/3 | `gt` —— 真实视频 | 真值 |

而每个 384×320 画面**内部又是三相机拼图**（`robot_video_dataset.py:168-192`）：
上 256×320 = 顶部相机，下 128×320 = 左腕 + 右腕各 128×160 并排。

> 所以一个 mp4 里其实有 3（pred/vae/gt）× 3（三个相机）= 9 个画面。

### 2. 数值指标

`[eval]` **日志行只打印 4 个**：`val_loss` / `infer_psnr` / `infer_ssim` / `action_l2` / `action_l1`。
⚠️ 其中 `infer_psnr` = **`psnr_rd`**（`trainer.py:789`），即 **pred vs VAE重建**，**不是** vs 真值。

**全部 8 个指标只在 wandb 里**（`trainer.py:800-810`）：

| wandb key | 含义 | 怎么用 |
|---|---|---|
| `eval/val_loss` | 验证集扩散 loss | 看过拟合。**本次从 step≈32,000 起持续上升** |
| `eval/action_l2`、`eval/action_l1` | 反归一化后动作误差 | **最重要** —— 任务相关，直接对应控制精度 |
| `eval/psnr_rg`、`eval/ssim_rg` | pred vs **gt** | 端到端视频质量（含 VAE 损失） |
| `eval/psnr_rd`、`eval/ssim_rd` | pred vs **vae重建** | 纯扩散质量（扣掉 VAE 损失），日志里那个 |
| `eval/psnr_dg`、`eval/ssim_dg` | vae重建 vs **gt** | **VAE 天花板**，与训练无关，近似常数基线 |

### 3. 离线批量评测（训练后另跑的，决策看这个）

`scripts/offline_eval.py` + `scripts/visualize_eval.py`，config 驱动。
与训练期 eval **指标定义完全一致**（已实测对齐：同一 step 上离线 `action_l2` = 0.0227
vs 训练日志 ±2500 窗口 0.0246，差 8%），但让**所有 checkpoint 评完全相同的样本**。

```bash
cd /home/gaomeng/FastWAM && source env.sh
python -u scripts/offline_eval.py --config configs/eval/final_A_newval71.yaml   # 39 ckpt × 8 卡，约 50 min
python scripts/visualize_eval.py --sweep eval_offline/final_A_newval71          # 12 张图，纯 CPU 几秒
```

出的 12 张图里，**选 checkpoint 只看这 3 张**：

| 图 | 用途 |
|---|---|
| `12_paired_delta.png` | **逐样本成对差分 + 95% CI**，`n.s.` = 差异不显著。判优劣就看这张 |
| `11_ckpt_table.png` | 39 个 ckpt 全指标排名表（精确读数看 `compare.csv`） |
| `15_angle_matrix.png` | 39 ckpt × 14 维的误差矩阵，关节列单位是**度**，橙框 = 该维最优 |

其余：`10_ckpt_sweep`（指标随训练步 + CI 带）、`14_angle_error_deg`（关节误差换成度 +
随时域增长）、`13_dim_error_by_ckpt`（逐维误差曲线）、`01`–`05`（单 ckpt 详情）。

设计、数据契约、全部踩坑见 **`OFFLINE_EVAL_DESIGN.md`**。

---

## 训练的两个 loss 到底是什么（容易误读，先看这个）

两个都是 **flow-matching 速度场的 MSE**，不是直接回归动作值、也不是像素 loss。
共同流程（`schedulers/scheduler_continuous.py`）：采时刻 σ → `x_σ = (1-σ)x₀ + σε` →
目标是**速度** `ε - x₀` → MSE → 乘 timestep 权重（σ=0.5 为中心的高斯钟形，归一化到均值 1）。

| | `loss_action` | `loss_video` |
|---|---|---|
| 代码 | `fastwam.py:590-601` | `fastwam.py:427, 579-588` |
| 作用空间 | **z-score 归一化后的 14 维动作** | **VAE latent（48 通道）**，不是 RGB 像素 |
| 求均值顺序 | 先对 14 个动作维求均值 → `[B,T]`，按 `action_is_pad` 掩码 → 有效步均值 | 先对 (C,H,W) 求均值 → `[B,T_latent]`，按 `image_is_pad` 掩码 → latent 帧均值 |
| 特殊处理 | — | **第一帧排除**（`fastwam.py:575-577`，它是给定条件） |
| σ 采样 shift | **1.0**（均匀） | **5.0**（偏向高噪声，Wan2.2 视频扩散惯例） |
| λ | 1.0 | 1.0（都是默认值） |

`loss_total = 1.0·loss_video + 1.0·loss_action`

**实测量级（末 20 点）：`loss_video = 0.1914`，`loss_action = 0.0114`。**
→ **video 占总 loss 的 94%**，所以"train loss 降到 0.2016"基本是在描述视频分支，
`val_loss` 转升也主要反映视频过拟合。

> ⚠️ 两个 loss **数值不可直接比较**：量纲不同（latent vs 归一化动作）、σ 采样分布不同。
> λ 都设 1.0 是约定，不代表"权重相等"。

## 时间尺度（部署必须知道）

`num_frames=33`、`global_sample_stride=1`、`action_video_freq_ratio=4`、**fps = 30**：

| 量 | 值 |
|---|---|
| 1 个动作步 | 1/30 s = **33.3 ms** |
| action horizon 32 步 | 第 1 步在 **t=0**（当前帧），第 32 步在 31/30 = **1.033 s** |
| 按 32 个控制周期算时长 | 32/30 = **1.067 s**（`summary.json` 的 `annotate.horizon_s`） |
| 视频 9 帧 | 间隔 4/30 = 133 ms，末帧在 32/30 = 1.067 s（比末动作步多一个周期），真实播放速率 **7.5 fps**（trainer 存的 mp4 是 8 fps，略快） |

**`replan_steps` 不该取满 32。** 实测误差从首步 **1.37°** 涨到末步 **8.55°**（6.2 倍，
见 `eval_offline/*/figs/14_angle_error_deg.png`）。8.5° 关节误差在末端会放大成几厘米，
抓取大概率失败。前 10 步（**0.33 s**）内误差约 ≤ 4°，这是建议的 replan 区间。

---

## 怎么比较不同 eval 结果的好坏

### 原则 1：不要再用训练期 eval 的单点或窗口平均去选 checkpoint

训练期 eval 每个点只有 **8 个样本**，而且**每步换一批随机样本**（`trainer.py:436-437` 按
`global_step + rank` 播种）—— 所以 checkpoint 之间根本不可比。本次 `action_l2` 单点在
**0.0082 ~ 0.0654** 之间跳，相差 8 倍。上一版交接用 ±2500 步窗口平均（88 样本）绕过这个噪声，
那是当时唯一的办法。

**现在有更好的办法：`scripts/offline_eval.py` 让所有 checkpoint 评完全相同的样本**，
于是可以做**逐样本成对差分**（`figs/12_paired_delta.png`）：

- 不配对时 `action_rmse` 的 95% CI 约 ±0.13；**配对后 Δ 的 CI 只有 ±0.003** —— 灵敏度差一个量级
- 判据：Δ 的 CI 不跨 0 才算显著；跨 0 图上标 `n.s.`

训练期 eval 从此只用来**看趋势**（尤其 `val_loss` 有没有转升），不用来排名。

### 原则 2：按指标的优先级排序

1. **`action_l2` / `action_l1`（首要）** —— 这是任务指标。Fast-WAM(uncond) 推理时跳过视频想象，
   直接出动作，所以动作误差最直接对应真机表现。
2. **`val_loss`（看过拟合）** —— 只看趋势有没有转升，不看绝对值。
3. **`psnr_rd` / `ssim_rd`（纯扩散质量）** —— 若将来要用 IDM 模式（先想象未来再动作）才重要。
4. **`psnr_dg` / `ssim_dg`（忽略）** —— VAE 天花板，是常数，不反映训练好坏。
   拿它当"模型变好了"的证据是错的。

### 原则 3：train loss 和 val 指标要一起看

本次的典型形态：train loss 一路降到 0.2046、`loss_action` 降 95%，但 `val_loss` 从
step≈32,000 起升到 0.675。**只看 train loss 会误判为"越训越好"。**

### 原则 4：视频要人眼看，不能只看 PSNR

PSNR/SSIM 对"机械臂位置对不对"不敏感 —— 背景占绝大多数像素，机械臂错位几厘米
PSNR 几乎不变。所以要**打开 mp4 人眼比对**：

```bash
# 挑几个关键 step 的同一张卡，横向对比
ls runs/agilex_final_3cam_384_1e-4/final_A/eval/step_0{30000,42000,68000,77500}_rank_000.mp4
```

看的时候：
- **上 1/3(pred) vs 下 1/3(gt)**：机械臂位置/朝向/夹爪开合对不对，物体有没有被正确抓起
- **上 1/3(pred) vs 中 1/3(vae)**：如果 pred 明显比 vae 差，是扩散没学好；
  如果两者接近，说明已逼近 VAE 天花板，再训视频分支没意义
- 9 帧跨 1.1 秒（33 帧 ÷ 4 抽样，30fps），看动作**连续性**是否合理

### 原则 5：验证集的选择比指标的选择更关键

**这是本次最大的教训。** 两套验证集给出了不同结论：

| | 旧验证集 | 新验证集 |
|---|---|---|
| 来源 | `val_set_proportion=0.01` 划出的 5 个 episode `[36,97,136,449,461]` | 542 数据集新增的 **71 个 episode**（471..541），**从未参与训练** |
| 样本量 | 64 | 284（每 episode 4 个） |
| 能否区分 checkpoint | ❌ 38 个里 **27 个**与最优无显著差异 | ✅ 38 个里 **32 个**显著劣于最优 |
| step_077690 的 rmse | 0.1454 | 0.2009（**高 38%**） |

旧验证集与训练集同分布（同一批采集的不同 episode），区分度已经用尽。新验证集是真正的
held-out，才测出了后期的退化。

**证据链（说明 38% 不是分布偏移造成的）**：几乎没训练的 `step_002000` 在两个集上只差
**3.6%**（0.2138 → 0.2215），而训练好的权重都差 **38%**。没学到东西的模型无从过拟合，
所以这个差值是**真实的泛化差距**。

> ⚠️ 但新增 episode 中位 4235 帧 vs 老数据平均 2131 帧，**长约 2 倍**，所以严格说混了
> "泛化到新数据"和"泛化到更长 episode"两件事。逐 episode 看 rmse 跨度 2.4 倍
> （最好 ep538=0.128 / 最差 ep519=0.309，中位 0.187），分布连续无双峰，不支持
> "新数据是另一个 mode"的解释。

**离线终究只是筛选。** 离线 `action_l2` 低不等于真机成功率高，
**最后必须真机 rollout 定胜负。**

---

## 上一会话得到的结论（供决策，不必重跑）

### A/B：热启动 RoboTwin 权重**无收益**

> ## ⛔ 2026-09-07 更正：**本小节的结论无效,已作废。**
>
> `resume=<文件>` 这条热启动路径当时是**空操作** —— 权重加载发生在
> `accelerator.prepare()` **之后**,而 DeepSpeed ZeRO-1 会在每个 optimizer step 用引擎
> 初始化时的 fp32 master 副本覆盖回 bf16 参数,所以热启动权重只活过第 1 次前向。
>
> 指纹:`ab_B` 加载的是**训练完整的 FastWAM**,step 10 的 `loss_action` 却是 1.9475,
> 与**随机初始化**的 ab_A(1.9514)只差 0.2%。修复后重跑同一份权重,`loss_action`
> 变成 3.25~4.38。
>
> 下面「已核实热启动确实生效」那段核实的是 key 名匹配与日志行,**没有验证权重活到
> 第 1 个 step 之后**,所以它证明不了想证明的事。
>
> bug 详情、证据链、修复与永久守卫见 **`WANPRETRAIN_TRANSFER_PLAN.md` §3**。
> `ab_C1` / `ab_C2`(GWP-0.5 迁移)同样作废。**`final_A` 不受影响**(它 `resume=null`)。
> 要重新回答"RoboTwin / GWP 热启动有没有用",必须用修复后的代码重跑。

各跑 5000 步，除 `resume` 外全同。`action_l2` 后 10 点均值 A=0.02848 / B=0.02856（差 0.28%），
train loss 差 4 位小数级 → **用 A**。

已核实热启动**真的生效**（key 名 1649/1649 全匹配、`proprio_encoder` 形状一致、日志有
`Loading weight checkpoint only`），所以这是**真实的负迁移**，不是 bug。
原因推测：归一化基准不同（RoboTwin 用自己的关节范围做 z-score）+ sim→real + 本体不同。

### 收敛与 checkpoint 选择（已按离线评测更新，结论变了）

全部 39 个 checkpoint × **新验证集**（71 个 held-out episode / 284 个相同样本 / joint 模式）。
以 `action_rmse` 最优的 `step_028000` 为基准做逐样本成对差分，**只有下面 6 个与它无显著差异
（并列第一），其余 32 个都显著劣于它**：

| checkpoint | epoch | action_rmse | Δ vs 028000 | 12 关节平均 |
|---|---:|---:|---:|---:|
| step_024000 | 1.54 | 0.1966 | +0.0032 ± 0.0037 | 7.38° |
| step_028000 | 1.80 | **0.1934** | 基准 | 7.37° |
| step_030000 | 1.93 | 0.1953 | +0.0019 ± 0.0033 | 7.26° |
| **step_032000** | 2.06 | **0.1934** | **+0.0000 ± 0.0030** | **7.12°** |
| step_038000 | 2.45 | 0.1943 | +0.0008 ± 0.0028 | 7.26° |
| step_048000 | 3.09 | 0.1964 | +0.0030 ± 0.0032 | 7.25° |
| step_052000 | 3.35 | 0.1962 | +0.0028 ± 0.0035 | 7.24° |

**推荐 `step_032000.pt`**：rmse 与最优并列（Δ 恰为 0.0000）、`action_l1`/`action_l2` 上是
**单独最优**、12 关节平均角度误差最低（7.12°）。而且它正好等于上一版用旧验证集窗口平均
定出的区间，两套方法交叉印证。

**⚠️ 与上一版交接的两处结论修正：**

1. **后期不只是视频分支过拟合，动作分支也在退化。** `step_028000` 之后每一个 checkpoint 都
   **显著劣于**它，Δ 稳定在 +0.004~+0.008。旧验证集上看不出来（那里 step_014000 之后全部
   `n.s.`），因为它与训练集同分布。时间上与 `val_loss` 从 step≈32,000 起上升吻合。
2. **最优点比原估计更早**：动作指标在 **epoch 2 左右**（约 21 小时）到位，后 3 个 epoch
   （约 32 小时）**不仅零收益，在真正 held-out 的数据上是负收益**。
   → **同类实验下次 2 epoch 就够。**

不要用这几个指标选 checkpoint：`psnr_rd` / `ssim_rd` / `action_max_abs` 在新验证集上的
"最优"都是 `step_004000` —— 那是个几乎没训练的权重，输出模糊、缺少高频细节，反而更接近
VAE 重建的低频成分。这类指标在训练早期不可靠。

旧验证集（5 episode / 64 样本）的结果保留在 `eval_offline/final_A_sweep/`，
只用于对照"验证集换了会发生什么"，**不要用它做决策**。

---

## 建议的下一步

1. **从 OSS 取回候选权重**（本地已被 prune 掉）
   ```bash
   cd /home/gaomeng/FastWAM && source env.sh
   W=/mnt/data/gaomeng/FastWAM/runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights
   cp $W/step_032000.pt runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/
   ```
2. **人眼比对视频**（离线评测已存好，pred/VAE/GT 竖向拼接，7.5 fps 真实速率）
   ```bash
   ls eval_offline/final_A_newval71/ckpt/step_0{28000,32000,077690}/videos/ | head
   ```
3. **真机部署**：参照 `experiments/robotwin/fastwam_policy/deploy_policy.py` 写 wrapper。
   - proprio 必须用**训练时同一套定义**（`[joint 12 维, gripper_position 2 维]`，夹爪是 0–1 百分比）
   - 必须带上 `runs/_shared/dataset_stats.json`
   - **`replan_steps` 建议 ≤ 10 步（0.33 s）**，依据见上方"时间尺度"
   - 部署走 `infer_action`（跳过视频想象）。注意 joint 模式下模型自带的交叉校验会报
     `infer_joint 与 infer_action 差 0.0156`（归一化空间），两条路径不完全等价 ——
     离线 joint 模式的动作指标取的是 joint 通道，与部署路径有这个量级的差异
4. **（可选）继续扩大验证集**：`configs/eval/final_A_newval71.yaml` 里
   `dataset.episodes` 可以随意改。想更严格就再采一批新数据当验证集
5. **重跑离线评测的方法**见 `OFFLINE_EVAL_DESIGN.md §5`；中断了用 `--skip-done` 续跑


---

## 已知坑（别重复踩）

| 坑 | 说明 |
|---|---|
| `mot_checkpoint_mixed_attn=true` | **不可用**，`mot.py:28-31` 直接抛错。本实现不支持梯度检查点。OOM 请降 batch_size |
| ZeRO-2 省显存 | **无效**，实测 bs=16 在 zero1/zero2 都 OOM —— 瓶颈是激活值不是优化器状态 |
| 单次保存 92 GB | weights 12 + ZeRO state 80。必须跑 `scripts/prune_ckpt.sh`，否则 38 次 = 3.5 TB |
| `grep "\[train\]"` 抓不到日志 | rich 会折行重排前缀。要先 `sed 's/\x1b\[[0-9;]*m//g'`，再 `tr -d '\n'` 后按 `step=` 切分 |
| `Error processing sample` | `robot_video_dataset.py:283-292` 会静默换随机样本继续。配置错了不一定崩，必须 grep 计数 |
| `nvidia-smi` 偶显 0% util | 瞬时采样落在 kernel 间隙，不是掉卡。看 step 是否推进即可 |
| **多卡跑推理不限 CPU 线程** | 8 进程各起 80 个 torch 线程，80 核机器 load average 冲到 265，每样本 1.55s → **16s（慢 12 倍）**。瓶颈在 CPU 侧（PIL 转换 / PSNR / h264）。`offline_eval.py` 的 launcher 已自动限成 总核数/卡数 |
| **子进程日志一直是空的** | stdout 重定向到文件时 Python 按块缓冲。`offline_eval.py` 已强制 `PYTHONUNBUFFERED=1`；launcher 自己要用 `python -u` |
| **换数据集必须先转列名** | `agilex_empty_the_box_all_*` 用的是原始列名（`actions` / `observation.image.top`…），FastWAM 要 `action` / `observation.images.cam_high`。跑 `scripts/convert_agilex_to_fastwam.py`（7 秒，视频只软链接）。`offline_eval.py` 的 preflight 会在 2 秒内报出缺哪些列 |
| **`load_state_dict(strict=False)` 静默丢权重** | `fastwam.py:1214`，键名不匹配会被忽略，跑出来是随机初始化的分数还看着挺"正常"。本模型应为 **1649/1649**；`offline_eval.py` 已加校验，不匹配直接退出 |
| 磁盘多人共享 | 现余约 800 G，但 `/root/litianyu`、`/root/qwen38_27b` 等其他用户也在用 |
| GPU 可能被别人占 | 上一会话就遇到 vLLM（`--tensor-parallel-size 8`）占满 8 卡。**不要擅自 kill 别人的服务** |

---

## 仍在后台运行的东西

`scripts/autosync_daemon.sh`（PID 见 `.autosync.pid`）每 30 分钟把工作目录增量同步到
`/mnt/data/gaomeng/FastWAM`。训练已结束，**这个可以停掉**：

```bash
bash scripts/autosync_daemon.sh stop
```

训练守护 `run_final.sh` 已随训练结束自行退出。GPU 守候 cron 已删除。
