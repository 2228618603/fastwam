# 用自有具身视频预训练权重替换原生 Wan2.2 —— 计划、一个严重 bug、与执行记录

> 目标 run:`runs/agilex_final_3cam_384_1e-4/final_D`
> 对照 run:`runs/agilex_final_3cam_384_1e-4/final_A`(已完成,5 epoch / 77,690 步 / 53h)
> 预训练权重:`/mnt/data/zzd/DiffSynth-Studio/runs/agilex_emptybox_470_video_pretrain_100k_log100_20260904/step-60000.safetensors`
> 创建:2026-09-07
> 相关文档:`AGILEX_TRAINING_PLAN.md`(final_A 全过程)、`NEXT_SESSION.md`(结论交接)、
> `GWP05_TRANSFER_PLAN.md`(同类迁移的上一次尝试)、`OFFLINE_EVAL_DESIGN.md`(评测口径)

---

## 0. 一句话

把 FastWAM 训练时加载的**原生 Wan2.2 DiT** 换成「在同一批 471 条 Agilex 数据上做过 60k 步视频
预训练的 Wan2.2 DiT」,其余(数据 / 超参 / 归一化统计 / LR 调度 / seed)全部与 `final_A` 逐字相同,
从而与 `final_A` 做**逐步可比**的单变量对照。

**执行过程中发现 `resume=<文件>` 这条热启动路径一直是空操作**,已定位、修复、加了永久守卫,
并且它**作废了之前三次热启动实验的结论**(见 §3)。

---

## 1. 权重可用性核实(全部实测)

| 检查项 | 结果 |
|---|---|
| 张量数 / 命名 | **825 个,key 名与 shape 与官方 Wan2.2-TI2V-5B DiT 逐一相同** |
| `hash_model_file` | `1f5ab7703c6fc803fdded85ff040c316` == 官方 == `WAN22_MODEL_REGISTRY` 注册值 |
| dtype | BF16(官方是 F32;训练本就转 bf16,无损失) |
| 参数量 | 4.9998 B(与 GWP-0.5 那次 video_only 转换的数字一致,结构对得上) |
| 权重漂移 vs 原生 Wan | ndim≥2 同名 cos **0.9807~0.9999**;1-D 最低 0.866;相对差 1.7%~12.7% —— 健康全参微调 |
| 预训练数据 | `agilex_emptybox_470_stitched_30fps`:471 条、h264、**320×384、30fps**、逐条帧长与 `agilex_empty_the_box_fastwam` 完全一致、prompt 相同 |
| 预训练状态 | 仍在跑(`loss.csv` 到 step 65,600 / 目标 100k);本次用 step-60000 |

**关键:拼图版式与分辨率与 FastWAM 训练时逐像素一致**(`concat_multi_camera="robotwin"` 出的就是
384×320)。这是迄今最"对得上"的一次热启动 —— 比 RoboTwin(sim)和 GWP-0.5(异构本体)都好。

### 1.1 这是「同一份数据上的更多算力」,不是「更多数据」

预训练用的就是 FastWAM 训练用的那 471 条。所以本质是一个 curriculum(纯视频预训练 → 视频+动作
联合微调),收益只能来自优化顺序 / 表征质量,不来自新信息。这一点降低了预期上限,与 GWP 那次
"2K 小时额外机器人数据"的论证不同,判读时不要混淆。

---

## 2. 落地机制:FastWAM 零代码改动

把 825 个张量加 `mixtures.video.` 前缀存成 `{"mot": {...}}`,用 task config 的 `resume:` 加载
(`trainer.py` → `FastWAM.load_checkpoint()`,`mot` 用 `strict=False`)。

```bash
python scripts/convert_wan_dit_to_fastwam.py \
  --dit /mnt/data/zzd/DiffSynth-Studio/runs/agilex_emptybox_470_video_pretrain_100k_log100_20260904/step-60000.safetensors \
  --out checkpoints/wanpre60k_video_only.pt
# -> 825 张量 / 4.9998 B / 9.31 GiB / bf16 + .provenance.json
```

`scripts/convert_wan_dit_to_fastwam.py` 复用了 `convert_gwp05_to_fastwam.py` 的
`build_target_manifest`(在 meta device 上构造真实 MoT,不占显存),内建两道门,失败即抛错:

- **V1 覆盖率** 825/825 映射后在真实 `MoT.state_dict()` 中存在且 shape 一致;未填充的
  824 个**恰好**全是 `mixtures.action.*`(本次只迁视觉专家)。
- **V2 血缘余弦** vs 本地原生 Wan2.2 同名 key,对照组是错位配对(block i vs block i+1)。

`action_expert` 保持原来的 ActionDiT 插值初始化(与 `final_A` 完全相同),payload 不含
`proprio_encoder`(`load_checkpoint` 会 warn 并保持当前初始化)—— 所以**只有 video expert 这一个变量**。

### 2.1 V2 判据踩的坑(写下来省得再犯)

第一版判据是「同名配对最低 cos ≥ 0.90」+「同名最低 > 错位最高」,**两条都误判**:

| 现象 | 原因 | 修法 |
|---|---|---|
| `blocks.7.cross_attn.v.bias` 同名 cos 只有 0.866 | bias 范数小,微调时相对位移可以很大(相对差 0.50)。而同一份权重的全部 2-D 矩阵都 ≥ 0.981 | 下限判据只作用于 `ndim >= 2`;1-D 只查中位数 |
| 错位对照最高 0.99972 / `modulation` 错位 0.61~0.86 | RMSNorm 权重各元素都在 1.0 附近,任意两个 cos 都 ~1.0;adaLN 的 `scale_shift_table` 层间天然相关 | 错位判据只作用于 `ndim == 2` 的线性权重矩阵 |

修正后实测:2-D 同名 cos ≥ 0.981、错位中位数 0.0002(最小 −0.003),**最小 gap +0.979**。
层偏移 / 转置 / 张量错配都会掉到 0 附近,判据锐利且离阈值很远。

---

## 3. ⛔ 严重 bug:`resume=<文件>` 的热启动一直是空操作

### 3.1 根因

`src/fastwam/trainer.py` 原本的顺序是:

```
line 170  self.model, ... = self.accelerator.prepare(...)      # DeepSpeed 引擎初始化
line 176  self._resume_or_load_checkpoint()                    # 才加载热启动权重
```

DeepSpeed ZeRO-1/2 在**引擎初始化时**把 bf16 参数 flatten 后 clone 成一份 fp32 master 副本
(`single_partition_of_fp32_groups`),之后**每个 optimizer step 结束都用 master 覆盖回 bf16 参数**。
所以在 `prepare()` 之后才 `load_state_dict` 的权重,只在第 1 次前向里生效,
**第 1 个 step 就被 master 副本静默还原**。

日志里有 `Loading weight checkpoint only: ...`、key 名 1649/1649 全匹配、shape 全对 ——
所有"看得见"的证据都是绿的,唯独权重在第一个 step 之后就没了。

### 3.2 证据链

**(a) 历史数据的指纹。** `ab_B` 加载的 `robotwin_uncond_3cam_384.pt` 是一个**完整训练过的
FastWAM**(含训练好的 action expert + proprio_encoder)。它在 step 10 的 `loss_action` 与
**随机初始化**的 `ab_A` 只差 0.2%:

| step 10 | loss | loss_action | loss_video |
|---|---:|---:|---:|
| ab_A(随机 action 头) | 2.3440 | **1.9514** | 0.3925 |
| ab_B(训练完整的 FastWAM) | 2.3507 | **1.9475** | 0.4031 |

训练好的动作专家不可能恰好落在随机初始化的数值上。三个互不相关的初始化
(RoboTwin / GWP-0.5 video_only / GWP-0.5 full)全都精确复现基线曲线到 <1%,
这不是三次独立的负迁移,是同一个空操作。

**(b) 阳性对照(修复后重跑同一份 RoboTwin 权重)。** `max_steps=999999` 使 lr 只有 1e-8,
几乎零学习,所以这是对加载权重本身的纯读数
(`runs/agilex_uncond_3cam_384_1e-4/_guard_check_robotwin/`):

| | loss_action | loss_video |
|---|---:|---:|
| ab_A 随机初始化 | 1.9514 | 0.3925 |
| ab_B 同一份权重,**修复前** | 1.9475 | 0.4031 |
| **修复后** | **3.25 ~ 4.38** | 0.37 ~ 0.49 |

方向也自洽:RoboTwin 的动作专家在它自己的 z-score 空间里预测,换到 Agilex 的统计量上是
"自信地错",比接近零输出的随机初始化更差。

**(c) 本次目标权重,修复前 vs 修复后**(与 final_A 同步长、同数据顺序对齐):

| | step 50 `loss_video` | step 100 `loss_video` |
|---|---:|---:|
| final_A(原生 Wan2.2) | 0.3905 | 0.4039 |
| final_D **修复前**(作废,`_void_final_D_resume_noop/`) | 0.3886(−0.5%) | 0.4072(+0.8%) |
| final_D **修复后** | **0.3199(−18.1%)** | **0.3424(−15.2%)** |

### 3.3 修复

**权重加载移到 `accelerator.prepare()` 之前**,这样 DeepSpeed 的 fp32 master 是从已经热启动的
权重上 snapshot 的。目录形式的完整状态恢复(`accelerator.load_state()`)仍必须在 `prepare()`
之后,所以两条路径拆开:

| `resume` 形式 | 语义 | 时机 |
|---|---|---|
| **文件** `xxx.pt` | 只加载权重,step 从 0 开始 | `prepare()` **之前** (`_load_weight_checkpoint_before_prepare`) |
| **目录** `state/step_XXXXXX` | 恢复优化器/调度器/step | `prepare()` 之后 (`_resume_or_load_checkpoint`) |

### 3.4 永久守卫(不靠人工记得)

`_load_weight_checkpoint_before_prepare` 加载后对 4 个参数(video/action 各 2 个)拍快照;
第 1 个 optimizer step 之后 `_verify_warmstart_survived()` 核对,相对位移超过 **1e-2** 直接抛错。

阈值依据:1 个 step 的合法位移是 `lr*grad` ~ 1e-5;被 master 副本还原成另一份初始化是 1e-1
量级(实测两份权重相差 1.7%~12.7%)。两者相隔 4 个数量级。

取参数用 `model.get_parameter(路径)` 而不是查 `named_parameters()` —— FastWAM 同时持有
`self.mot` 和 `self.video_expert`/`self.action_expert` 指向**同一批**模块,
`named_parameters()` 默认去重,报出的名字取决于属性注册顺序,不可靠。

日志里应能看到:
```
Loading weight checkpoint only (before DeepSpeed init): ./checkpoints/wanpre60k_video_only.pt
Warm-start guard armed on 4 probe tensors.
Warm-start guard passed: max relative drift after first step = ...  热启动权重确认生效。
```

### 3.5 ⚠️ 被作废的历史结论

以下结论建立在空操作之上,**全部无效,需要重跑**:

| 出处 | 原结论 | 实际情况 |
|---|---|---|
| `AGILEX_TRAINING_PLAN.md` Phase 6 / `NEXT_SESSION.md`「A/B:热启动 RoboTwin 权重**无收益**」 | "已核实热启动确实生效…是真实的负迁移,不是 bug" | 权重从未参与训练。当时的"核实"只验证了 key 名匹配和日志行,没验证权重活到第 1 个 step 之后 |
| `GWP05_TRANSFER_PLAN.md` §5 的 C1 / C2 两臂 | C1 −2.5%(`n.s.`)、C2 +21% 更差 | 同上。C1/C2 都没真正测到 GWP-0.5 预训练 |

好消息:`final_A` 本身**不受影响**(它 `resume=null`,没走这条路径),所以那 39 个 checkpoint、
两套离线评测、RTC 与部署工作全部有效。

---

## 4. final_D 的配置:与 final_A 的唯一差别就是 `resume`

配置文件**一个字都没改**,只加一个 CLI override:

```bash
cd /home/gaomeng/FastWAM && source env.sh
RID=final_D \
  INIT_RESUME=./checkpoints/wanpre60k_video_only.pt \
  WGROUP=final_wanpre SKIP_PRELUDE=1 \
  bash scripts/run_final.sh start
```

| 项 | 值 |
|---|---|
| task config | `configs/task/agilex_final_3cam_384_1e-4.yaml`(未修改) |
| 数据 | `data/agilex_empty_the_box_fastwam`(471 条,466 训练 / 5 验证) |
| 归一化统计 | `runs/_shared/dataset_stats.json`(与 final_A **逐比特同一份**,配置里已钉死) |
| 规格 | bs 8 × 8 卡 / nw 8 / ZeRO-1 / `num_epochs=5` → **77,690 步** ≈ 42h |
| LR | 1e-4 / cosine / `warmup_ratio=0.02`(1,553 步)—— **与 final_A 逐步相同,所以任意 step 前缀都可比** |
| seed | 42(train/val 切分与数据顺序相同) |
| 保存 | `save_every=2000` → 与 final_A 的 checkpoint 步号**逐一对齐** |
| 变化 | **只有 video expert 的 825 个张量** |

`scripts/run_final.sh` 本次改成可用环境变量覆盖(`RID` / `INIT_RESUME` / `WGROUP` /
`SKIP_PRELUDE`),默认值保持原 final_A 行为。要点:**崩溃续训用最新 state 目录,不会再用
`INIT_RESUME`**,否则已训练的进度会被丢掉、从 step 0 重来。

### 4.1 免费的逐步基线:不需要单独短跑

LR 调度由 `num_epochs=5` 固定,所以 final_D 第 k 步的 lr 与 final_A 第 k 步完全相同,
`final_A/train.log` 就是逐步对齐的基线。对比命令:

```bash
python scripts/parse_train_log.py \
  A=runs/agilex_final_3cam_384_1e-4/final_A/train.log \
  D=runs/agilex_final_3cam_384_1e-4/final_D/train.log \
  --csv /tmp/AD.csv --compare
```

⚠️ **`loss_video` 单点噪声很大**(final_A 的 step 50/100/150 是 0.3905/0.4039/0.3454),
判读必须取多点均值,不能看单点。

### 4.2 磁盘与持久化

单次保存 92 GB(weights 12 + ZeRO state 80),38 次 = 3.5 TB,而 `/` 只剩约 520 GB。

- `scripts/run_final.sh` 每 5 分钟跑 `prune_ckpt.sh`,本地只留最近 1 state + 3 weights(稳态约 116 GB)
- `scripts/autosync_daemon.sh` 已启动(30 分钟增量 rsync 到 `/mnt/data/gaomeng/FastWAM`,排除 state)
  —— final_A 那 39 份权重之所以还在纯属 prune/sync 的时序巧合,**这次要当成设计而不是运气**
- 另有 80 GB 可回收:`final_A/checkpoints/state/step_077690`(训练已完成,只用于续训,不会再用)

---

## 5. 判读:用 71 条 held-out episode 做跨 run 成对差分

### 5.1 旧的 5 条 val split 对本实验已经污染

FastWAM 的 val split 是 `[36, 97, 136, 449, 461]`(`np.random.default_rng(42)` 切出),
**全在预训练用过的那 471 条里** —— 视频专家背过它们。所以训练期 eval 的
`val_loss` / `psnr` / `ssim` 这次**只能看趋势,不能用来判优劣**。

### 5.2 干净的验证集

`configs/eval/final_A_newval71.yaml` 用的 542 数据集 episode **471–541**(71 条 / 284 个固定样本):
既不在 FastWAM 训练集,也**不在预训练集**(那边只有 471 个 mp4)。对 final_D 依然干净。

基准已算好:`eval_offline/final_A_newval71/`,最优是 **step_032000,`action_rmse` 0.1934**。
因为 `save_every` 不变、样本相同,可以做**跨 run 的逐样本成对差分**(灵敏度比非配对高一个量级:
非配对 CI ±0.13,配对后 Δ 的 CI ±0.003)。

```bash
# 复制 configs/eval/final_A_newval71.yaml,只改 checkpoints.dir 与 out_dir
python -u scripts/offline_eval.py --config configs/eval/final_D_newval71.yaml   # ~50 min
python scripts/visualize_eval.py --sweep eval_offline/final_D_newval71
```

### 5.3 指标优先级(沿用 `NEXT_SESSION.md`「怎么比较不同 eval 结果」)

1. **`action_l2` / `action_l1`(首要)** —— 任务指标。uncond 变体推理时跳过视频想象,直接出动作
2. `val_loss` —— 只看趋势有没有转升(本实验因污染,参考价值下降)
3. `psnr_rd` / `ssim_rd` —— 纯扩散质量
4. **不要**用 `psnr_dg` / `ssim_dg` 判训练好坏(VAE 天花板,近似常数)

### 5.4 预期与风险

- **`loss_video` 一定会明显更好**(已实测 −16%),因为它背下了这些视频
- **`action_l2` 会不会改善是真的未知。** uncond 推理跳过视频想象,更好的视频先验只能通过
  "给动作专家更好的视觉表征"这条间接路径起作用
- **过拟合可能来得更早。** final_A 已经在 step≈32,000(epoch 2)见顶、之后每个 checkpoint 在
  held-out 上都显著更差。一个已经记住这 471 条视频的视频专家大概率更早退化 ——
  这本身是要观测的量,不是失败

---

## 6. 预训练配方与 regime 差异:已澄清,**不构成大问题**

用户确认(2026-09-07):zzd 那边就是**普通的 video DiT 训练,没有考虑首帧条件**。
下面把差异逐条量清,结论是**不致命**。

### 6.1 FastWAM 侧的实际形态(实测)

| 项 | 值 | 出处 |
|---|---|---|
| RGB 输入 | 9 帧 × 384×320 | `num_frames=33`, `action_video_freq_ratio=4` |
| VAE 时间压缩 | 4 → **3 个 latent 帧** | `wan_video_vae.py:1076` |
| 每 latent 帧 token 数 | 120（24/2 × 20/2） | patch `[1,2,2]` |
| latent 帧 0 | **干净首帧条件,timestep 恰好 = 0** | `wan_video_dit.py:563` `token_timesteps[:, 0, :] = 0` |
| loss 计算范围 | **只在 latent 帧 1、2 上** | `fastwam.py:658-660` `pred_video[:, :, 1:]` |
| 注意力掩码 | `first_frame_causal` | `wan_video_dit.py:517-521` |

### 6.2 两条差异,量级差很多

**(a) 注意力掩码 —— 影响很小。** `first_frame_causal` 的实现是
`video_mask[:120, 120:] = False`,即**只有 latent 帧 0 的 query 被挡住**,看不到后面两帧。
**帧 1、2 的 query 依然看到全部 token,和 bidirectional 完全一样。**
而承担 loss 的正是帧 1、2 —— 所以预训练学到的注意力模式对有梯度的那部分是原样在用。

**(b) 逐帧 timestep —— 这条是真差异。** T2V 训练每个视频采**一个** σ,σ=0 是零测集;
FastWAM 在一次前向里给帧 0 喂 timestep=0、帧 1/2 喂采样到的 σ。
所以 3 个 latent 帧里有 1 个(120/360 token)对 `time_embedding` / adaLN `modulation`
这条通路是**分布外输入**。

但这是可以被微调吸收的:42 小时里 DiT 全部参数都在**正确的 regime 下**训练,
预训练的价值在**初始化**,不在"拿来就能直接用"。

**(c) 没有 I2V 分支会被"练坏"。** 这一点值得单独说,因为它是最容易担心错的地方:
FastWAM **不用** Wan 原生的 I2V 通路 —— `has_image_input: false`、
`require_clip_embedding: false`、`require_vae_embedding: false`、
`fuse_vae_embedding_in_latents: true`。首帧条件是 FastWAM **自己实现**的(把干净的首帧 latent
放进序列、timestep 给 0),没有独立的 I2V adapter 权重。
所以"纯 T2V 微调 60k 步"不会损坏 FastWAM 依赖的任何东西。

### 6.3 实测已经回答了这个问题

理论在这件事上没有投票权 —— 修复后与 final_A 逐步对齐的前 1650 步:
`loss_video` **−11.5%,25/25 个对齐点全部更优**。这是**带着**上述全部 regime 错配拿到的。

### 6.4 代价是"留了余量",不是"坏了"

一次在 FastWAM regime 下做的预训练(首帧条件 + 逐帧 timestep + 9 帧窗口)大概率迁移得更好。
这是给 zzd 下一轮预训练的**低成本建议**,不值得为此重做当前这轮。

### 6.5 仍然缺但不影响判读的两项

`num_frames`(`metadata_min121.csv` 的命名疑似暗示 121 = 4k+1)与训练分辨率。
数据本身是 320×384,所以分辨率大概率一致。**这两项都不改变上面的结论,不必追。**


---

## 7. 执行记录

| 时间 | 事件 |
|---|---|
| 2026-09-07 15:2x | 核实 ckpt:825 张量 / hash 匹配注册表 / 漂移 1.7%~12.7% / 预训练数据 320×384 30fps 与训练数据逐条对齐 |
| 2026-09-07 16:0x | 写 `scripts/convert_wan_dit_to_fastwam.py`;V2 判据两次误判后修正(§2.1);产出 `checkpoints/wanpre60k_video_only.pt`(825 / 4.9998 B / 9.31 GiB) |
| 2026-09-07 16:1x | `scripts/run_final.sh` 参数化(`RID`/`INIT_RESUME`/`WGROUP`/`SKIP_PRELUDE`);启动 autosync;启动 final_D |
| 2026-09-07 16:2x | **step 50 `loss_video` 0.3886 vs final_A 0.3905(−0.5%)** —— 触发熔断判据 |
| 2026-09-07 16:3x | 7 个对齐点全部吻合到 ±0.8%(step 250 是 0.3072 vs 0.3072)。结合 ab_B/C1/C2 的同型指纹,定位到 `prepare()` 与权重加载的顺序问题。**abort 训练** |
| 2026-09-07 16:4x | 修复 trainer.py(拆分两条 resume 路径 + 加守卫);用 RoboTwin 权重做阳性对照,`loss_action` 1.95 → 3.25~4.38,**修复确认生效** |
| 2026-09-07 16:56 | 归档 `_void_final_D_resume_noop/` 与 `_guard_check_robotwin/`;用修复后的代码重启 final_D。守卫 armed 且 passed |
| 2026-09-07 17:0x | **step 50 `loss_video` 0.3199(−18.1%)、step 100 0.3424(−15.2%)** —— 具身视频预训练确实迁移进来了 |

### 监控

```bash
RID=final_D bash scripts/run_final.sh status
tail -30 .run_final.final_D.log
```
