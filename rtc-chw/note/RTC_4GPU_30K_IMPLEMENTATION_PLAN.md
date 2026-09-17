# 基于 fastwam3wstep checkpoint 的 RTC 推理与 4 卡 10w step 微调方案

本文档记录当前如何落地 RTC 推理跑通、4 卡无梯度累积 batch size 8 训练 100000 step，以及用 wandb 和离线评测看曲线与效果。

当前不改训练代码。优先使用已经解压出的现成 RTC 代码与配置，通过命令行覆盖关键参数完成实验。

## 0. 代码与环境位置

代码已经解压到：

```bash
/home/chw/code/packages/FastWAM/rtc-chw/fastwam-gm/fastwam_train_rtc
```

后续所有命令默认从该目录执行：

```bash
cd /home/chw/code/packages/FastWAM/rtc-chw/fastwam-gm/fastwam_train_rtc
export FASTWAM_ENV=/home/chw/miniconda3/envs/fastwam
source env.sh
```

`env.sh` 会处理：

- 把 conda 环境的 `bin` 加到 `PATH`，否则 `accelerate` 可能找不到。
- 设置 `DIFFSYNTH_MODEL_BASE_PATH=./checkpoints`。
- 设置 HF / ModelScope 缓存目录到项目内。
- 如果 `.wandb_key` 存在，会自动导出 `WANDB_API_KEY`。

## 1. 总体目标

目标分三段：

1. 基于 `fastwam3wstep` checkpoint 跑通 RTC 推理链路。
2. 使用 4 张 GPU，`gradient_accumulation_steps=1`，探出能稳定塞满显存的最大 `batch_size`。
3. 从 `fastwam3wstep` 权重热启动，训练 RTC 100000 step，并用 wandb 查看训练与验证曲线。

这里的 `batch_size` 是每卡 batch。4 卡时全局 batch 为：

```text
global_batch = batch_size * 4 * gradient_accumulation_steps
```

因为要求不梯度累积，所以：

```text
gradient_accumulation_steps = 1
global_batch = batch_size * 4
```

## 2. 需要先确认的输入

已找到 `fastwam3wstep` checkpoint：

```bash
CKPT_3W=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_030000.pt
```

注意：本仓库里 `resume` 指向 `.pt` 文件时，是“只加载模型权重，optimizer / scheduler / global step 重新开始”。这正是 RTC 微调需要的行为。不要把 `resume` 指向 checkpoint state 目录，除非目的是完整续训。

## 3. 使用哪套 RTC 配置

训练任务使用现有配置：

```bash
task=agilex_rtc_3cam_384_1e-4
```

对应文件：

```bash
configs/task/agilex_rtc_3cam_384_1e-4.yaml
```

这个配置已经打开：

```yaml
model:
  rtc:
    enabled: true
    max_delay: 12
    delay_distribution: uniform
    execution_horizon: 8
```

它默认从 `step_032000.pt` 开始训 8000 step。我们不直接改文件，正式实验用命令行覆盖：

```bash
resume=$CKPT_3W
max_steps=30000
gradient_accumulation_steps=1
batch_size=<探出来的最大稳定值>
```

## 4. 第一步：RTC 基础自检

先跑不需要 GPU 的 RTC 数值自检：

```bash
python scripts/rtc_selftest.py
```

它主要确认：

- delay 采样合法。
- 前缀 mask 正确。
- 前缀 timestep 被置为干净数据时刻。
- 前缀 loss 被正确剔除。
- scheduler 的 `[B]` / `[B,H]` timestep 广播没有破坏旧逻辑。

然后跑真实模型端到端 RTC 检查：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rtc_e2e_check.py \
  task=agilex_rtc_3cam_384_1e-4 \
  resume=$CKPT_3W \
  output_dir=/tmp/rtc_e2e_from3w
```

它确认：

- RTC 训练 loss 能跑通。
- ActionDiT 收到 per-token timestep。
- `infer_action(delay=0)` 保持旧路径。
- `infer_action(delay>0, action_prefix=...)` 能走前缀条件路径。
- 返回动作的前缀能被精确 clamp。
- delay 变化不改变张量形状，利于 `torch.compile`。

只有这两步通过后再进入 batch size 探测。

## 5. 第二步：跑通 RTC 推理和测速

先用 synthetic 输入跑单卡推理测速：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/dryrun_fastwam.py --config-name sim_agilex \
  ckpt=$CKPT_3W \
  +DRYRUN.use_random_context=true \
  +DRYRUN.compile_action_infer=true \
  +DRYRUN.delay=5 \
  +DRYRUN.warmup=10 \
  +DRYRUN.iters=200
```

重点看输出：

- mean / median / p95 / p99 延迟。
- 输出里建议的 `rtc.max_delay` 范围。
- peak allocated 显存。

当前训练配置的 `model.rtc.max_delay=12` 表示训练覆盖 `d=0..11`。如果测速显示 p99 对应控制步过大，超过这个范围，就需要考虑把 `max_delay` 调大后再训，否则部署时实际 delay 超出训练分布，RTC 前缀条件会失效。

如果只想确认前缀路径和非前缀路径延迟是否一致，可以额外对比：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/dryrun_fastwam.py --config-name sim_agilex \
  ckpt=$CKPT_3W \
  +DRYRUN.use_random_context=true \
  +DRYRUN.compile_action_infer=true \
  +DRYRUN.delay=0 \
  +DRYRUN.warmup=10 \
  +DRYRUN.iters=100
```

理论上 delay 前缀路径几乎零额外开销。

## 6. 第三步：4 卡最大 batch size 探测

要求是 4 卡、无梯度累积，所以固定：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
gradient_accumulation_steps=1
```

建议探测顺序：

```text
batch_size = 8 -> 10 -> 12 -> 14 ...
```

历史文档里 `batch_size=8` 是保守稳定点；`batch_size=12` 在某些场景余量较小，所以不要直接假设 12 一定能长训。

每个候选 batch 至少跑 20 到 50 step。先不保存 checkpoint，避免探测阶段写入几十 GB 文件。

示例，先探 `batch_size=8`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=probe_rtc3w_bs8 \
bash scripts/train_zero1.sh 4 \
  task=agilex_rtc_3cam_384_1e-4 \
  resume=$CKPT_3W \
  max_steps=50 \
  batch_size=8 \
  gradient_accumulation_steps=1 \
  eval_every=0 \
  save_every=999999 \
  log_every=1 \
  wandb.enabled=false
```

如果成功，再试：

```bash
batch_size=10
batch_size=12
batch_size=14
```

判断标准：

- 不 OOM。
- 训练前 50 step 稳定。
- 单步时间没有异常变慢。
- `nvidia-smi` 峰值显存最好保留一定余量，不能只追求刚好塞满，因为正式训练会有 eval、保存、碎片和长时间运行风险。

如果打开 eval 会额外吃显存，可以对最终候选 batch 再跑一次：

```bash
eval_every=25
```

如果 eval OOM，但训练不 OOM，有两个选择：

1. 降低正式训练 batch size。
2. 正式训练先 `eval_every=0`，训练完再单独离线评测。

我更倾向第一种，除非特别需要最大训练吞吐。

## 7. 第四步：正式 4 卡训练 100000 step

正式 run 名建议：

```bash
RUN_ID=rtc_from3w_bs8_100k_save5k_log100
```

启动命令模板：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=rtc_from3w_bs8_100k_save5k_log100 \
bash scripts/train_zero1.sh 4 \
  task=agilex_rtc_3cam_384_1e-4 \
  resume=$CKPT_3W \
  max_steps=100000 \
  batch_size=8 \
  gradient_accumulation_steps=1 \
  warmup_steps=600 \
  save_every=5000 \
  eval_every=500 \
  log_every=100 \
  wandb.enabled=true \
  wandb.project=fastwam-agilex \
  wandb.name=rtc_from3w_bs8_100k_save5k_log100
```

参数解释：

- `max_steps=100000`：按最新要求训练到 10w step。
- `warmup_steps=600`：沿用 smoke 与正式启动验证过的设置。
- `save_every=5000`：减少 checkpoint 数量，权重落到 `/mnt/data/chw/fastwam/runs` 下。
- `eval_every=500`：保留较密的验证点，便于观察动作精度。
- `log_every=100`：每 100 step 输出一次 loss 与 ETA，降低日志开销。
- `gradient_accumulation_steps=1`：满足不梯度累积。
- `batch_size=8`：用户指定，且 4 卡探测稳定。

如果磁盘压力大，可以把 `save_every` 改成 3000 或 5000。但 RTC 训练可能中途某个 checkpoint 最好，保存太稀会降低后续挑点能力。

## 8. 第五步：训练时磁盘清理

checkpoint 会很大。正式训练时建议同步清理 ZeRO state，只保留少量 state 和尽量多的 weights：

```bash
watch -n 900 'KEEP_STATE=1 KEEP_WEIGHTS=99 bash scripts/prune_ckpt.sh runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_30k'
```

含义：

- `KEEP_STATE=1`：只保留最近 1 份完整训练 state，防止磁盘爆。
- `KEEP_WEIGHTS=99`：尽量保留所有 `.pt` weights，方便后续扫 checkpoint。

如果磁盘仍然紧张，把 `KEEP_WEIGHTS` 降到 20 或 15。

## 9. 第六步：wandb 看什么

wandb 会由训练代码自动记录。重点看这些 key：

```text
train/loss
train/loss_action
train/loss_video
train/rtc_mean_delay
train/lr
train/grad_norm
performance/steps_per_sec
performance/samples_per_sec
eval/val_loss
eval/action_l1
eval/action_l2
eval/psnr_rg
eval/ssim_rg
```

优先级：

1. `train/loss_action`：动作分支是否稳定下降。
2. `eval/action_l1` / `eval/action_l2`：无前缀动作精度有没有明显退化。
3. `train/rtc_mean_delay`：应接近 uniform `[0,12)` 的均值 5.5，短窗口会有波动。
4. `performance/samples_per_sec`：判断 batch size 是否真的提升吞吐。
5. `eval/val_loss`：辅助观察过拟合，不单独作为选 checkpoint 的唯一标准。

注意：训练期 `eval/action_l1/l2` 默认是无前缀推理，不直接衡量 RTC 接缝质量。RTC 的关键指标需要单独跑 `rtc_seam_eval.py`。

## 10. 第七步：RTC 接缝离线评测

训练过程中可以挑几个 checkpoint 做接缝评测，例如：

```text
step_006000.pt
step_012000.pt
step_018000.pt
step_024000.pt
step_030000.pt
```

命令模板：

```bash
python scripts/rtc_seam_eval.py \
  --task agilex_rtc_3cam_384_1e-4 \
  --ckpt runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_30k/checkpoints/weights/step_030000.pt \
  --delays 0,2,4,8 \
  --num-samples 32 \
  --out eval_offline/rtc_seam/rtc_from3w_30k_step030000
```

主要看：

- `seam_ratio`：越接近 1 越好，表示接缝处加速度接近轨迹普通位置。
- `seam_accel`：接缝跳变绝对大小。
- `postfix_L1`：前缀条件下后缀动作精度。
- `prefix_err`：应为 0，说明 clamp 生效。

选 checkpoint 时不能只看 seam。理想结果是：

- `seam_ratio` 明显低于 naive async 基线。
- `postfix_L1` 不要显著恶化。
- `eval/action_l1/l2` 不要比 3w base 明显退化。

## 11. 部署推理怎么接

无硬件时先跑部署侧自检：

```bash
python scripts/deploy_real.py rtc-check --config configs/deploy/agilex_real_rtc.yaml
python scripts/deploy_real.py selfcheck --config configs/deploy/agilex_real_rtc.yaml
python scripts/deploy_real.py benchmark --config configs/deploy/agilex_real_rtc.yaml
```

如果要用训练出的新权重跑部署配置，需要把 `configs/deploy/agilex_real_rtc.yaml` 里的：

```yaml
checkpoint: ...
```

指到新 run 的 checkpoint，例如：

```yaml
checkpoint: ./runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_30k/checkpoints/weights/step_030000.pt
```

当前先不改代码时，可以临时复制配置或命令行覆盖，避免污染默认部署配置。

真机/拆分部署推荐路径是：

```bash
bash scripts/start_server.sh
bash scripts/start_client.sh --mode async --rtc --confirm-safety --i-am-watching --max-duration 600
```

需要注意 server 和 client 的 RTC 开关必须一致；否则 client 会根据 server 上报的 `rtc_trained` 拒绝继续。

## 12. 风险与回退

### 12.1 找不到 3w checkpoint

阻塞正式训练。必须先明确 `$CKPT_3W`。

回退方案：

- 用已有 `step_032000.pt` 跑默认 RTC 微调。
- 或用已有 `step_077690.pt` 跑 `agilex_rtc_from077690_20k` 变体，但该起点历史上处在过拟合区，不是首选。

### 12.2 batch size 探出来很小

如果 4 卡无梯度累积只能用很小 batch，训练会更慢且全局 batch 变小。

因为用户明确要求不梯度累积，所以不使用 `gradient_accumulation_steps>1`。回退只降 `batch_size`，不改梯度累积。

### 12.3 eval OOM

如果训练不 OOM 但 eval OOM：

- 首选降低 batch size。
- 次选正式训练设 `eval_every=0`，训练后用 `offline_eval` 或 `rtc_seam_eval` 单独评。

### 12.4 RTC 接缝好了但动作精度下降

这在历史文档里出现过。处理方式：

- 扫多个 checkpoint，不默认使用最后一步。
- 同时看 `eval/action_l1/l2` 和 `rtc_seam_eval`。
- 如果早期 checkpoint seam 已经修好而后期精度变差，就选早期点。

### 12.5 delay 超出训练覆盖

如果部署测速显示实际 delay 经常超过 `max_delay-1`，需要重新考虑训练配置里的 `model.rtc.max_delay`。否则模型没见过更长前缀，RTC 的稳定性没有保证。

## 13. 最终交付物

本次实验完成后应留下：

```text
/mnt/data/chw/fastwam/runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_bs8_100k_save5k_log100/
  checkpoints/weights/step_*.pt

/home/chw/code/packages/FastWAM/rtc-chw/fastwam-gm/fastwam_train_rtc/runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_bs8_100k_save5k_log100/
  config.yaml
  logs/
  wandb/
  checkpoints -> /mnt/data/chw/fastwam/runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_bs8_100k_save5k_log100/checkpoints

eval_offline/rtc_seam/
  rtc_from3w_bs8_100k_step*.pt/
```

最终选择 checkpoint 的标准：

1. RTC 接缝指标足够好，尤其 `seam_ratio` 明显下降。
2. 动作精度没有明显比 3w base 退化。
3. wandb 上训练稳定，没有梯度爆炸、loss 异常反弹或吞吐异常。
4. 部署 benchmark 下实际 delay 落在训练覆盖范围内。

## 14. 最小执行清单

## 15. 当前执行结论与最终配置

截至 2026-09-17 已完成：

- `fastwam3wstep` checkpoint 已确认存在：
  `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_030000.pt`
- RTC 端到端检查已通过。
- 3w checkpoint 的 RTC 推理测速已跑通：`delay=5` eager 模式 `mean=0.3118s`，`p99=0.3483s`，约 `10.4` 个 30Hz 控制步。
- 4 卡无梯度累积 batch 探测结果：
  - `batch_size=8`：通过，峰值约 `70653 MiB`，建议正式训练使用。
  - `batch_size=10`：通过但峰值约 `77911 MiB`，余量太小，不建议长训。
- 正式训练 smoke test 已通过：
  - run: `rtc_from3w_bs8_smoke6_offline_20260917_093515`
  - `max_steps=6`
  - `batch_size=8`
  - `gradient_accumulation_steps=1`
  - `world_size=4`
  - wandb offline 已记录 6 个 step 的 loss。

smoke test 的 6 步 loss：

```text
step=1 loss=0.3844 loss_action=0.0931 loss_video=0.2913 rtc_mean_delay=5.4375
step=2 loss=0.3883 loss_action=0.0862 loss_video=0.3021 rtc_mean_delay=5.4375
step=3 loss=0.3393 loss_action=0.0693 loss_video=0.2700 rtc_mean_delay=4.8750
step=4 loss=0.4063 loss_action=0.0966 loss_video=0.3097 rtc_mean_delay=5.0000
step=5 loss=0.3603 loss_action=0.0956 loss_video=0.2647 rtc_mean_delay=5.5000
step=6 loss=0.3519 loss_action=0.1174 loss_video=0.2345 rtc_mean_delay=5.5938
```

wandb 已通过 `wandb login` 登录，正式训练使用 online 模式。当前已启动的正式训练为：

```bash
cd /home/chw/code/packages/FastWAM/rtc-chw/fastwam-gm/fastwam_train_rtc
export FASTWAM_ENV=/home/chw/miniconda3/envs/fastwam
source env.sh
export PYTHONPATH=$(pwd)/src

export CKPT_3W=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_030000.pt
export ACTION_DIT=/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export STATS=/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=rtc_from3w_bs8_100k_save5k_log100 \
bash scripts/train_zero1.sh 4 \
  task=agilex_rtc_3cam_384_1e-4 \
  resume=$CKPT_3W \
  model.action_dit_pretrained_path=$ACTION_DIT \
  data.train.pretrained_norm_stats=$STATS \
  data.val.pretrained_norm_stats=$STATS \
  max_steps=100000 \
  batch_size=8 \
  gradient_accumulation_steps=1 \
  warmup_steps=600 \
  save_every=5000 \
  eval_every=500 \
  log_every=100 \
  wandb.enabled=true \
  wandb.mode=online \
  wandb.project=fastwam-agilex \
  wandb.name=rtc_from3w_bs8_100k_save5k_log100
```

运行状态：

- tmux session: `rtc_from3w_bs8_100k`
- wandb run: `https://wandb.ai/cuihanwen621-beijing-institute-of-technology/fastwam-agilex/runs/grgiw12y`
- checkpoint 目录: `/mnt/data/chw/fastwam/runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_bs8_100k_save5k_log100/checkpoints`
- 本地 run 目录: `/home/chw/code/packages/FastWAM/rtc-chw/fastwam-gm/fastwam_train_rtc/runs/agilex_rtc_3cam_384_1e-4/rtc_from3w_bs8_100k_save5k_log100`

已确认日志：

```text
Starting training with max_steps=100000
Warm-start guard passed
step=100/100000 loss=0.4080 loss_action=0.0798 loss_video=0.3282 eta=39:37:45
```
