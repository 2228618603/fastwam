# AgileX Giga Init BS14 训练和实时 Loss 曲线

本文档用于启动 7 卡、单卡 batch size 14、无梯度累积的 FastWAM AgileX 训练，并在文件夹中实时维护 loss 曲线图片和动图。

## 当前实测结论

- 使用 GPU：`1,2,3,4,5,6,7`
- 单卡 batch size：`14`
- 全局 batch size：`14 * 7 = 98`
- 梯度累积：`1`
- `bs=20` 已 OOM。
- `bs=16` 单步 benchmark 成功，但正式训练第 2 step 在 RoPE/RMSNorm 处 OOM，不适合作为 5k 稳定配置。
- `bs=14` 连续 3 step benchmark 成功，峰值 reserved 约 `73.65GiB/卡`。
- `bs=14` benchmark 最近 3 step 平均约 `3.70s/step`，但首轮包含预热效应；按此前单步与多步观测，5000 step 保守按 `6-8h+` 预估，实际会受保存 state I/O 影响。

## 一键启动

```bash
cd /home/chw/code/packages/FastWAM
bash load-giga/code/06_start_bs14_7gpu_training_tmux.sh
```

默认会启动 tmux session：

```bash
tmux attach -t fastwam_bs14_7gpu
```

tmux 中有两个窗口：

- `train`：正式训练进程。
- `plot`：解析训练 log，并持续刷新 loss 曲线 PNG、动图 GIF、CSV 和状态 JSON。

## 直接打开文件查看曲线

不需要端口转发。连上机器后，直接打开这个目录：

```text
/home/chw/code/packages/FastWAM/load-giga/code/live
```

里面会持续更新：

- `loss_curve.gif`：动图，默认展示最近 300 个训练点，训练时持续重写。
- `loss_curve.png`：静态曲线图，包含 train loss、loss_action、loss_video 和吞吐。
- `index.html`
- `train_metrics.csv`
- `eval_metrics.csv`
- `status.json`

如果你用 VS Code Remote、JetBrains Gateway、Jupyter 文件浏览器或其他远程文件管理器，直接点开 `loss_curve.gif` 或 `index.html` 即可看到最新曲线。绘图进程默认每 5 秒刷新一次；训练配置为 `log_every=1`，所以每个 step 都会进入日志并被曲线捕获。

监控脚本优先解析训练 log；如果 `/mnt/data` 上的 log 临时不可读或解析不到 loss 点，会自动回退到 `fastwam_bs14_7gpu:train` 的 tmux pane 最近输出。可在 `status.json` 里看 `source` 字段确认当前数据来源。

## 训练输出

正式训练输出目录：

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs14_5k
```

训练 log：

```text
/mnt/data/chw/fastwam/logs/giga_to_fastwam/agilex_empty_box_giga_init_bs14_5k.log
```

初始化 checkpoint：

```text
/mnt/data/chw/fastwam/checkpoints/giga_to_fastwam/fastwam_giga_pro_video_init.pt
```

## 默认训练参数

关键参数如下：

```text
batch_size=14
gradient_accumulation_steps=1
max_steps=5000
save_every=500
eval_every=0
log_every=1
mixed_precision=bf16
learning_rate=1.0e-5
weight_decay=1.0e-2
```

说明：

- `eval_every=0` 是默认关闭内置 eval，避免训练器在 `/mnt/data` 上写出的 mp4 出现 `moov atom not found`。
- 训练中主要通过 `load-giga/code/live/loss_curve.gif` 和 `loss_curve.png` 观察状态，不需要打开 tmux。
- 如需推理视频验证，用 `load-giga/code/04_verify_fastwam_checkpoint_inference.py` 对某个 checkpoint 单独跑。

## 常用 tmux 操作

进入 session：

```bash
tmux attach -t fastwam_bs14_7gpu
```

从 tmux 暂时退出但不断训练：

```text
Ctrl-b 然后按 d
```

查看窗口：

```text
Ctrl-b 然后按 w
```

停止训练：

```bash
tmux attach -t fastwam_bs14_7gpu
```

进入 `train` 窗口后按：

```text
Ctrl-c
```

如果要关闭整个 session：

```bash
tmux kill-session -t fastwam_bs14_7gpu
```

## 自定义

换 tmux session 名：

```bash
SESSION=fastwam_bs14_run2 bash load-giga/code/06_start_bs14_7gpu_training_tmux.sh
```

换输出目录：

```bash
RUN_DIR=/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs14_5k_run2 \
bash load-giga/code/06_start_bs14_7gpu_training_tmux.sh
```
