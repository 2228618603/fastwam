# FastWAM 4090 真机本地部署教程

这个目录可以打包发到真机 4090 主机上运行。它把模型推理、ROS2 相机读取、Piper 双臂状态读取和动作下发放在同一个进程里，不再走“真机 client 请求远程 GPU server”的 websocket 通信链路。

当前建议拆成两个目录：

```text
代码目录：/home/geekplus/chw/fastwam-load-giga
权重目录：/media/geekplus/PortableSSD/chw/fastwam-load-giga
```

代码目录只放 Python 源码、部署脚本、配置、小的 stats/context 文件；权重目录放 34G 左右的大模型文件。

## 0. 在源机器上生成两个包

在 `geekplus-h800-141` 上执行：

```bash
cd /home/chw/code/packages/FastWAM/deploy-all-on-4090/code
bash scripts/make_split_archives.sh
```

会生成：

```text
/home/chw/code/packages/FastWAM/deploy-all-on-4090/dist/fastwam-load-giga-code.tar.gz
/home/chw/code/packages/FastWAM/deploy-all-on-4090/dist/fastwam-load-giga-models.tar.zst
```

## 1. 真机侧启动前检查

先解压代码包到本机 SSD：

```bash
mkdir -p /home/geekplus/chw
tar -C /home/geekplus/chw -xzf fastwam-load-giga-code.tar.gz
```

再解压权重包到移动 SSD：

```bash
mkdir -p /media/geekplus/PortableSSD/chw
tar -C /media/geekplus/PortableSSD/chw --use-compress-program zstd \
  -xf fastwam-load-giga-models.tar.zst
```

进入代码目录：

```bash
cd /home/geekplus/chw/fastwam-load-giga
```

确认代码和权重都完整：

```bash
python scripts/verify_bundle_integrity.py \
  --root /home/geekplus/chw/fastwam-load-giga \
  --model-root /media/geekplus/PortableSSD/chw/fastwam-load-giga
```

如果真机默认 `python` 不是 FastWAM 环境，可以显式指定：

```bash
export PYTHON_BIN=/path/to/your/fastwam/env/bin/python
```

如果真机有多张 GPU，可以指定使用哪张卡：

```bash
export CUDA_VISIBLE_DEVICES=0
```

默认权重目录就是 `/media/geekplus/PortableSSD/chw/fastwam-load-giga`。如果你临时放在别处，设置：

```bash
export FASTWAM_MODEL_ROOT=/your/model/root
```

## 2. 第一步：离线 Smoke Test

这个命令只加载模型，用假图像和零状态跑一次推理，不连接 ROS2，也不会碰机械臂：

```bash
bash scripts/run_self_test.sh
```

看到类似下面输出即可：

```text
SELF TEST PASSED: action chunk (32, 14), dtype=float32
```

如果 4090 显存紧张，可以降低进程显存上限：

```bash
CUDA_MEMORY_FRACTION=0.35 bash scripts/run_self_test.sh
```

## 3. 第二步：Log-Only 看模型输出

这个模式会读取真机相机和机械臂状态，跑模型推理并打印动作，但不会给机械臂下发任何命令：

```bash
bash scripts/run_local_log_only.sh
```

如果只是想离线看推理链路，不接 ROS2 相机：

```bash
python -m app.local_robot_runner \
  --mode log-only \
  --no-ros2 \
  --num-inference-steps 4
```

建议先观察几轮输出，确认：

- 三路相机 topic 能正常收到。
- 当前 state 不是全零异常值。
- 输出动作没有明显离谱关节角。
- 日志没有 `WOULD-REJECT` 的安全拒绝提示。

## 4. 第三步：可选 Replay 验证机械臂单位换算

replay 不跑模型，只回放数据集里已有的 action，用来确认“弧度到 Piper raw、夹爪比例到硬件开度”的换算没问题。

```bash
python -m app.local_robot_runner \
  --mode replay \
  --episode-parquet /path/to/episode.parquet \
  --confirm-safety \
  --speed-percent 10 \
  --replay-max-frames 60
```

`--confirm-safety` 表示你已经确认急停可触达、工作区清空、没有人在机械臂运动范围内。没有这个参数，所有会动机械臂的模式都会拒绝启动。

## 5. 第四步：Step 模式，只执行少量模型动作

step 是第一次让模型控制真机时建议使用的模式。默认脚本会先移动到训练数据的平均起始位姿，然后只执行 1 个模型动作：

```bash
bash scripts/run_local_step.sh
```

常用改法：

```bash
MAX_STEPS=3 SPEED_PERCENT=10 CUDA_MEMORY_FRACTION=0.35 bash scripts/run_local_step.sh
```

等价的完整命令是：

```bash
python -m app.local_robot_runner \
  --mode step \
  --max-steps 3 \
  --confirm-safety \
  --goto-start \
  --speed-percent 10 \
  --action-hz 30 \
  --num-inference-steps 4 \
  --cuda-memory-fraction 0.35
```

## 6. 第五步：Closed-Loop 闭环运行

closed-loop 会持续“读相机/状态 -> 模型推理 -> 执行动作 -> 重新观测”。必须显式给运行时长：

```bash
MAX_DURATION=30 bash scripts/run_local_closed_loop.sh
```

常用改法：

```bash
MAX_DURATION=60 \
SPEED_PERCENT=10 \
REPLAN_STEPS=8 \
CUDA_MEMORY_FRACTION=0.35 \
bash scripts/run_local_closed_loop.sh
```

等价完整命令：

```bash
python -m app.local_robot_runner \
  --mode closed-loop \
  --confirm-safety \
  --i-am-watching \
  --goto-start \
  --max-duration 60 \
  --speed-percent 10 \
  --action-hz 30 \
  --replan-steps 8 \
  --num-inference-steps 4 \
  --cuda-memory-fraction 0.35
```

`--i-am-watching` 表示运行期间有人持续看着真机。没有这个参数，closed-loop 会拒绝启动。

## 7. 常改参数速查

`--action-hz`

控制动作下发频率，单位 Hz。默认是 `30`，和训练数据 fps 对齐。

```bash
--action-hz 20
```

频率越低，机械臂越容易跟上，但动作会更慢；频率越高，越接近训练节奏，但跟踪误差可能变大。

`--max-duration`

closed-loop 最长运行时间，单位秒。只对 `--mode closed-loop` 有效。

```bash
--max-duration 30
```

`--max-steps`

step 模式最多执行多少个模型动作。

```bash
--max-steps 1
```

`--speed-percent`

Piper 速度百分比。默认脚本用 `10`，代码里硬上限是 `30`。第一次真机建议从 `10` 开始。

```bash
--speed-percent 10
```

`--replan-steps`

每次模型会输出一段 action chunk，当前是 `(32, 14)`。`--replan-steps` 控制每次推理后连续执行前几个动作，然后重新看相机再推理。默认 `8`。

```bash
--replan-steps 8
```

数值小：更频繁看相机，反馈更及时，但推理调用更频繁。数值大：动作更连贯，但开环时间更长。

`--num-inference-steps`

扩散推理步数。默认脚本用 `4`，主要为了真机低延迟。更大可能提升质量但会变慢。

```bash
--num-inference-steps 4
```

`--cuda-memory-fraction`

限制当前进程最多使用某张 GPU 的显存比例。4090 独占时可以不设或设高一些；和别的任务共用时建议保守一点。

```bash
--cuda-memory-fraction 0.35
```

`--goto-start`

开始推理前先把双臂移动到训练数据的平均起始位姿。模型输出是绝对关节角，训练数据不是从机械零位开始，所以真机推理通常建议加这个参数。

```bash
--goto-start
```

`--obs-timeout-s`

相机观测超时时间。如果图像太旧，会拒绝继续执行。默认 `0.5` 秒。

```bash
--obs-timeout-s 0.5
```

`--max-joint-delta-rad`

连续两条命令之间允许的最大关节跳变。默认 `0.2` rad，用来拦截模型突然输出的大跳变。

```bash
--max-joint-delta-rad 0.2
```

`--max-tracking-error-rad`

命令位置和机械臂当前实测位置之间允许的最大差值。默认 `0.5` rad。如果频繁触发，通常应该降低 `--action-hz` 或提高 `--speed-percent`，而不是直接放宽这个值。

```bash
--max-tracking-error-rad 0.5
```

## 8. 推荐真机启动顺序

按这个顺序来：

```bash
cd /home/geekplus/chw/fastwam-load-giga

# 1. 只测模型加载和推理，不碰真机硬件
bash scripts/run_self_test.sh

# 2. 接 ROS2/Piper，只打印模型输出，不下发动作
bash scripts/run_local_log_only.sh

# 3. 可选：回放数据集动作，确认单位换算和夹爪比例
python -m app.local_robot_runner \
  --mode replay \
  --episode-parquet /path/to/episode.parquet \
  --confirm-safety \
  --speed-percent 10

# 4. 只执行 1 个模型动作
MAX_STEPS=1 bash scripts/run_local_step.sh

# 5. 短时间闭环
MAX_DURATION=15 bash scripts/run_local_closed_loop.sh
```

## 9. 状态和动作布局

这部分不要改。

模型输入 state 是训练布局：

```text
state[0:6]   left arm joints
state[6:12]  right arm joints
state[12]    left gripper
state[13]    right gripper
```

模型输出 action 是动作布局：

```text
action[0:6]   left arm joints
action[6]     left gripper
action[7:13]  right arm joints
action[13]    right gripper
```

旧 reference 包出过布局错位问题，所以这里不要把 state/action 的顺序混用。

## 10. 包内模型文件

当前权重目录应包含：

- `weights/step_035000.pt`
- `weights/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- `model_cache/Wan-AI/Wan2.2-TI2V-5B`
- `model_cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`
- `assets/fixed_task_context.pt`
- `assets/train_stats.json`

其中 `assets/` 在代码目录内，`weights/` 和 `model_cache/` 在移动 SSD 的权重目录内。

本部署使用固定任务 embedding，启动时会看到：

```text
Skipping pretrained text encoder/tokenizer load
```

这是正常的，表示没有加载 T5 文本编码器。
