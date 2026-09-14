# FastWAM 真机部署教程

这份教程对应 `deploy-9-8` 目录下的独立部署副本。

目标：

- GPU 服务器负责模型加载和推理。
- 机器人侧 client 负责 ROS2 相机订阅、Piper 状态读取和动作发送。
- 两端通过 websocket + msgpack 通信。

## 1. 部署内容

### GPU 服务器

GPU 服务器不能只拷贝 `robot_server.py`。它还需要完整 FastWAM 运行环境：

- `deploy-9-8/robot_server.py`
- `configs/`
- `src/fastwam/`
- `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- `checkpoints/Wan-AI/` 和 `checkpoints/DiffSynth-Studio/` 下的 Wan 基础模型缓存
- checkpoint: `/mnt/data/chw/fastwam/cp-copy/step_020000.pt`
- dataset stats: `/mnt/data/chw/fastwam/cp-copy/dataset_stats.json`
- 训练数据 `meta/tasks.jsonl`，或启动时显式传 `--instruction`

### 机器人侧 Client

机器人侧需要：

- `deploy-9-8/robot_client.py`
- `deploy-9-8/goto_start.py`，可选；如果使用 `robot_client.py --goto-start`，可以不单独运行
- ROS2 Humble 环境
- Piper SDK
- `numpy<2`、`websockets`、`msgpack`、`msgpack-numpy`、`pyarrow`、`opencv-python`

## 2. 分层验收顺序

按下面顺序推进。不要直接跳到闭环。

### Step 0. GPU 资源和端口检查

在 GPU 服务器上：

```bash
nvidia-smi
ss -ltnp | grep ':8000' || true
```

成功检查点：

- 至少有一张 GPU 显存足够空闲。本机 2026-09-09 检查时 GPU 4-7 空闲，建议 `CUDA_VISIBLE_DEVICES=4`。
- `8000` 端口没有被旧 server 占用。

### Step A. GPU Server 自测

首次部署、换代码、换 checkpoint、换 stats、重装环境后必须跑。日常重启同一套服务时可以跳过。

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
CUDA_VISIBLE_DEVICES=4 python deploy-9-8/robot_server.py \
  --ckpt /mnt/data/chw/fastwam/cp-copy/step_020000.pt \
  --dataset-stats /mnt/data/chw/fastwam/cp-copy/dataset_stats.json \
  --task agilex_empty_box_uncond_3cam384 \
  --self-test
```

成功检查点：

- 进程正常退出，退出码为 `0`。
- 日志出现 `Loaded trained checkpoint`。
- 日志出现 `Loaded normalization stats`。
- 日志出现 `action chunk shape: (32, 14)`。
- 日志出现 `=== SELF TEST PASSED ===`。

本机 2026-09-09 已通过，单次 synthetic 推理约 `1.0s`。

### Step B. 启动 GPU Server 正式服务

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
CUDA_VISIBLE_DEVICES=4 python deploy-9-8/robot_server.py \
  --ckpt /mnt/data/chw/fastwam/cp-copy/step_020000.pt \
  --dataset-stats /mnt/data/chw/fastwam/cp-copy/dataset_stats.json \
  --task agilex_empty_box_uncond_3cam384 \
  --host 0.0.0.0 \
  --port 8000
```

成功检查点：

- 日志出现 `serving on ws://0.0.0.0:8000`。
- `ss -ltnp | grep ':8000'` 能看到 Python 进程监听。
- 没有模型加载报错。
- GPU 显存开始占用，服务保持运行。

### Step C1. 本机协议 Smoke Test

只验证 server 协议，不需要 ROS2 / Piper / 真机。这个步骤在本机 2026-09-09 已通过，真实联调时可以跳过。

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
python deploy-9-8/robot_client.py \
  --server ws://127.0.0.1:8000 \
  --mode log-only \
  --no-ros2
```

成功检查点：

- 能打印 server info。
- 能看到 `--no-ros2: using synthetic gray images`。
- 能打印 `--- inference #1: chunk (32, 14) ---`。
- 动作前三帧 safety verdict 为 `OK` 或能解释的 `WOULD-REJECT`。

### Step C2. 机器人侧真实 Log-only

这是第一次真实 client 联调，不能跳过，也不要加 `--no-ros2`。它会初始化相机和读取 Piper 状态，但不会发送动作。

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode log-only
```

成功检查点：

- 能连上 server，不报 `Connection refused` / `timeout`。
- ROS2 三路相机 ready。
- Piper 能读到非异常关节状态。
- 输出里能看到 action chunk `(32, 14)` 和前三帧动作描述。
- server 端持续出现 `infer #N -> chunk (32, 14) in ...s`。

### Step D. 单步动作测试

第一次会移动机械臂的测试。必须低速、只执行 1 条动作、现场确认急停。

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode step \
  --max-steps 1 \
  --goto-start \
  --confirm-safety \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 1
```

成功检查点：

- `goto-start` 平滑到达训练起始姿态。
- server 返回动作块。
- client 只执行 1 条动作后停止。
- 不触发 joint limit、trajectory jump、tracking error。
- 结束时打印 holding current pose。

### Step E. 短闭环测试

只有 Step C2 和 Step D 都通过后再做。

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode closed-loop \
  --goto-start \
  --confirm-safety \
  --i-am-watching \
  --max-duration 30 \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 8
```

成功检查点：

- client 运行约 30 秒后自动停止。
- server 持续返回动作块。
- 没有观测过期。
- 没有 joint jump。
- tracking error 不持续接近或超过阈值。

## 3. 哪些可以跳过

可以跳过：

- 同机、同代码、同 checkpoint/stats、同环境下，已经通过的 `--self-test` 可以不在每次启动前重复。
- 已经在本机验证过 C1 后，真实联调可以直接做 C2。
- 使用 `robot_client.py --goto-start` 时，可以不单独运行 `goto_start.py`。
- 只走本目录 websocket 方案时，可以忽略 openpi / `deploy_real.py` 路线。

不能跳过：

- 端口和 GPU 检查。
- 真实机器人侧 log-only。
- 第一次运动前的 `--confirm-safety`。
- 第一次运动的 `--mode step --max-steps 1`。
- 闭环时的 `--i-am-watching` 和 `--max-duration`。
- 夹爪行程标定。当前 `--gripper-scale 0.105` 会让模型输出大于约 `0.667` 的开度都夹到硬件上限 `70000`。

## 4. 运行时应该看到什么

server 正常输出：

- `State shape_meta rebuilt from dataset_stats.json`
- `Loaded trained checkpoint`
- `Loaded normalization stats`
- `serving on ws://0.0.0.0:8000`
- `infer #N -> chunk (32, 14) in ...s`

client 正常输出：

- `server info`
- `server expects state`
- `MODE: log-only` / `MODE: step` / `MODE: closed-loop`
- log-only 下打印动作前三帧
- step / closed-loop 下打印 `queued ... action(s) from chunk ...` 和执行动作

## 5. 不要做的事

- 不要把 cpp / GGUF 和这条 FastWAM websocket 路线混用。
- 不要在机器人侧 ROS2 client 环境使用 `numpy 2.x`。
- 不要在第一次联调时直接上长时闭环。
- 不要在没有真实相机 ready 的情况下去做 step / closed-loop。
- 不要把 synthetic `--no-ros2` 结果当作真机相机链路通过。

## 6. 最终验收

这条部署线可用于短闭环的条件：

1. GPU server `--self-test` 通过。
2. GPU server 正式服务可监听 `0.0.0.0:8000`。
3. 机器人侧 `--mode log-only` 使用真实 ROS2 相机和真实 Piper 状态通过。
4. 机器人侧 `--mode step --max-steps 1 --goto-start --confirm-safety` 通过。

四项都过了，再做 30 秒闭环。
