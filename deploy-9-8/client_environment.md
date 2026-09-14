# Client 服务器环境说明

这个文档只说明真机所在的 client 服务器需要什么环境、怎么检查、哪些检查可以跳过。

当前主线是路线 A：`deploy-9-8` websocket client / server 方案。路线 B：openpi / `deploy_real.py` 只作为历史备选，不走这条路线时可以跳过整节路线 B。

## 1. 路线 A 结论

机器人侧 client 建议使用轻量 conda 环境，不装 torch，不装 CUDA 推理栈。

```bash
conda create -n gwp_client python=3.10 -y
conda activate gwp_client
```

真实机器人侧必须具备：

- ROS2 Humble。
- `cv_bridge`。
- Piper SDK。
- `numpy<2`。
- `websockets`、`msgpack`、`msgpack-numpy`。
- `opencv-python`。
- `pyarrow`，只在 replay 或从 parquet 读取起始姿态时需要。

本机 synthetic smoke test 使用 `fastwam` 环境和 `--no-ros2` 已通过，但它只证明 server 协议可用，不能替代机器人侧环境检查。

## 2. 最小安装

```bash
conda activate gwp_client
pip install "numpy<2"
pip install opencv-python websockets msgpack msgpack-numpy pyarrow piper_sdk
```

如果 `pip install piper_sdk` 找不到包，使用机器人侧已有的 Piper SDK 安装方式，或者把 SDK 放进 `gwp_client` 的 Python 路径。

## 3. 每个新终端先执行

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
```

如果真实系统不是 Humble，把 `humble` 换成实际 ROS2 发行版。

## 4. 必需检查

### Python 和 NumPy

```bash
python --version
python -c "import numpy as np; print(np.__version__)"
```

期望：

- Python 为 `3.10.x`。
- NumPy 必须 `<2`，否则 ROS2 / `cv_bridge` 很容易出 ABI 问题。

### ROS2 和相机桥接

```bash
echo $ROS_DISTRO
python -c "import rclpy, cv_bridge; print('ros2 stack ok')"
```

期望：

- `$ROS_DISTRO` 非空。
- `rclpy` 和 `cv_bridge` 都能导入。

### Piper SDK

```bash
python -c "import piper_sdk; print('piper ok')"
```

期望：

- `piper_sdk` 能导入。
- 真机侧 CAN 设备名与脚本默认一致：左臂 `can0`，右臂 `can1`。不一致时用 `--can-left` / `--can-right` 覆盖。

### 通信依赖

```bash
python -c "import cv2, websockets, msgpack, msgpack_numpy; print('net/image deps ok')"
python -c "import pyarrow; print('pyarrow ok')"
```

`pyarrow` 只在 `--mode replay` 或 `--start-pose-parquet` 使用，不做这两件事时可暂时跳过。

### Client 入口

```bash
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py --help
```

如果 `--help` 正常输出，说明脚本至少能启动到 argparse 层。

## 5. 联调命令

### 只验证 server 协议

可以在 GPU 本机或机器人侧执行。使用 `--no-ros2` 时不需要 ROS2/Piper，也不会读取真实相机和关节状态。

```bash
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode log-only \
  --no-ros2
```

这个步骤可以跳过，前提是同一 server 已经在本机完成过 synthetic smoke test。

### 真实机器人侧 log-only

不能加 `--no-ros2`。这一步必须做，因为它验证真实相机、真实 Piper 状态和 server 推理三者同时可用。

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode log-only
```

### 第一次单步运动

必须现场确认安全后执行。

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

## 6. 可以跳过的检查

- 只做 `--mode log-only --no-ros2` 时，可以跳过 ROS2、`cv_bridge`、Piper SDK 和 CAN 检查。
- 不做 replay、不从 parquet 读取起始姿态时，可以暂时跳过 `pyarrow`。
- 使用 `robot_client.py --goto-start` 时，可以不单独运行 `goto_start.py`。
- 不走 openpi / `deploy_real.py` 时，可以跳过下面路线 B。

## 7. 不能跳过的检查

- 真机侧 `log-only` 前不能跳过 ROS2、`cv_bridge`、Piper SDK 检查。
- 真机运动前不能跳过 `--confirm-safety`。
- 闭环前不能跳过单步动作测试。
- 真机前需要确认 `--gripper-scale 0.105`。当前该值会把模型输出 `1.0` 映射成 `105000` raw，再被硬件上限夹到 `70000` raw；模型输出高于约 `0.667` 的夹爪开度都会饱和。

## 8. 路线 B: openpi / deploy_real.py 机器人端环境

如果你走的是下面这一套：

```bash
1. 启动机械臂端 topic
2. conda activate piper_ros
3. python /home/geekplus/workspace/xf/openpi/scripts/deploy/start_robot_arm_service.py
4. conda activate fastwam
5. export FASTWAM_ENV=/home/geekplus/miniforge3/envs/fastwam
6. source env.sh
7. python scripts/deploy_real.py run --config configs/deploy/agilex_real.yaml ...
```

那么 client 服务器上需要的是两套环境，而不是一套。本路线不是 `deploy-9-8` 当前主线。

### 8.1 `piper_ros` 环境

这个环境负责机械臂服务进程。

检查：

```bash
conda activate piper_ros
python -c "import piper_sdk; print('piper_ros ok')"
```

如果 `start_robot_arm_service.py` 起不来，优先检查：
- `piper_sdk` 是否可导入
- 机械臂 topic 是否已经启动
- 网卡是否正常

### 8.2 `fastwam` 环境

这个环境负责模型侧脚本。

检查：

```bash
conda activate fastwam
export FASTWAM_ENV=/home/geekplus/miniforge3/envs/fastwam
source env.sh
python scripts/deploy_real.py --help
```

如果 `deploy_real.py` 起不来，优先检查：
- `FASTWAM_ENV` 是否指向真实环境
- `env.sh` 是否成功把需要的变量和路径导入
- 对应的 yaml 里 checkpoint 配置是否正确

### 8.3 路线 B 的推荐启动顺序

1. 先启动机械臂端 topic
2. 再启动 `piper_ros` 的 service
3. 再启动 `fastwam` 的模型侧脚本
4. 最后做非 RTC 或 RTC 的部署跑批

### 8.4 路线 B 的成功检查点

#### 启动 service 成功
- 终端里没有 `ModuleNotFoundError`
- 机械臂 service 进程常驻

#### 启动模型侧成功
- `deploy_real.py` 能正常读取 yaml
- checkpoint 和去噪配置能从 yaml 里解析出来
- 进程开始等待端点连接

#### 真机联通成功
- client 能拿到 observation
- 模型能回传 action
- episodes 能按 `num-episodes` 跑完

### 8.5 远程重启后网卡恢复

如果工控机重启后远程连不上，在真机服务器上执行：

```bash
sudo dhclient -r eno1
sudo dhclient -v eno1
```

这个操作的目标只是把 `eno1` 的地址重新拉起来，恢复远程登录和网络连通性。
