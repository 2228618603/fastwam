# WAM emptybox 本地 4090 部署包说明

这个文件夹用于把 `emptybox` 的 WAM cpp/GGUF 推理部署到机械臂电脑本机运行。核心目的：不再让机械臂 client 跨机器访问 GPU 服务器，而是在机械臂电脑的 4090 上启动 server，client 连接 `ws://127.0.0.1:8000`，尽量减少网络延迟导致的动作卡顿。

## 文件内容

```text
/home/zzd/project/wam_local_emptybox_deploy_package/
├── gwp_local_emptybox_deploy.tar.gz   # 最小部署代码包，不含 22GB GGUF 权重
└── README.md                          # 当前说明文件
```

解压后的包里包含：

```text
deploy/robot_server.py                 # 本地 cpp 推理 server
deploy/robot_client.py                 # 机械臂 client：ROS2 相机 + Piper CAN
deploy/goto_start.py                   # 归位脚本
deploy/real_robot_eval/emptybox/*.sh   # emptybox 原始/v3 启动脚本
prompt_tokens/fixed_task_token_bf16.npz
wam.cpp/                               # cpp runtime 源码和离线构建依赖
```

## checkpoint / 权重地址

部署包不包含 GGUF 权重，权重需要单独从 GPU 服务器拷到机械臂电脑解压目录的 `weights/` 下。

当前建议先只测 original 50k：

```text
/mnt/data/zzd/giga-world-policy/weights/giga_geek_470_step50000.gguf
```

如果还要对比 prompt_v3 50k，再拷：

```text
/mnt/data/zzd/giga-world-policy/weights/giga_geek_470_prompt_v3_256_step50000.gguf
```

文件大小和 sha256：

```text
gwp_local_emptybox_deploy.tar.gz
  size   : 36,323,845 bytes
  sha256 : 31d444ade6d634cbb0f2441441be9ebc55aa255efd6e41ff5d52d06f9b62ea72

giga_geek_470_step50000.gguf
  size   : 23,512,826,496 bytes
  sha256 : d9a67d1b318da248acc5ebb6250c01ee1eb0349aa8201f9f80867466553e47bb

giga_geek_470_prompt_v3_256_step50000.gguf
  size   : 23,512,826,496 bytes
  sha256 : 09567800b22d8d08e03e47c16104193979b5b4648c90805defaa8f1af52dfdf1
```

## 机械臂电脑上是否需要重新配置环境？

大概率不需要从头重新配。

如果你之前在机械臂电脑上已经能运行：

```bash
conda activate gwp_client
python robot_client.py --server ws://10.11.0.93:8000 --mode log-only
```

并且 CAN、ROS2 相机、Piper SDK 都正常，那么这个 `gwp_client` 环境可以继续用。现在新增的主要工作是：

1. 把部署包拷过去并解压；
2. 在机械臂电脑本地编译一次 `wam.cpp`，因为 4090 是 Ada / SM89，不建议直接复用 A800 / SM80 的动态库；
3. 把 GGUF 权重拷到解压目录的 `weights/`；
4. server 和 client 都在机械臂电脑本机运行。

如果机械臂电脑没有 `nvcc` / CUDA Toolkit，`wam.cpp` 编译会失败；这不是 Python 环境问题，而是本地 C++/CUDA 编译环境问题。

## 从 GPU 服务器拷贝到机械臂电脑

在机械臂电脑上执行：

```bash
cd ~
scp zzd@10.11.0.93:/home/zzd/project/wam_local_emptybox_deploy_package/gwp_local_emptybox_deploy.tar.gz .
tar -xzf gwp_local_emptybox_deploy.tar.gz
cd gwp_local_emptybox_deploy
```

至少拷贝 original 50k 权重：

```bash
scp zzd@10.11.0.93:/mnt/data/zzd/giga-world-policy/weights/giga_geek_470_step50000.gguf weights/
```

可选拷贝 prompt_v3 50k 权重：

```bash
scp zzd@10.11.0.93:/mnt/data/zzd/giga-world-policy/weights/giga_geek_470_prompt_v3_256_step50000.gguf weights/
```

传输完成后可以校验：

```bash
sha256sum gwp_local_emptybox_deploy.tar.gz
sha256sum weights/giga_geek_470_step50000.gguf
sha256sum weights/giga_geek_470_prompt_v3_256_step50000.gguf
```

## 在机械臂电脑上安装/补齐依赖

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client

cd ~/gwp_local_emptybox_deploy
bash install_local_runtime_deps.sh
```

如果依赖已经装过，这一步通常不会破坏已有环境。它只是补本地 server/client 需要的 Python 包。

## 在机械臂电脑上编译 4090 版 wam.cpp

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client

cd ~/gwp_local_emptybox_deploy
bash build_wam_cpp_4090.sh
```

成功后应看到类似文件：

```text
wam.cpp/build-cuda/libwam_c_api.so
wam.cpp/build-cuda/wam-validate
```

如果这里失败，优先检查：

```bash
nvidia-smi
nvcc --version
cmake --version
```

## 启动本地 server

先启动 original 50k：

```bash
cd ~/gwp_local_emptybox_deploy
source /opt/ros/humble/setup.bash
conda activate gwp_client

GPU_ID=0 bash run_emptybox_original_server_local.sh
```

如果要测 prompt_v3 50k：

```bash
GPU_ID=0 bash run_emptybox_v3_server_local.sh
```

这两个脚本都默认监听本机：

```text
127.0.0.1:8000
```

## 启动本地 client

另开一个机械臂电脑终端：

```bash
cd ~/gwp_local_emptybox_deploy
source /opt/ros/humble/setup.bash
conda activate gwp_client
```

建议按这个顺序来，不要一上来就长时间闭环：

```bash
# 1. 只检查通讯、相机、server 响应，不动机械臂
MODE=log-only bash run_emptybox_client_local.sh

# 2. 归位
MODE=goto-start SPEED=70 bash run_emptybox_client_local.sh

# 3. 单步动作，确认方向和幅度正常
MODE=step SPEED=70 bash run_emptybox_client_local.sh

# 4. 短闭环测试
MODE=closed-loop DURATION=30 SPEED=70 HZ=30 REPLAN=24 bash run_emptybox_client_local.sh
```

确认稳定后，再逐步加时长或速度。

## 调速建议

之前远程 server 时，动作卡顿主要可能来自跨机器通信和重规划等待。本地 4090 推理后，可以逐步尝试：

```text
保守：SPEED=70  HZ=15  REPLAN=15
中等：SPEED=85  HZ=20  REPLAN=20
较快：SPEED=100 HZ=30  REPLAN=24
```

每次结束时看 client 输出里的：

```text
worst tracking error observed
```

如果 tracking error 明显增大，说明机械臂跟不上当前命令节奏。优先降低 `HZ`，其次降低 `SPEED`。

## 常见问题

### 1. `[ros2] waiting for all camera topics`

检查三路相机是否真的有图像流：

```bash
ros2 topic hz /camera/top/camera/color/image_raw
ros2 topic hz /camera/left/camera/color/image_raw
ros2 topic hz /camera/right/camera/color/image_raw
```

### 2. CAN 报 `CAN port can0 is not UP`

先看状态：

```bash
ip link show can0
ip link show can1
```

需要是 `UP`。如果不是，按机械臂电脑原来的 CAN 启动方式重新拉起 `can0/can1`。

### 3. server 报找不到权重

确认权重放在：

```text
~/gwp_local_emptybox_deploy/weights/giga_geek_470_step50000.gguf
```

或者：

```text
~/gwp_local_emptybox_deploy/weights/giga_geek_470_prompt_v3_256_step50000.gguf
```

### 4. 4090 上编译失败

最常见是机械臂电脑没有 CUDA Toolkit / `nvcc`。仅有显卡驱动和 `nvidia-smi` 不等于能编译 CUDA 代码。

### 5. 本地 4090 推理仍然卡顿

先看 server 日志里的：

```text
infer #... -> chunk (48, 14) in ...s
```

`48` 表示一次输出 48 步动作，`14` 表示动作维度：双臂各 7 维，通常是 6 个关节 + 1 个夹爪。

如果单次推理很快但动作仍顿挫，重点调 `HZ` / `REPLAN` / 机械臂速度和跟踪误差；如果单次推理本身很慢，再看 4090 编译是否启用了 CUDA、是否用到了正确的 `libwam_c_api.so`。

