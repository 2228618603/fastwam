# GWP aligned 真机部署说明

> 创建时间（北京时间）：2026-09-11 18:02

本目录只面向修正后的 14 维布局：

```text
state/action = [左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]
```

旧部署代码已原样备份到：

```text
/home/zzd/project/giga-world-policy/deploy_old_legacy_joint12_grip2
```

旧 checkpoint、旧 GGUF 和旧 `norm_stats.json` 属于 `legacy_joint12_grip2`，不能与本目录代码混用。

## 当前状态

- aligned emptybox 100k 基线已经完成。
- Python 推理服务和机器人客户端已经准备好，均使用修正后的 14 维布局。
- 默认目标 checkpoint：

```text
/mnt/data/models/zzd/checkpoint/giga_470_aligned_l6g_r6g_bs128_gpu8_b16g1_100k/models/checkpoint_epoch_13_step_100000/transformer_ema
```

- 默认 norm stats：

```text
/mnt/data/zzd/giga-world-policy/geek_data/norm_stats_aligned_l6g_r6g.json
```

- aligned GGUF 需要由 aligned checkpoint + aligned norm stats 重新转换。旧 GGUF 不可混用。
  已提供 `start_server_aligned_cpp.sh`，但只应加载文件名包含 `aligned_l6g_r6g` 的 GGUF。

## 文件

| 文件 | 运行位置 | 用途 |
|---|---|---|
| `robot_server.py` | GPU 服务器或带 4090 的机械臂电脑 | Python/cpp 推理服务；当前推荐 Python aligned。 |
| `robot_client.py` | 机械臂电脑 | ROS2 相机、Piper CAN、动作执行和安全检查。 |
| `goto_start.py` | 机械臂电脑 | 不加载模型，单独归位。 |
| `start_server_aligned_python.sh` | 模型所在机器 | 启动 aligned Python 服务。 |
| `start_server_aligned_cpp.sh` | 模型所在机器 | 启动 aligned cpp/GGUF 服务。 |
| `start_client.sh` | 机械臂电脑 | 按 `log-only → goto-start → step → closed-loop` 启动。 |
| `install_client_env.sh` | 机械臂电脑 | 补齐轻量 client 依赖。 |

## 为什么不能用旧 cpp/GGUF

GGUF 不只是 Transformer 权重，还内嵌归一化统计和 PolicySpec。旧文件使用旧 state 语义，
即使模型文件可以加载，输入也会按错误参考系归一化。必须使用从 aligned checkpoint、
aligned norm stats 重新转换的 GGUF。

## 1. 检查 aligned checkpoint

确认最终目录存在：

```bash
test -d /mnt/data/models/zzd/checkpoint/\
giga_470_aligned_l6g_r6g_bs128_gpu8_b16g1_100k/models/\
checkpoint_epoch_13_step_100000/transformer_ema
```

如果最终 epoch 编号不同，以实际 `checkpoint_*_step_100000/transformer_ema` 为准，并通过
`CHECKPOINT=...` 覆盖启动脚本。

## 2. GPU 服务器启动 Python 服务

先只做自检，不监听端口：

```bash
cd /home/zzd/project/giga-world-policy
SELF_TEST=1 GPU_ID=0 bash deploy/start_server_aligned_python.sh
```

自检通过后启动服务：

```bash
GPU_ID=0 PORT=8000 bash deploy/start_server_aligned_python.sh
```

默认使用固定 original task T5 token，不加载完整 T5 encoder。需要测试 `torch.compile` 时：

```bash
ENABLE_COMPILE=1 GPU_ID=0 PORT=8000 bash deploy/start_server_aligned_python.sh
```

首次 compile 会明显变慢，应在 `log-only` 阶段完成预热后再允许机械臂运动。

## 3. GPU 服务器启动 cpp 服务

先确认 GGUF 文件存在。aligned 100k 默认路径：

```text
/mnt/data/zzd/giga-world-policy/weights/giga_470_aligned_l6g_r6g_step100000.gguf
```

自检：

```bash
cd /home/zzd/project/giga-world-policy
SELF_TEST=1 GPU_ID=0 bash deploy/start_server_aligned_cpp.sh
```

启动服务：

```bash
GPU_ID=0 PORT=8000 bash deploy/start_server_aligned_cpp.sh
```

切换到 30k：

```bash
GGUF=/mnt/data/zzd/giga-world-policy/weights/giga_470_aligned_l6g_r6g_step30000.gguf \
GPU_ID=0 PORT=8000 bash deploy/start_server_aligned_cpp.sh
```

## 4. 机械臂电脑环境

已有 `gwp_client` 环境并能导入 ROS2、Piper SDK 时不必重建，只需补依赖：

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd <新部署包>/deploy
bash install_client_env.sh
```

确认 CAN 和相机：

```bash
ip link show can0
ip link show can1
ros2 topic hz /camera/top/camera/color/image_raw
ros2 topic hz /camera/left/camera/color/image_raw
ros2 topic hz /camera/right/camera/color/image_raw
```

## 5. 分级真机验证

在机械臂电脑的 `deploy/` 下运行。远程 GPU 服务示例地址为 `10.11.0.93:8000`；如果 server
就在机械臂电脑的 4090 上，将 `SERVER=ws://127.0.0.1:8000`。

```bash
# 只看相机、state 和模型响应，不动机械臂
SERVER=ws://10.11.0.93:8000 MODE=log-only bash start_client.sh

# 单独归位
MODE=goto-start SPEED=30 bash start_client.sh

# 只执行一个模型动作
SERVER=ws://10.11.0.93:8000 MODE=step SPEED=30 bash start_client.sh

# 人员全程观察下进行 30 秒闭环
SERVER=ws://10.11.0.93:8000 MODE=closed-loop \
DURATION=30 SPEED=70 HZ=15 REPLAN=15 bash start_client.sh
```

只有前三步的方向、夹爪和幅度全部正确后，才进入 closed-loop。

## RTC 与部署的关系

- aligned RTC checkpoint 还没有训练。
- 当前同步 `closed-loop` 可以加载 RTC checkpoint，但相当于 `d=0`，不会体现异步 RTC 收益。
- `scripts/inference_openloop.py` 已支持 `action_prefix + delay` 的 Python 推理。
- 当前 `deploy/robot_client.py` 尚未实现异步请求和已承诺动作前缀协议。
- wam.cpp 的展开去噪图也尚未实现逐步 prefix clamp。

因此，RTC 模型训练完成后应先做离线 d-sweep，再单独实现和验证异步真机 client；不要直接把
RTC checkpoint 当成“换个路径即可获得异步加速”。

## 安全提示

- 默认 client 模式是 `log-only`。
- `step` 和 `closed-loop` 都需要显式安全确认，脚本会传入必要 flag。
- 首次测试使用 `SPEED=30`、`DURATION=30`，确认跟踪误差后再提速。
- 退出时的 ROS2 core dump 若发生在双臂已 hold 后通常是清理问题，但仍应确认机械臂实际停止。
- `--gripper-scale 0.105` 仍需以实物行程校准，不能仅凭旧部署经验认定正确。
