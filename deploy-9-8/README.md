# FastWAM 真机部署 9-8

这是一套独立于原始 FastWAM 仓库的真机部署副本。新增部署入口都放在本目录，当前主线只走 Python FastWAM websocket 服务，不走 cpp / GGUF。

## 当前结论

2026-09-09 已在本机 `fastwam` 环境验证通过：

- server `--self-test`：通过，输出 action chunk `(32, 14)`。
- server 正式监听 `0.0.0.0:8000`：通过。
- 本机 websocket `ping/reset/infer`：通过。
- `robot_client.py --mode log-only --no-ros2` synthetic smoke：通过。

这说明 GPU server 和协议链路可以跑通。真实相机、Piper 读状态、归位、单步动作仍必须在机器人侧逐级验证。

## 文件用途

| 文件 | 运行位置 | 用途 |
|---|---|---|
| `robot_server.py` | GPU 服务器 | 加载 FastWAM checkpoint，提供 websocket 推理服务 |
| `robot_client.py` | 机器人侧 client | 采集 ROS2 相机和 Piper 状态，连接 server，执行 log-only / step / closed-loop |
| `goto_start.py` | 机器人侧 client | 只做归位，不推理；可选工具，优先使用 `robot_client.py --goto-start` |
| `deploy_tutorial.md` | 两端 | 按阶段联调和验收 |
| `note/server_environment.md` | GPU 服务器 | server 环境、模型文件和本机验证记录 |
| `client_environment.md` | 机器人侧 client | ROS2 / Piper / client 依赖检查 |

## 默认模型

- checkpoint: `/mnt/data/chw/fastwam/cp-copy/step_020000.pt`
- dataset stats: `/mnt/data/chw/fastwam/cp-copy/dataset_stats.json`
- task config: `agilex_empty_box_uncond_3cam384`
- ActionDiT backbone: `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- Wan 基础模型缓存: `checkpoints/Wan-AI/` 和 `checkpoints/DiffSynth-Studio/`

## GPU Server 启动

本机 GPU 0-3 当前常被占用，建议先用空闲 GPU 4：

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

首次部署、换代码、换 checkpoint、换 stats、重装环境后先跑自测：

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

## Client 启动

只做本机协议 smoke test，可以跳过 ROS2/Piper，使用 synthetic 灰图和零状态：

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
python deploy-9-8/robot_client.py \
  --server ws://127.0.0.1:8000 \
  --mode log-only \
  --no-ros2
```

真实机器人侧 log-only 不能加 `--no-ros2`，需要读取真实相机和真实关节状态：

```bash
source /opt/ros/humble/setup.bash
conda activate gwp_client
cd /home/chw/code/packages/FastWAM
python deploy-9-8/robot_client.py \
  --server ws://<GPU_SERVER_IP>:8000 \
  --mode log-only
```

## 可跳过与不可跳过

可跳过：

- 同一台 GPU 机器、同一份代码、同一 checkpoint/stats 下，已经自测通过后，不必每次启动前重复 `--self-test`。
- 本机已经做过 `--mode log-only --no-ros2` 后，真实联调时不必重复 synthetic smoke test。
- 如果直接使用 `robot_client.py --goto-start`，可以不单独运行 `goto_start.py`。
- 不使用 openpi / `deploy_real.py` 路线时，可以忽略 `client_environment.md` 里的路线 B。

不可跳过：

- 启动前确认端口和 GPU 空闲。
- 真实机器人侧 `--mode log-only`，用于确认 ROS2 相机、Piper 状态读取、server 返回动作范围。
- 第一次运动必须先低速 `--mode step --max-steps 1 --goto-start --confirm-safety`。
- 闭环必须带 `--confirm-safety --i-am-watching --max-duration`。
- 真机前必须确认 `--gripper-scale 0.105` 是否符合实际夹爪行程；当前高于约 `0.667` 的模型开度会被硬件上限夹住。

完整上线流程见 `deploy_tutorial.md`。
