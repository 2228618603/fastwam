# FastWAM 本地 4090 真机部署实现策略

目标：把目前已验证可用的 `deploy-9-8` 远程 websocket 部署链路，改造成可以整体打包、直接放到真机所在 4090 主机上运行的本地部署包。最终真机侧不再跨机器请求 GPU server，从而去掉网络通信延迟。

本文只描述实现策略和检查点，不做代码实现。

## 0. 当前已知事实

### 可复用的稳定来源

当前可信主线是：

```text
/home/chw/code/packages/FastWAM/deploy-9-8
```

它已经包含：

- `robot_server.py`: 已验证的 FastWAM 模型加载、三相机预处理、state 归一化、action 反归一化逻辑。
- `robot_client.py`: 已验证的 ROS2 相机读取、Piper 状态读取、动作安全检查、log-only / replay / step / closed-loop 流程。
- `goto_start.py`: 独立归位工具。
- 文档：server/client 环境和分阶段真机验证步骤。

最新部署 checkpoint 已切到：

```text
/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt
```

本地 server `--self-test` 已验证：

```text
action chunk shape: (32, 14)
infer time: 0.842s
SELF TEST PASSED
未 OOM
```

### reference 代码的问题

参考目录：

```text
/home/chw/code/packages/FastWAM/deploy-all-on-4090/reference
```

其中旧包路线主要是 cpp/GGUF 本地部署。`数据布局问题复盘.md` 已明确指出历史问题：

```text
旧代码曾把 state 拼成 [左6关节, 右6关节, 左爪, 右爪]
但 action 实际是 [左6关节, 左爪, 右6关节, 右爪]
```

因此旧包和旧 checkpoint 应标记为：

```text
legacy_joint12_grip2
```

它可以参考“打包方式、启动脚本结构、4090 本地化思路”，但不能直接复用其推理口径或模型结果。

本次实现不应沿用旧 cpp/GGUF 路线作为主线。主线应使用 `deploy-9-8` 中已经验证过的 Python FastWAM 推理逻辑，避免重新引入布局错位和模型转换风险。

## 1. 总体方案

最终在：

```text
/home/chw/code/packages/FastWAM/deploy-all-on-4090/code
```

组织一个可直接打包的本地部署目录。建议最终包名类似：

```text
fastwam_local_4090_emptybox_step015000/
```

核心变化：

```text
deploy-9-8:
  robot_client.py  --websocket-->  robot_server.py  --FastWAM--> action

deploy-all-on-4090:
  local_robot_runner.py  --in-process FastWAM--> action
```

也就是说，真机侧一个进程内同时完成：

- ROS2 三路相机读取。
- Piper 双臂状态读取。
- FastWAM 模型推理。
- safety guard。
- log-only / step / closed-loop 动作执行。

这样去掉：

- websocket 序列化/反序列化。
- 图像和状态跨机器传输。
- server/client 两端调度延迟。
- 网络抖动导致的动作卡顿。

保留：

- `deploy-9-8` 的图像预处理。
- `deploy-9-8` 的 state/action layout。
- `deploy-9-8` 的 normalization / denormalization。
- `deploy-9-8` 的安全检查和真机 bring-up 流程。

## 2. 关键设计决策

### 2.1 不加载 text encoder，使用固定任务 text embedding

4090 通常是 24GB 显存，和当前 L20Y 80GB 环境不同。`deploy-9-8/robot_server.py` 当前为了支持在线传入任意 instruction，会设置：

```python
model_cfg["load_text_encoder"] = True
```

这会显著增加显存占用。emptybox 真机部署是固定任务，没必要在线编码文本。

本地 4090 包应改为：

```text
load_text_encoder = false
使用预计算的 context/context_mask
```

需要打包的固定任务 embedding：

```text
/mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711/36a916de41f5bacbbd1a8c25c384fe35f22b6dbcab169f8e97761316eff08a74.t5_len128.wan22ti2v5b.pt
```

大小约：

```text
1.1 MB
```

好处：

- 避免打包 11GB 左右的 T5 text encoder 权重。
- 减少 4090 显存压力。
- 降低启动时间。
- 固定任务语义和训练/评测保持一致。

检查点：

- 本地 runner 的 `infer_action` 必须走 `context/context_mask` 参数，不传 `prompt`。
- package 中必须包含固定任务文本和 embedding 的校验信息。
- 如果未来要支持任意 instruction，再单独做 text encoder 版本，不和本次固定任务部署混在一起。

### 2.2 使用 Python FastWAM，而不是旧 cpp/GGUF

reference 旧包里的 cpp/GGUF 路线有两个问题：

- 历史布局问题已经明确，容易混用 legacy checkpoint / stats。
- 当前最新训练产物是 PyTorch checkpoint，并已用 Python FastWAM 路线验证。

因此本次实现策略：

```text
主线：Python FastWAM local runner
非主线：reference cpp/GGUF 仅参考目录结构和脚本组织
```

不在本阶段做：

- PyTorch checkpoint -> GGUF 转换。
- `wam.cpp` 编译。
- cpp runtime 对齐验证。

### 2.3 保持单机包相对路径

因为目标是打包后 `scp` 到真机 4090，所以包内不能依赖当前机器的绝对路径，例如：

```text
/mnt/data/chw/fastwam/...
/home/chw/code/packages/FastWAM/...
```

代码应统一从包根目录解析相对路径：

```text
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PACKAGE_ROOT / "weights"
MODEL_CACHE_DIR = PACKAGE_ROOT / "model_cache"
CONFIG_DIR = PACKAGE_ROOT / "configs"
ASSET_DIR = PACKAGE_ROOT / "assets"
```

检查点：

- `python code/local_robot_runner.py --print-paths` 能打印所有 resolved path。
- 所有默认路径都指向包内相对路径。
- 不使用绝对 symlink 作为最终交付形式。

## 3. 目标目录结构

建议在 `deploy-all-on-4090/code` 下组织为：

```text
deploy-all-on-4090/code/
├── README.md
├── MANIFEST.json
├── requirements_server_4090.txt
├── requirements_client_4090.txt
├── scripts/
│   ├── pack_local_4090_bundle.sh
│   ├── verify_bundle_integrity.py
│   ├── run_self_test.sh
│   ├── run_local_log_only.sh
│   ├── run_local_step.sh
│   └── run_local_closed_loop.sh
├── app/
│   ├── __init__.py
│   ├── local_robot_runner.py
│   ├── fastwam_local_policy.py
│   ├── robot_io.py
│   ├── safety.py
│   ├── start_pose.py
│   └── layout.py
├── configs/
│   ├── train.yaml
│   ├── data/
│   ├── model/
│   └── task/
├── src/
│   └── fastwam/
├── assets/
│   ├── train_stats.json
│   ├── fixed_task_context.pt
│   ├── task.txt
│   └── start_pose_action14.npy
├── weights/
│   ├── step_015000.pt
│   └── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
└── model_cache/
    ├── Wan-AI/
    │   └── Wan2.2-TI2V-5B/
    └── DiffSynth-Studio/
        └── Wan-Series-Converted-Safetensors/
```

说明：

- `app/` 是本地部署代码，不再有远程 server/client 分裂。
- `src/fastwam/` 是 FastWAM runtime 源码子集或完整 `src/fastwam`。
- `configs/` 是构建模型必须的 Hydra 配置。
- `assets/` 放小文件：归一化统计、固定任务 embedding、任务文本、起始姿态。
- `weights/` 放训练 checkpoint 和 ActionDiT backbone。
- `model_cache/` 放 Wan2.2 diffusion/VAE 基础权重。

## 4. 需要打包的模型/资产

### 必需

```text
weights/step_015000.pt
  来源: /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt
  大小: 约 12GB

weights/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  来源: /mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  大小: 约 2GB

model_cache/Wan-AI/Wan2.2-TI2V-5B/
  来源: /mnt/data/chw/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B/
  主要包含 diffusion_pytorch_model-00001/00002/00003-of-00003.safetensors
  大小: 约 19GB

model_cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
  来源: /mnt/data/chw/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
  大小: 约 1.4GB

assets/train_stats.json
  来源: /mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json
  大小: 约 90KB

assets/fixed_task_context.pt
  来源: /mnt/data/chw/fastwam/text_embeds_cache/agilex_empty_box_542_0711/*.t5_len128.wan22ti2v5b.pt
  大小: 约 1.1MB

assets/task.txt
  内容: emptybox 固定任务文本
```

### 不建议打包

```text
DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
```

原因：

- 约 11GB。
- 只用于在线 text encoding。
- 固定任务部署不需要。
- 会增加 4090 OOM 风险。

### 预计包大小

不含压缩，大约：

```text
12GB checkpoint
+ 2GB ActionDiT
+ 19GB Wan2.2 diffusion
+ 1.4GB VAE
+ 小文件
= 约 34-35GB
```

如果误打包 text encoder，会额外增加约 11GB，总体接近 46GB，并且推理显存风险更高。

## 5. 代码拆分策略

### 5.1 `fastwam_local_policy.py`

职责：封装 FastWAM 模型加载与 action 推理。

从 `deploy-9-8/robot_server.py` 迁移：

- Hydra config compose。
- checkpoint load。
- ActionDiT path override。
- dataset stats load。
- FastWAMProcessor 初始化。
- 三相机 robotwin composite。
- state normalize。
- action denormalize。
- synthetic self-test。

需要修改：

- 不启动 websocket。
- 默认 `load_text_encoder=false`。
- 加载 `assets/fixed_task_context.pt`。
- `infer()` 内调用：

```python
model.infer_action(
    prompt=None,
    context=context,
    context_mask=context_mask,
    input_image=image_tensor,
    action_horizon=32,
    proprio=proprio,
    num_inference_steps=4,
)
```

检查点：

- `python app/local_robot_runner.py --mode self-test --no-ros2` 能返回 `[32,14]`。
- self-test 不加载 T5 text encoder。
- 日志打印 loaded files、dtype、device、inference latency、action min/max。

### 5.2 `robot_io.py`

职责：封装 ROS2 和 Piper。

从 `deploy-9-8/robot_client.py` 迁移：

- `ObservationBuffer`
- `ROS2ObservationCollector`
- `PiperArmJoint`
- `assemble_state14`
- `split_action14`
- `describe_action`

需要保持：

```text
state layout:
[left_j0..j5, right_j0..j5, left_gripper, right_gripper]

action layout:
[left_j0..j5, left_gripper, right_j0..j5, right_gripper]
```

检查点：

- `--mode log-only --no-ros2` 可用 synthetic image + zero state。
- 真机 log-only 不发送动作，只读相机和关节。
- Piper 读取失败不能静默回零，必须失败或明确 warning。

### 5.3 `safety.py`

职责：安全检查。

从 `deploy-9-8/robot_client.py` 迁移：

- `SafetyGuard`
- joint limit
- trajectory smoothness
- tracking error
- gripper clamp
- speed cap
- violation abort

检查点：

- log-only 模式只打印 `OK` / `WOULD-REJECT`，不发命令。
- step / closed-loop 触发 safety violation 时立即 hold。

### 5.4 `start_pose.py`

职责：起始姿态。

从 `robot_client.py` / `goto_start.py` 迁移：

- 内置 `START_POSE_ACTION14`。
- `load_start_pose(parquet_path)`。
- `goto_start_pose()`。

需要在 `assets/` 中额外保存：

```text
start_pose_action14.npy
```

检查点：

- `--goto-start` 必须要求 `--confirm-safety`。
- 默认起始姿态要和当前训练数据口径一致，不再引用 old 470/legacy 口径。

### 5.5 `local_robot_runner.py`

职责：本地单进程入口。

建议支持模式：

```text
--mode self-test       # 加载模型，synthetic image + zero state，一次推理后退出
--mode log-only        # 真实相机/真实状态，推理并打印，不发动作
--mode replay          # 不加载模型，回放 parquet 动作
--mode step            # 低速执行有限步数
--mode closed-loop     # 限时闭环
```

命令风格尽量继承 `deploy-9-8/robot_client.py`：

```bash
python app/local_robot_runner.py --mode log-only
python app/local_robot_runner.py --mode step --max-steps 1 --goto-start --confirm-safety
python app/local_robot_runner.py --mode closed-loop --confirm-safety --i-am-watching --max-duration 30
```

不再需要：

```text
--server ws://...
```

检查点：

- argparse 中没有 `--server` 必填。
- 所有 inference mode 都使用本地 `FastWAMLocalPolicy`。
- replay 模式不加载模型，便于单独验证机械臂单位转换。

## 6. 环境策略

4090 真机侧会同时需要：

- ROS2 / cv_bridge / Piper SDK。
- torch + CUDA 推理栈。
- FastWAM Python 依赖。

这和 `deploy-9-8` 远程部署不同：原来 client 环境不需要 torch，server 环境不需要 ROS2/Piper；现在本地单机需要两者合并。

建议策略：

```text
优先使用一个 conda 环境: fastwam4090
Python: 3.10
Torch: 和目标 CUDA/driver 匹配，优先 torch 2.7.1 + cu128
NumPy: 需要重点验证 ROS2 cv_bridge ABI
```

风险点：

- `fastwam` 依赖当前是 `numpy==2.2.6`。
- ROS2 Humble / cv_bridge 往往对 `numpy<2` 更稳。
- 如果真机 ROS2 环境无法接受 NumPy 2，需要拆成两进程本地 IPC 或找兼容版本。

检查点：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import rclpy, cv_bridge; print('ros2 ok')"
python -c "import piper_sdk; print('piper ok')"
python -c "import fastwam; print('fastwam ok')"
```

如果单环境无法同时满足 ROS2 和 torch，备选方案：

```text
同一台 4090 主机上保留本地 websocket 或 Unix domain socket：
  local_model_server.py 负责 torch/FastWAM
  local_robot_client.py 负责 ROS2/Piper

这仍然避免跨机器网络延迟，但不是完全 in-process。
只有在 NumPy/ROS2 ABI 冲突无法解决时才采用。
```

## 7. 打包策略

### 7.1 生成 bundle

实现时新增：

```text
scripts/pack_local_4090_bundle.sh
```

职责：

- 创建 `fastwam_local_4090_emptybox_step015000/`。
- 拷贝 `app/`、`configs/`、`src/fastwam/`、文档和启动脚本。
- 拷贝必要 weights/assets。
- 写入 `MANIFEST.json`。
- 计算 sha256。
- 可选压缩成 `.tar.zst` 或 `.tar.gz`。

注意：

- 不使用绝对 symlink。
- 不拷贝 `.git`、`__pycache__`、训练 logs、eval result。
- 不拷贝无关 checkpoint。

### 7.2 MANIFEST

`MANIFEST.json` 至少记录：

```json
{
  "package": "fastwam_local_4090_emptybox_step015000",
  "checkpoint": "weights/step_015000.pt",
  "checkpoint_source": "/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt",
  "dataset_stats": "assets/train_stats.json",
  "text_context": "assets/fixed_task_context.pt",
  "task": "agilex_empty_box_uncond_3cam384",
  "dataset_layout": "aligned_default_14d_state_action_l6g_r6g",
  "action_horizon": 32,
  "num_inference_steps_default": 4,
  "created_at": "...",
  "files": {
    "relative/path": {
      "bytes": 123,
      "sha256": "..."
    }
  }
}
```

检查点：

- 目标机器上运行 `scripts/verify_bundle_integrity.py` 能校验所有大文件。
- package 解压后不依赖源机器路径。

## 8. 分步实施计划

### Step 1. 冻结部署口径

要先固定以下内容：

```text
checkpoint: step_015000.pt
task: agilex_empty_box_uncond_3cam384
action horizon: 32
num inference steps: 4
state layout: [L6, R6, Lg, Rg]
action layout: [L6, Lg, R6, Rg]
text input: fixed precomputed context
```

验收：

- 文档写清楚这些口径。
- `MANIFEST.json` 有对应字段。
- 代码启动时打印这些字段。

### Step 2. 抽取本地 policy

从 `deploy-9-8/robot_server.py` 抽出：

```text
FastWAMLocalPolicy
```

验收：

```bash
python app/local_robot_runner.py --mode self-test --no-ros2
```

预期：

```text
Loaded checkpoint: weights/step_015000.pt
Loaded fixed context: assets/fixed_task_context.pt
action chunk shape: (32, 14)
SELF TEST PASSED
```

### Step 3. 抽取 robot I/O 和 safety

从 `deploy-9-8/robot_client.py` 拆出：

```text
robot_io.py
safety.py
start_pose.py
layout.py
```

验收：

```bash
python app/local_robot_runner.py --help
python app/local_robot_runner.py --mode log-only --no-ros2
```

预期：

- 不需要 `--server`。
- synthetic 灰图 + zero state 可跑通。
- 不发送任何动作。

### Step 4. 组织权重和资源

拷贝必要文件到 `weights/`、`assets/`、`model_cache/`。

验收：

```bash
python scripts/verify_bundle_integrity.py
python app/local_robot_runner.py --mode self-test --no-ros2 --device cuda:0
```

预期：

- sha256 全部通过。
- 4090 显存不 OOM。
- self-test latency 可接受。

### Step 5. 真机侧 dry-run

在目标 4090 真机上：

```bash
source /opt/ros/humble/setup.bash
conda activate fastwam4090
cd ~/fastwam_local_4090_emptybox_step015000
python app/local_robot_runner.py --mode log-only
```

验收：

- 三路相机 ready。
- Piper 状态读取正常。
- 模型返回 action chunk `(32, 14)`。
- log-only 只打印动作，不发送控制。

### Step 6. 真机低速动作

先 replay，再模型单步：

```bash
python app/local_robot_runner.py \
  --mode replay \
  --episode-parquet <episode.parquet> \
  --confirm-safety \
  --speed-percent 10 \
  --action-hz 10

python app/local_robot_runner.py \
  --mode step \
  --max-steps 1 \
  --goto-start \
  --confirm-safety \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 1
```

验收：

- replay 方向和单位正确。
- step 只执行 1 个 action。
- safety guard 不报 joint jump / tracking error。
- 执行后 hold current pose。

### Step 7. 短闭环

```bash
python app/local_robot_runner.py \
  --mode closed-loop \
  --goto-start \
  --confirm-safety \
  --i-am-watching \
  --max-duration 30 \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 8
```

验收：

- 30 秒内无 stale observation。
- 无连续 safety violation。
- tracking error 不持续接近上限。
- 本地 inference latency 明显低于远程链路总延迟。

## 9. 最终启动命令规划

### 本地模型 self-test

```bash
cd ~/fastwam_local_4090_emptybox_step015000
source /opt/ros/humble/setup.bash
conda activate fastwam4090
CUDA_VISIBLE_DEVICES=0 python app/local_robot_runner.py \
  --mode self-test \
  --no-ros2 \
  --num-inference-steps 4
```

### 本地 log-only

```bash
cd ~/fastwam_local_4090_emptybox_step015000
source /opt/ros/humble/setup.bash
conda activate fastwam4090
CUDA_VISIBLE_DEVICES=0 python app/local_robot_runner.py \
  --mode log-only \
  --num-inference-steps 4
```

### 本地单步

```bash
cd ~/fastwam_local_4090_emptybox_step015000
source /opt/ros/humble/setup.bash
conda activate fastwam4090
CUDA_VISIBLE_DEVICES=0 python app/local_robot_runner.py \
  --mode step \
  --max-steps 1 \
  --goto-start \
  --confirm-safety \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 1 \
  --num-inference-steps 4
```

### 本地短闭环

```bash
cd ~/fastwam_local_4090_emptybox_step015000
source /opt/ros/humble/setup.bash
conda activate fastwam4090
CUDA_VISIBLE_DEVICES=0 python app/local_robot_runner.py \
  --mode closed-loop \
  --goto-start \
  --confirm-safety \
  --i-am-watching \
  --max-duration 30 \
  --speed-percent 10 \
  --action-hz 10 \
  --replan-steps 8 \
  --num-inference-steps 4
```

## 10. 风险和应对

### 风险 A: 4090 OOM

原因：

- 4090 只有 24GB。
- FastWAM + Wan2.2 video expert + VAE + action expert 显存压力高。
- 加载 text encoder 会明显增加风险。

应对：

- 固定任务使用预计算 context，不加载 text encoder。
- 默认 `bf16`。
- 默认 `num_inference_steps=4`。
- 只走 action-only inference，不生成视频帧。
- self-test 阶段记录峰值显存。

通过标准：

```text
self-test 能连续跑 5 次，不 OOM，峰值显存低于 4090 可用显存并留有余量。
```

### 风险 B: ROS2 / NumPy / torch 环境冲突

原因：

- FastWAM 当前依赖 NumPy 2.2.6。
- ROS2 Humble / cv_bridge 对 NumPy 版本可能敏感。

应对：

优先尝试单环境。如果失败，退到同机双进程本地 IPC：

```text
model process: torch + FastWAM
robot process: ROS2 + Piper
communication: localhost websocket 或 Unix socket
```

这仍然比跨机器低延迟，并且隔离 ABI 冲突。

### 风险 C: state/action layout 再次混用

应对：

- `layout.py` 集中定义 layout。
- 启动日志打印 layout。
- self-test 打印 state/action 维度解释。
- package 名和 manifest 标记：

```text
aligned_default_14d_state_action_l6g_r6g
```

### 风险 D: 目标机器缺基础模型或路径错误

应对：

- 所有权重进 package 相对路径。
- `verify_bundle_integrity.py` 先跑。
- `--print-paths` 打印 resolved path。

### 风险 E: 本地推理快了，但机械臂跟不上

应对：

- 不因为延迟降低就直接提高速度。
- 初始仍用：

```text
speed_percent = 10
action_hz = 10
replan_steps = 1 or 8
```

- 观察 worst tracking error 后再逐步调高。

## 11. 不在本轮做的事情

本轮用户只要求实现策略，因此暂不做：

- 不写 `local_robot_runner.py`。
- 不拷贝大模型权重。
- 不生成 tar 包。
- 不改 `deploy-all-on-4090/code`。
- 不在真机上执行命令。
- 不做 cpp/GGUF 转换。

审查通过后，下一轮实现顺序建议：

```text
1. 先创建 code/app 的本地 runner 最小版本，只支持 self-test 和 log-only --no-ros2。
2. 本机验证最新 checkpoint + fixed context 不 OOM。
3. 再接入 ROS2/Piper 和 motion modes。
4. 最后组织 weights/assets/model_cache 并生成 manifest。
```
