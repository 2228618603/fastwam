# GPU 服务器端环境说明

这份文档只回答一个问题：跑 `deploy-9-8/robot_server.py` 需要什么环境、怎么检查、缺了怎么补。

## 1. 结论

服务器端直接使用本机已有 conda 环境 `fastwam`，不要用 `base`。本次部署默认使用
2026-09-16 最新训练 checkpoint `step_015000.pt`；换代码、换权重、换 stats、重装环境后都要重跑 `--self-test`。

先用 `nvidia-smi` 选一张最空的卡。2026-09-16 本机 8 卡都有任务，我用 GPU 4 做过低影响 self-test；
下面以 GPU 4 为例：

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
CUDA_VISIBLE_DEVICES=4 python deploy-9-8/robot_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --num-inference-steps 4
```

## 2. 常见报错与原因

在 `(base)` 环境下直接运行会出现：

```
ModuleNotFoundError: No module named 'numpy'
```

原因：`base` 环境是 Python 3.14，不装项目依赖；服务器脚本依赖 numpy / torch / hydra 等推理栈，这些都已经装在 `fastwam` 环境里。这不是代码问题，是环境没选对。

## 3. 环境要求（已在 `fastwam` 环境验证通过）

- Python：3.10（`requires-python = ">=3.10"`）
- 关键包（版本与 `pyproject.toml` 一致）：

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.2.6 | 图像拼接 / 状态归一化 |
| torch | 2.7.1+cu128 | 模型推理（CUDA） |
| torchvision | 0.22.1+cu128 | 随 torch 安装 |
| hydra-core | 1.3.2 | 读取 `configs/` 下的 train/task 配置 |
| omegaconf | 2.3.0 | hydra 依赖 |
| pillow | 12.0.0 | 三相机图像 resize |
| websockets | 16.1.1 | websocket 服务 |
| msgpack / msgpack-numpy | — | 请求/响应序列化 |
| einops / transformers / safetensors / accelerate | 见 pyproject | 模型结构、文本编码器、权重加载 |

- 代码依赖：`fastwam` 包源码（`/home/chw/code/packages/FastWAM/src/`）。`robot_server.py` 启动时会自动把项目根目录和 `src/` 加进 `sys.path`，不需要额外 `pip install -e .` 也能跑。

## 4. 环境检查清单

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
python --version   # 应为 3.10.x
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
# 期望: 2.2.6 2.7.1+cu128 True

python -c "import hydra, omegaconf, PIL, websockets, msgpack, msgpack_numpy, einops, transformers; print('deps ok')"
```

GPU 检查（服务器推理必须走 CUDA）：

```bash
nvidia-smi   # 能看到 L20Y 等 GPU，显存未占满
ss -ltnp | grep ':8000' || true
```

启动时建议显式设置 `CUDA_VISIBLE_DEVICES=<选中的单卡>`，避免抢占已有任务。不要直接暴露 8 张卡给部署服务。

## 5. 模型和数据文件检查

```bash
cd /home/chw/code/packages/FastWAM
ls -lh /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt
ls -lh /mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json
ls -lh checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
test -d checkpoints/Wan-AI/Wan2.2-TI2V-5B
test -d checkpoints/Wan-AI/Wan2.1-T2V-1.3B
test -d checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors
head -n 1 /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711/meta/tasks.jsonl
```

说明：

- `step_015000.pt` 是当前默认部署 checkpoint。
- `dataset_stats.json` 是归一化统计，必须与 checkpoint 匹配。
- `ActionDiT` 和 Wan 基础模型缓存用于构建模型骨架。
- 如果训练数据目录不可用，需要在启动 server 时传 `--instruction "<任务文本>"`。

## 6. 自测命令

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
CUDA_VISIBLE_DEVICES=4 python deploy-9-8/robot_server.py \
  --self-test \
  --num-inference-steps 4 \
  --cuda-memory-fraction 0.32
```

`--self-test` 跑完后应看到：

- `Loaded trained checkpoint`
- `Loaded normalization stats`
- `action chunk shape: (32, 14)`（本数据集 `num_frames=33`，action horizon 为 32）
- `=== SELF TEST PASSED ===`
- 进程退出码为 0

## 7. 正式服务和协议检查

启动正式服务：

```bash
cd /home/chw/code/packages/FastWAM
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
CUDA_VISIBLE_DEVICES=4 python deploy-9-8/robot_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --num-inference-steps 4
```

另开一个终端检查：

```bash
ss -ltnp | grep ':8000'
python - <<'PY'
import msgpack, msgpack_numpy
from websockets.sync.client import connect
msgpack_numpy.patch()
with connect("ws://127.0.0.1:8000", open_timeout=10, max_size=None) as ws:
    ws.send(msgpack.packb({"ping": True}, use_bin_type=True))
    resp = msgpack.unpackb(ws.recv(), raw=False)
    print(resp["ok"], resp["info"]["action_horizon"], resp["info"]["camera_keys"])
PY
```

期望输出包含：

- `True 32 ['top', 'left_wrist', 'right_wrist']`

## 8. 如果 `fastwam` 环境丢失 / 需要重建

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
# torch 用 cu128 源
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# 其余依赖按 pyproject.toml 装，或直接：
cd /home/chw/code/packages/FastWAM
pip install -e .
# 网络服务额外需要（pyproject 未列出）：
pip install websockets msgpack msgpack-numpy
```

> 注意：`pyproject.toml` 里的依赖是训练用的全集；纯推理最少只需要上面第 3 节表格里的包。但为了和训练环境一致、避免版本漂移，重建时建议按 `pyproject.toml` 完整安装。

## 9. 验证记录（2026-09-16，最新 checkpoint）

| 检查项 | 结果 |
|---|---|
| `conda activate fastwam` 后依赖导入 | 全部 OK（numpy 2.2.6 / torch 2.7.1+cu128 / hydra / websockets / msgpack ...） |
| 模型和 stats 文件 | `step_015000.pt` 与 `/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json` 可读 |
| 任务文本 | 从训练数据 `meta/tasks.jsonl` 读取成功 |
| `--self-test` | `SELF TEST PASSED`，退出码 0，chunk `(32, 14)`，单次推理 `0.842s`，未 OOM |
| 正式服务模式 | 日志出现 `serving on ws://0.0.0.0:8000` |
| websocket ping 往返 | `ok: True`，返回 horizon 32 和三路 camera keys |
| websocket infer 往返 | 返回 action chunk `(32, 14)`，全为有限值，单次推理约 0.94s |
| `robot_client.py --mode log-only --no-ros2` | 连续 3 次 synthetic 推理成功 |
| 语法检查 | `python -m compileall -q deploy-9-8/robot_server.py deploy-9-8/robot_client.py deploy-9-8/goto_start.py` 通过 |

## 10. 可以跳过的服务端步骤

可以跳过：

- 同机、同代码、同 checkpoint/stats、同环境下，已经通过后不必每次启动前重复依赖导入检查和 `--self-test`。
- 本机已完成 websocket synthetic smoke 后，真实联调可以直接让机器人侧 client 做真实 `log-only`。
- 不使用 openpi / `deploy_real.py` 时，不需要启动任何 cpp / GGUF 服务。

不能跳过：

- 每次启动前确认 GPU 和端口。
- 换代码、换 checkpoint、换 stats、重装环境后重跑 `--self-test`。
- 真机联调前启动正式 server 并确认 `serving on ws://0.0.0.0:8000`。

## 11. 一个已修复的坑：state 配置与 checkpoint 统计不匹配

**现象**：环境配好后（`fastwam` env），自测仍报

```
KeyError: 'default'   (fastwam_processor.py set_normalizer_from_stats)
```

**原因**：历史 checkpoint 的 state 可能是 `[joint 12D, gripper 2D]` 两个分量，而仓库里
`configs/data/agilex_empty_box.yaml` 当前声明的是单个 `default` 14D，两者对不上时，
normalizer 按 `default` 去 stats 里取键就会报错。最新 `step_015000.pt` 使用的 stats
是单 `default` 14D，但 server 仍保留自动适配逻辑，方便兼容历史 checkpoint。

**修复**：在 `deploy-9-8/robot_server.py` 的 `FastWAMActionServer.__init__` 里，compose
完 config 后以 `dataset_stats.json` 为准重建 `shape_meta.state`（从 stats 的 state
键和 `global_mean` 维度推导），再实例化 processor。这样归一化与训练时完全一致，且不依赖
yaml 里 state 的写法；以后换用别的 checkpoint/stats（比如单 `default` 14D 的）也能自动
适配。改动只在 `deploy-9-8/` 内，未动上游代码。

## 12. 一个已清理的告警：msgpack state 只读数组

**现象**：websocket infer 时出现 PyTorch warning：

```text
The given NumPy array is not writable
```

**原因**：`msgpack-numpy` 解码后的 state 可能是只读 view。

**修复**：server 请求入口已改成 `np.array(req["state"], dtype=np.float32, copy=True)`，让传给 torch 的 state 拥有可写内存。这个改动不改变协议和数值。

## 13. 服务器端 vs 客户端环境（不要混用）

| | 服务器端（本机） | 客户端（机器人侧） |
|---|---|---|
| conda 环境 | `fastwam` | `gwp_client` |
| Python | 3.10 | 3.10 |
| numpy | 2.x | **< 2**（ROS2 cv_bridge 要求） |
| torch / CUDA | 需要 | 不需要 |
| 入口 | `deploy-9-8/robot_server.py` | `deploy-9-8/robot_client.py` |

客户端环境的完整说明见同目录 `../client_environment.md`，启动顺序见 `../deploy_tutorial.md`。

## 14. 其他常见问题

- **`Checkpoint not found` / `dataset_stats.json not found`**：确认最新权重
  `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt`
  和 stats `/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json` 都存在。
- **hydra 报 `ConfigAttributeError` / 找不到 task**：确认 `--task` 名称与 `configs/task/` 下的 yaml 文件名一致（本任务为 `agilex_empty_box_uncond_3cam384`）。
- **CUDA 不可用**：脚本会自动回落到 CPU 并打 warning，但推理会非常慢，真机部署不可接受，先修 GPU 环境。
- **训练数据目录**：服务端读取任务文本（`meta/tasks.jsonl`）需要访问 `dataset_dirs` 指向的训练数据（当前指向 `/mnt/data/dataset/...`），如果该目录不可用，启动时会报错，可改用 `--instruction` 显式传入任务文本。
