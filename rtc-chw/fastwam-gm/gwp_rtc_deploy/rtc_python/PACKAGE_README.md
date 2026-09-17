# aligned emptybox 本地 4090 部署包

> 创建时间（北京时间）：2026-09-11 18:04

本目录替代旧的 legacy GGUF 包。旧包已备份为：

```text
/home/zzd/project/wam_local_emptybox_deploy_package_old_legacy_joint12_grip2
```

## 当前结论

- aligned 100k checkpoint 已完成，新包可以直接转移到机械臂电脑。
- 新包使用修正后的 14 维布局，和 aligned 训练代码一致：
  `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`。
- 旧 `giga_geek_470_step50000.gguf` 不能放进新包，它内嵌的是 legacy 归一化。
- Python/BF16 是当前最稳的真机路径；aligned GGUF 只加载文件名包含 `aligned_l6g_r6g`
  的新转换版本。

## 生成代码包

```bash
cd /home/zzd/project/wam_local_emptybox_deploy_package
bash build_python_package.sh
```

输出：

```text
gwp_aligned_emptybox_python_deploy.tar.gz
```

代码包不包含模型权重。它包含：

```text
deploy/
scripts/inference_openloop.py
world_action_model/
third_party/giga-models/
prompt_tokens/fixed_task_token.pt
requirements.txt
```

## 权重需要额外复制

aligned Transformer：

```text
/mnt/data/models/zzd/checkpoint/giga_470_aligned_l6g_r6g_bs128_gpu8_b16g1_100k/models/
checkpoint_epoch_13_step_100000/transformer_ema
```

Wan2.2 Diffusers 基座：

```text
/mnt/data/zzd/giga-world-policy/Wan2.2-TI2V-5B-Diffusers
```

aligned norm stats：

```text
/mnt/data/zzd/giga-world-policy/geek_data/norm_stats_aligned_l6g_r6g.json
```

aligned GGUF（cpp 后端，转换完成后使用）：

```text
/mnt/data/zzd/giga-world-policy/weights/giga_470_aligned_l6g_r6g_step30000.gguf
/mnt/data/zzd/giga-world-policy/weights/giga_470_aligned_l6g_r6g_step100000.gguf
```

建议在机械臂电脑上组织成：

```text
~/gwp_aligned_emptybox_python_deploy/
├── deploy/
├── scripts/
├── world_action_model/
├── third_party/giga-models/
├── prompt_tokens/
├── models/aligned_transformer_ema/
├── models/Wan2.2-TI2V-5B-Diffusers/
├── models/norm_stats_aligned_l6g_r6g.json
└── models/giga_470_aligned_l6g_r6g_step100000.gguf   # 可选，cpp 后端
```

然后覆盖启动路径：

```bash
cd ~/gwp_aligned_emptybox_python_deploy
CHECKPOINT=$PWD/models/aligned_transformer_ema \
BASE_MODEL=$PWD/models/Wan2.2-TI2V-5B-Diffusers \
NORM_STATS=$PWD/models/norm_stats_aligned_l6g_r6g.json \
PYTHON=$(which python) GPU_ID=0 PORT=8000 \
bash deploy/start_server_aligned_python.sh
```

cpp 后端（需要先在机械臂电脑上编译好 `wam.cpp` 的 CUDA 库）：

```bash
cd ~/gwp_aligned_emptybox_python_deploy
GGUF=$PWD/models/giga_470_aligned_l6g_r6g_step100000.gguf \
WAM_LIB=$PWD/wam.cpp/build-cuda/libwam_c_api.so \
PYTHON=$(which python) GPU_ID=0 PORT=8000 \
bash deploy/start_server_aligned_cpp.sh
```

客户端流程见包内 `deploy/README.md`。先跑 `log-only`，不得直接进入长时间闭环。

## 环境说明

此前机械臂电脑的 `gwp_client` 只负责 ROS2/Piper client，不一定包含 PyTorch、Diffusers 和
Transformer 依赖。若要在 4090 本地同时运行 Python server，建议单独创建/复用完整的 `giga`
推理环境；不要为了 server 破坏已经能控制机械臂的 client 环境。

推荐单独创建 server 环境：

```bash
conda create -n gwp_server python=3.11 -y
conda activate gwp_server
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`gwp_client` 继续保持 `numpy<2`；`gwp_server` 使用包内推理依赖。两者不要混装。
