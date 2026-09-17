# aligned RTC 工程优化（2026-09-17）

把 `/home/gaomeng/giga-world-policy-self` 的工程加速迁入 zzd 当前 RTC 部署。
默认继续使用现有 **RTC step30000 transformer BF16** 权重、aligned norm 和固定任务 token。
RTC step50000 已训练完成，可用同一套工程优化代码通过 `start_rtc50k_python_server.sh`
切换测试。没有切换到小伙伴的 checkpoint。

## 改了什么

| 文件 | 修改与原因 |
|---|---|
| `scripts/inference_openloop.py` | `compile_policy_action_blocks(..., compile_prefix=True)` 额外编译每层 `forward_prefix_cache`。这段视觉/状态条件计算原来在 eager 模式，是小伙伴定位的主要固定开销。原调用默认仍为 False，避免影响其他离线调用方。 |
| `world_action_model/models/transformer_wa_casual_mot.py` | 5 处缓存 `.detach()` 改成 `.detach().clone()`，在 compiled forward 外复制，防止 CUDA Graph 输出存储复用影响跨去噪步缓存。 |
| `deploy/robot_server.py` | 统一编译入口，设置 pipeline 的 `_torch_compile_mark_step`，推理入口补 `cudagraph_mark_step_begin()`。默认 shell 采用 action-blocks + prefix + reduce-overhead。 |
| 同上 | 专用单线程同时执行预热和线上请求，替代 `asyncio.to_thread` 默认池；服务监听前分别预热 delay=0/RTC，再连续测 median/p95/max。关闭 diffusers 进度条，限制推理 CPU 线程。 |
| 同上 | 公布 `wire_target_sizes`，目标尺寸的图片不再重复缩放；保留原分辨率客户端兼容。自测固定噪声，检查归一化 RTC 前缀严格相等，支持保存动作供 A/B 比较。 |
| `deploy/robot_client.py` | 按服务端公布尺寸，在上行前做训练同款 PIL BILINEAR + center crop。top=320×192，双腕各160×192。RGB 合成图仍为320×384。禁止用 cv2 resize 替代。 |
| 同上 | 增加 shrink/pack/send/recv/unpack、上行字节数、server/request 耗时。原异步 `received ... in ...` 含承诺动作执行等待，改成单独报告 request 和 consume_after_submit，避免误判推理慢。 |
| `deploy/start_server_aligned_python.sh` | 默认开启完整编译，暴露编译范围、预热、线程等开关，支持随时切回 eager 作对照。 |
| `deploy/test_rtc_optimizations.py` | 像素等价、缓存独立性、编译接线、真实 loopback RTC 协议及线程一致性测试；可选微型 BF16 MoT CUDA Graph 测试。 |

关键被替换语句用 `# 原...` 等注释保留，并注明替换原因。完整旧文件另有备份，
不在源码里堆放整份重复类。主项目中原先落后的 server/client 同步到当前部署包的 RTC 版本。

保持：`[L6,L_gripper,R6,R_gripper]`、10 步去噪、48 帧动作时域、BF16、
现有异步提交时机 `pending == RTC_PREFIX`、动作队列、速度/跟踪/跳变阈值。
本次没有迁入小伙伴的自动归位或关闭 tracking error 中止等行为。

## 文件在哪

108 和 113 均提供：

```text
/home/zzd/project/robot1_zzd_test_bundle/home_zzd_test/rtc_python/
/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz
/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz.sha256
/home/zzd/project/giga-world-policy/README_RTC_ENGINEERING_OPTIMIZATIONS.md
```

113 的 `/home/zzd/project/wam_local_emptybox_deploy_package/` 也重新打包了原名称的
`gwp_aligned_emptybox_python_deploy.tar.gz`，包含这次工程优化和 RTC 启动脚本。
旧包先备份；`build_python_package.sh` 也已更新，重建不会漏掉文档和入口脚本。
机器人现有目录是 `rtc_python`，建议用上面的 `rtc_python_optimized_20260917.tar.gz` 更新。

108 旧文件备份：`/home/zzd/project/rtc_engineering_optimization_20260917/before/`。
113 旧主项目/打包文件备份：`/home/zzd/project/rtc_engineering_optimization_20260917/before/`。
该目录同时保存 diff、测试日志及同步校验记录。没有启动或重启真机控制进程。

## 在机械臂电脑更新（只传代码）

先停止正在运行的 client/server，再在机械臂电脑终端执行：

```bash
cd /tmp
scp zzd@10.11.0.108:/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz .
scp zzd@10.11.0.108:/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz.sha256 .
sha256sum -c rtc_python_optimized_20260917.tar.gz.sha256

cp -a /home/geekplus/zzd_test/rtc_python \
  "/home/geekplus/zzd_test/rtc_python.before_opt_$(date +%Y%m%d_%H%M%S)"
tar -xzf /tmp/rtc_python_optimized_20260917.tar.gz -C /home/geekplus/zzd_test/
```

113 上也有完全相同的 tar/sha256，可把以上地址改成 `10.11.0.113`。
包内只有 `rtc_python/` 代码和小型固定任务 token，不含大权重，不修改 cpp_serial 或 SSD 权重。

## 自测与启动

推理环境使用现有 `gwp_infer`。参考开发环境为 torch 2.7.1+cu126、
torchvision 0.22.1、diffusers 0.36.0；先在现有环境测试，不必盲目重装训练依赖。

```bash
cd /home/geekplus/zzd_test/rtc_python
conda activate gwp_infer
python deploy/test_rtc_optimizations.py  # CPU + 本地 WebSocket，无机械臂操作

# 加载真实 RTC30k 权重，预热和检查两条路径，结束后退出；不会连接机械臂。
SELF_TEST=1 WARMUP_RTC_DELAY=8 bash start_rtc30k_python_server.sh

# 正式服务；预热完成才出现 serving on ...，首次可能需要几十秒或更久。
WARMUP_RTC_DELAY=8 bash start_rtc30k_python_server.sh

# 测试 RTC50k 时使用同样参数，改成：
SELF_TEST=1 WARMUP_RTC_DELAY=8 bash start_rtc50k_python_server.sh
WARMUP_RTC_DELAY=8 bash start_rtc50k_python_server.sh
```

需看到：

```text
torch.compile(mode=reduce-overhead, scope=action-blocks, prefix=True): ...forward_prefix_cache[30]...
STEADY delay=0 ... median=... p95=... max=...
STEADY delay=8 ... median=... p95=... max=... prefix_err=0.0
serving on ws://...
```

`warmup ... (not steady)` 不代表稳态。初次编译、路径切换和后台 GPU 负载都会影响延迟。
5 个稳态样本只用于初筛；正式延迟预算要根据实际相机请求的多次测量决定。

客户端另开终端：

```bash
cd /home/geekplus/zzd_test/rtc_python
source /opt/ros/humble/setup.bash
conda activate gwp_client
python -c 'from PIL import Image; print(Image.__version__)'
# 如上条报缺依赖：python -m pip install pillow

bash start_rtc_client_log_only.sh
# 确認当前双臂状态适合运行后，短时测试异步 RTC：
SPEED=50 HZ=10 REPLAN=16 RTC_PREFIX=8 DURATION=30 \
  bash start_rtc_client_async.sh
```

原 `start_rtc_client_closed_loop.sh` 仍是同步推理，要评估 RTC 连续性请用 `async` 脚本。
如果你使用 `RTC_PREFIX=10`，服务端相应设 `WARMUP_RTC_DELAY=10`。

客户端日志：

```text
[client] 上行预处理已开: ...
[timing] request=...ms server=...ms ... payload_mib=0.35 ...
[rtc async] submitted next inference delay=8, pending=8
[rtc async] received chunk ... request=...s, consume_after_submit=...s ...
```

`request` 包括缩图、传输、服务端执行和返回；`recv_ms` 已含服务端时间，不要再次相加。
`consume_after_submit` 还包括承诺队列执行时间，不能拿它当模型耗时。

## 参数与 A/B 对照

| 服务端环境变量 | 默认 | 含义 |
|---|---|---|
| `ENABLE_COMPILE` | 1 | 0 关闭编译，回到 eager；新预热/传输逻辑仍保留 |
| `COMPILE_PREFIX` | 1 | 0 仅编译动作去噪；只有 ENABLE_COMPILE=1 时生效 |
| `COMPILE_SCOPE` | action-blocks | 可选 action-stack；先用小伙伴验证过的 blocks 配置 |
| `COMPILE_MODE` | reduce-overhead | CUDA Graph 优化；default 可作排障对照 |
| `INFER_THREADS` | nproc/4，至少2 | CPU 线程数，给相机和控制循环留核 |
| `WARMUP` | 3 | 每条路径的预热次数 |
| `WARMUP_RTC_DELAY` | 8 | 与本轮 client 的 RTC_PREFIX 相同 |
| `BENCHMARK_REPEAT` | 5 | 每条路径稳态样本数 |
| `SELF_TEST_OUTPUT` | 不保存 | SELF_TEST=1 时可指定 .npz，记录固定种子动作和耗时 |

相同权重、设备、线程数、随机种子下，分别启动独立进程做对照：

```bash
SELF_TEST=1 ENABLE_COMPILE=0 SELF_TEST_OUTPUT=/tmp/rtc_eager.npz \
  bash start_rtc30k_python_server.sh
SELF_TEST=1 ENABLE_COMPILE=1 COMPILE_PREFIX=0 SELF_TEST_OUTPUT=/tmp/rtc_action.npz \
  bash start_rtc30k_python_server.sh
SELF_TEST=1 ENABLE_COMPILE=1 COMPILE_PREFIX=1 SELF_TEST_OUTPUT=/tmp/rtc_full.npz \
  bash start_rtc30k_python_server.sh

python - <<'PY'
import numpy as np
a = np.load('/tmp/rtc_eager.npz')
for name in ['action', 'full']:
    b = np.load(f'/tmp/rtc_{name}.npz')
    for d in [0, int(a['delay'])]:
        e = np.abs(a[f'action_d{d}'] - b[f'action_d{d}'])
        print(name, 'delay', d, 'median_ms', np.median(b[f'latency_ms_d{d}']),
              'joint_max_rad', e[:, [0,1,2,3,4,5,7,8,9,10,11,12]].max(),
              'gripper_max_fraction', e[:, [6,13]].max())
PY
```

compile 可能改变 BF16 算子融合和舍入；不能要求整个后缀逐比特一致，
也不能只凭“数值可复现”认定轨迹效果合格。自测中的归一化硬前缀必须精确保持，
真实轨迹后缀差异和任务效果还需要用你自己的观测验证。合成输入自测不代表成功率评测。

窗口应满足 `RTC_PREFIX / HZ > 实际 request p95 + 余量`，且训练覆盖 delay=0..11。
例如 request p95=0.30s，10Hz×8步有0.8s窗口；30Hz×8步只有0.267s就不够。
先保持10Hz验证，再根据端到端日志调频率。提速不自动解决 tracking error，不要据此放宽保护。

## 验证范围

- 108：CPU 回归（像素等价、缓存独立性、编译接线、预热、真实 loopback 协议/线程）。
- 113：额外使用微型随机 MoT 验证 CUDA Graph，多轮 delay=0/8/10 切换，FP32 与 BF16 均测试。
- 同步后校验核心文件 SHA256；旧文件及旧打包产物先备份。
- 未在机器人电脑运行完整权重测速或下发动作，未宣称你的模型已达到200ms。
  小伙伴记录的187–194ms是其服务端稳态，端到端约274ms；你的结果以以上脚本实测为准。

回退：停止两端进程后恢复 `rtc_python.before_opt_*` 目录；只排查编译问题时先用
`ENABLE_COMPILE=0` 起服务，不必动权重。
