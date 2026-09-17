# 拆分部署：让 RTC 用上 CUDA Graphs

> 目标：RTC（异步执行）下也能开 `torch.compile(mode="reduce-overhead")`，实测比 `default` 快 **2.4 倍**。
> 做法：把推理拆成独立进程，**钉在一个专属线程上**。
> 日期：2026-09-10　|　前置阅读：`DEPLOY_DESIGN.md`（部署契约）、`INFER_LATENCY_DEBUG.md`（五个真机坑）

---

## 1. 问题：RTC 与 CUDA Graphs 看起来互斥

RTC 要求 `control.mode=async`（chunk 之间必须有重叠才有前缀可条件）。而 `deploy_real.py` 的
`run_async` 把推理丢进 `ThreadPoolExecutor`，于是撞上：

```
File ".../torch/_inductor/cudagraph_trees.py", line 325, in get_obj
    assert torch._C._is_key_in_tls(attr_name)
AssertionError
```

原先的结论是「async 与 CUDA Graphs 不兼容」，只能退到 `compile_mode=default`，白扔掉
本路径最主要的加速来源（action-only 推理每步只算 32 个 token，瓶颈在 kernel launch 而非算力，
CUDA Graphs 正是对症的）。

## 2. 实测：真实规则比原结论宽

本机（torch 2.7.1+cu128 / A800）逐个组合测下来：

| 场景 | 结果 |
|---|---|
| 专属线程首次 compile+record，之后一直用它 | ✅ 正常，**0.188 ms/iter** |
| 同条件 `mode=default` | 0.498 ms/iter |
| 同条件 eager | 0.635 ms/iter |
| 一个线程先 record，再换另一个线程 record 新图 | ❌ `AssertionError` |
| 一个线程 record，另一线程 replay **同一张**已录图 | ⚠️ 不报错，但原线程后续 0.355 → **0.797 ms** |

> **CUDA Graph Trees 不要求「主线程」，只要求「从头到尾同一个线程」。**

`local = threading.local()` 与那两行 `_stash_obj_in_tls` 是**模块级语句、只执行一次**
（`cudagraph_trees.py:277-292`），所以谁先 import 谁拥有那份 TLS。

所以 RTC 与 CUDA Graphs **不矛盾** —— 只要把编译、预热、每一次推理全钉在同一个线程上。
`deploy_real.py` 之所以做不到，是因为它的预热池（`deploy_real.py:730`）和推理池
（`:740`）是两个不同的 `ThreadPoolExecutor`，正好落在上表第四行。

## 3. 架构

```
robot PC (openpi env)              GPU 机 / 或同机
┌───────────────────────────┐     ┌──────────────────────────────┐
│ start_robot_arm_service.py │◄───►│ fastwam_client.py            │
│ :9900  ROS2 + Piper SDK    │ TCP │ 控制环 + 安全层 + RTC 调度     │
└───────────────────────────┘     │ **无 torch**                  │
                                  └──────────┬───────────────────┘
                                             │ TCP + msgpack
                                  ┌──────────▼───────────────────┐
                                  │ fastwam_server.py :8900      │
                                  │ InferenceThread(专属线程)     │
                                  │ -> reduce-overhead 可用       │
                                  └──────────────────────────────┘
```

`openpi` 的 `RobotArmService` **保留不动** —— 那一层已真机验证过。刻意不学 GWP 让客户端直接
驱动 Piper SDK + ROS2：重写只为省一跳 localhost 不值得，还会把 GWP 的夹爪比例问题带进来（见 §6）。

### 代码分层

契约（排布/单位/预处理/安全层）**只有一份**，否则三处迟早漂移：

| 文件 | torch? | 谁用 |
|---|---|---|
| `src/fastwam/deploy/layout.py` | ✗ | 三方都用。14 维两套排布、夹爪换算 |
| `src/fastwam/deploy/control.py` | ✗ | client + deploy_real。`SafetyFilter` / `Pacer` / `DelayEstimator` |
| `src/fastwam/deploy/policy.py` | ✓ | server + deploy_real。`ObsPipeline` / `FastWAMPolicy` / **`InferenceThread`** |

`layout` 与 `control` 的 torch-free 是**被测过的**（用 `sys.meta_path` 屏蔽 torch 后仍能 import）。

`deploy_real.py` 从 1954 行降到 1512 行，删掉的 450 行全部改为 import 共享实现；
保留它是因为 sync 路径与三个无硬件子命令已经真机验证过。

## 4. `InferenceThread`：唯一的新机制

```python
inf = InferenceThread(lambda: FastWAMPolicy(cfg, dcfg))   # ★ 连模型构建都在该线程
inf.warmup(joint, grip, images, rtc=True)                 # 两张图都热
res = inf.submit(joint, grip, images).result()             # 每次推理都回到它
```

三条硬性要求，缺一条 CUDA Graphs 就静默失效或直接崩：

1. **模型也在该线程构建** —— 让首次 CUDA 上下文与之后的图录制同线程。
2. **`assert_owner()` 守卫** —— 换线程调用立刻报错，而不是等到静默劣化。
3. **两张图都要预热**：`delay=0` 走 1-D timestep、`delay>0` 走 2-D，是两张独立的图，
   各要编译一次（约 41 s）。`delay` 的**数值**不影响图，所以任意一个 `d>0` 即可覆盖全部取值。

> ⚠️ GWP 的 `deploy/robot_server.py` 在**主线程** warmup（第 680 行）、再交给 `_infer_pool`
> serve（第 550 行），正好落在 §2 表格第五行。照抄那个顺序会带上风险：一旦控制环里出现
> 需要新录的图（RTC 的两张图正是如此），就是硬 `AssertionError`。

## 5. RTC 前缀走「绝对物理量」上线

客户端**不持有** `dataset_stats.json`，也就不可能用错一份。server 收到后换算回归一化空间。

前提是 `use_stepwise_action_norm=False`（本数据集正是如此），此时 norm↔phys 是与 proprio
无关的固定逐元素仿射。实测往返：

```
phys -> norm -> phys   max|Δ| = 1.19e-07
norm -> phys -> norm   max|Δ| = 4.77e-07
```

server 启动时用 `action_affine()` 做一次往返自检把这个前提**钉死**，而不是假设
（`use_stepwise_action_norm=True` 会直接报错而非静默走偏）。

同理，限位（`action_limits`）与 `rtc_trained` 也由 server 在 `ping` 里下发，client 拿来喂
`SafetyFilter` 并硬拒错误组合 —— 只有一个真值来源。

## 6. 照抄 GWP 时改掉的四处

| # | GWP 的做法 | 本实现 | 为什么 |
|---|---|---|---|
| 1 | `--gripper-scale 0.105` | **0.07** | 硬件上限 `GRIPPER_RAW_MAX=70000` = 0.07 m。0.105 会让 0.667 以上的输出全部塌缩成"全开"，顶段分辨率丢失（GWP 自己标着"待重新验证"） |
| 2 | 客户端预缩图（2.64 MB→0.35 MB） | **原样转发 JPEG** | openpi 给的本来就是 JPEG q95（三路约 0.19 MB，实测），那个问题不存在；且 `selfcheck` 的逐元素一致性只对 server 侧一份代码成立，客户端插一次 resize 就破坏它（GWP 自己也警告 cv2 与 PIL BILINEAR 最大差 96）。顺带把 JPEG 解码也移出控制环 |
| 3 | 起始位姿硬编码成字面量数组 | server 下发 `action_limits` | 换数据集不会静默漂移 |
| 4 | 主线程 warmup + pool serve | `InferenceThread` 全程同线程 | 见 §4 |

## 7. 验证结果（mock 后端 + 假 service，三进程）

GPU 全被占满时也能跑通整条链路：`--mock` 后端形状/单位/排布/协议全真，只有动作是编的。

```
async + RTC   200 控制步 / 40 次规划 / 6.8 s   有效 29.4 Hz（目标 30）
              节拍超时 0 次 (0.0%)   chunk 用尽 0 次   安全层零触发
              实测 elapsed 中位 4 / p90 5 步     前缀 overrun 2 次
sync          90 控制步 / 9 次规划    有效 18.9 Hz（推理时冻结，符合预期）
```

**接缝质量**（相邻指令的最大关节步长，从落盘记录算）：

| | 接缝处均值 | chunk 内均值 | seam_ratio |
|---|---|---|---|
| RTC 前缀 | 0.00050 | 0.00050 | **1.002** |
| naive 异步 | 0.01005 | 0.00050 | 20.17 |

`seam_ratio ≈ 1.0` 表示 chunk 边界与 chunk 内部的运动**不可区分** —— 时序对齐与前缀切片
（`chunk[idx : idx+d]`，不是 `[0:d]`）经端到端验证正确。

另外验证过：`InferenceThread` 在真实 `reduce-overhead` 下的线程亲和性（cudagraph 容器
落在 worker 线程上、off-thread 守卫会报错）；共享 `DelayEstimator` 与 `deploy_real` 原版
`_predict_delay` 在 8 个用例上**逐个一致**。

## 8. 顺带修掉的两个既有 bug

都在 `deploy_real.py rtc-check` 的测试脚手架里，与本次重构无关（用备份确认过原版同样存在）：

1. **`AttributeError: '_elapsed_hist'`** —— 脚手架用 `EpisodeRunner.__new__` 绕过 `__init__`，
   而 `_predict_delay` 读 `_elapsed_hist`（加那个字段时漏了同步）。`rtc-check` 的 [2] 组
   一直是跑不完的。
2. **两条断言的期望值过时** —— 写在「infer_s 路径 +1 步兜底」之前（`INFER_LATENCY_DEBUG` 坑 5
   的修法）。代码是对的，断言是旧的。顺带补了一条覆盖**首选路径**（有 elapsed 样本时用其
   p90 且不再 +1）的断言。

## 9. 上手顺序

```bash
source env.sh

# 0) 无 GPU 也能验证整条链路（三进程，全在本机）
python scripts/fastwam_server.py --mock --port 8900 &
python scripts/deploy_fake_service.py --port 19912 &      # 首次先 --extract
python scripts/fastwam_client.py --server tcp://127.0.0.1:8900 \
    --endpoint tcp://127.0.0.1:19912 --mode async --rtc \
    --confirm-safety --i-am-watching --max-steps 150 --record-dir /tmp/rec

# 1) 预处理与训练逐元素一致（仍然是 deploy_real 的 selfcheck，共享同一份 ObsPipeline）
python scripts/deploy_real.py selfcheck --config configs/deploy/agilex_real_rtc.yaml

# 2) 实测 d —— 在专属线程上跑，所以 reduce-overhead 的数字是**真能拿到**的
python scripts/fastwam_server.py --config configs/deploy/agilex_real_rtc.yaml \
    --benchmark --compile-mode both --steps-list 5,10

# 3) 起服务（RTC + CUDA Graphs）
bash scripts/start_server.sh

# 4) 真机：dry 核对单位方向 -> sync 小步试 -> async RTC 闭环
bash scripts/start_client.sh                       # MODE=dry，不动手臂
#   改 scripts/start_client.sh 里的 MODE=sync / async
```

**上真机前必看的三个数**（`ping` 的 info 里，client 会打出来）：

- `compile_mode` 不是 `null` —— 否则 d 远超训练覆盖的 11，RTC 等于没开
- `rtc_trained: true` —— 拿基线 checkpoint 配 RTC 会掉精度（离线实测 MSE 近翻倍），client 会硬拒
- `torch_threads_infer` 有限制 —— 与 client 同机时不限会抢满核，实测节拍超时 25% → 0%

## 10. 遗留

- **本机 GPU 全占满（最多剩 2.8 GB，模型需 13.4 GB），真模型的端到端数字没能实测。**
  `reduce-overhead` 在专属线程上可用是用真实 `torch.compile` 验证的（stand-in 模块），
  但「10 步 + reduce-overhead 的真实 d」必须在真机上跑 §9 第 2 步确认。
- 模型精度问题独立存在，与本文无关：RTC `step_008000` 的执行窗口关节误差约 7.4~7.6°，
  非 RTC 基线约 4~5°。RTC 修好了接缝，代价是单步定位精度差约 1.8 倍。
  怀疑主因是「从已过拟合的 step_032000 再走 8000 步」而非 RTC 本身 ——
  **非 RTC 的 8000 步对照跑完才能定论**，见 `RTC_TRAIN_DESIGN.md` §5、§6。
