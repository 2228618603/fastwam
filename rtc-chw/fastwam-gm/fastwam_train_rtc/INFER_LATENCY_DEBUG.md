# 真机推理延迟排查记录 —— 从 378 ms 到 178 ms，以及沿路踩的五个坑

> 场景：把 training-time RTC（`RTC_TRAIN_DESIGN.md`）接进真机异步执行（`DEPLOY_DESIGN.md`）时，
> 推理延迟 `d` 超出训练覆盖范围，导致 RTC 的接缝保护完全失效。
> 环境：机器人 PC / RTX 4090 / 20 核 / torch 2.7.1+cu128 / py 3.10
> 排查日期：2026-09-03　|　最终状态：跑通，`d=6~7`、async 停顿 0、安全层零触发

---

## 0. 为什么值得记下来

这次排查的五个问题**没有一个是模型问题**，全部是「异步 + torch.compile + 实时控制环」三者交互产生的工程问题。
它们的共同特征是：

> **离线 benchmark 全部正常，只在真机上炸；而且报错位置离根因很远。**

如果以后再遇到「benchmark 很快但真机很慢」，按第 6 节的清单逐条排一遍。

---

## 1. 起点：问题长什么样

RTC 要求 `control.mode: async`（chunk 之间必须有重叠才有前缀可条件）。切过去之后：

```
[async] chunk 21 step 278  infer 472 ms  接入 chunk 第 15 步  d(预估)=11
⚠️ chunk 首步跳变 0.271 rad (15.5 度),已由 delta 钳位吸收
```

- 实测 `elapsed = 12~15` 步，而 `rtc.max_delay = 11`（训练时 `d ~ uniform[0,12)`，模型只见过 `d ≤ 11`）
- 于是 `d` 被钳死在 11，**切入点 12~15 全部落在前缀 `[0,11)` 之外**
- 前缀 overrun 100% 发生 → 接缝退化成 naive 拼接 → 首步跳变 8.6~15.5 度

**关键恒等式**（这一点当时理解错了，值得单独强调）：

```
切入 chunk 的第几步  ==  推理延迟（控制步）
```

它与「什么时候发起推理」**无关**。等一会儿再发起不会让你从第 0 步切入，只会让上一个 chunk
往更深处走（误差更大）：

| wait | 切入 idx | 执行到 idx | 误差范围（error(k)≈1.37+0.232(k-1) 度） |
|---|---|---|---|
| 0 | 8 | 15 | 2.99 ~ 4.62 |
| 5 | 8 | 20 | 2.99 ~ 5.78 |
| 8 | 8 | 23 | 2.99 ~ 6.47 |

把这个想法推到极限（算完再动）就是 `sync` 模式。所以这不是调度技巧问题，**唯一出路是把延迟压下去**。

---

## 2. 五个坑，按发现顺序

### 坑 1：`mode="reduce-overhead"` 在后台线程上必然崩

**现象**

```
File ".../torch/_inductor/cudagraph_trees.py", line 325, in get_obj
    assert torch._C._is_key_in_tls(attr_name)
AssertionError
```

**机制**（`torch/_inductor/cudagraph_trees.py`）

```python
local = threading.local()                                    # line 277
local.tree_manager_containers = {}                           # line 280  ← 只在 import 的线程上执行
torch._C._stash_obj_in_tls("tree_manager_containers", ...)   # line 291  ← 只 stash 到那个线程

def get_obj(local, attr_name):                               # line 321
    if hasattr(local, attr_name):        # 新线程 -> False（threading.local 隔离）
        return getattr(local, attr_name)
    else:
        assert torch._C._is_key_in_tls(attr_name)            # line 325 ← 炸在这
```

那几行是**模块级语句，只执行一次**。`reduce-overhead` / `max-autotune` 会启用 CUDA Graph Trees，
所以**任何非 import 线程调用编译后的函数都会炸**。

`run_async` 恰好是 `pool.submit(self.policy.infer_chunk, ...)` —— 推理在工作线程。
而 `sync` 模式 / `dryrun` / `offline_eval` / `benchmark` 全在主线程，所以这个坑**在切到 async 之前从未暴露**。

**修法**：把写死的 mode 变成可配置，async 下用 `default`（只做算子融合，不用 CUDA Graphs）。

- `fastwam.py:853` `_compiled(name, fn, mode)` —— 按 `(name, mode)` 缓存，mode 变了重新编译而非返回旧版本
- `fastwam.py:864` `_mode_uses_cudagraphs(mode)` —— 用它门控 `cudagraph_mark_step_begin()` 与 KV cache 的 `.clone()`
  （那次 clone 约 44 MB，只有 CUDA Graphs 才需要）
- `deploy_real.py:1016` **启动守卫**：`async + compile + cudagraph mode` → 直接 `SystemExit`

> 守卫很重要：这个组合原本是**跑到第一次后台推理才崩**，那时机器人已经在动了。

**代价**：`default` 拿不到 CUDA Graphs 的收益。实测 `reduce-overhead` 67 ms vs `default` 159 ms —— 
CUDA Graphs 值 2.4 倍。但 `default` 已经够用（见第 3 节），所以没有去做「把推理挪回主线程」的重构。

### 坑 2：编译发生在实时控制环里面

**现象**

```
⚠️ [async] chunk 用尽,保持上一条指令(第 100 次)
⚠️ [async] 推理耗时 902 步 >= action_horizon 32 步
推理: 均值 15017 ms  p90 35800 ms          ← benchmark 里是 158.8 ms
!! episode 中止: chunk 首步与实测状态的关节差 0.622 rad (35.7 度) 超过 abort 阈值
```

**机制**：`torch.compile` 是**惰性**的，第一次调用才编译（实测 17~84 s）。这 17 秒里机器人拿不到新
chunk，只能「保持上一条指令」876 次；等新 chunk 终于到达，它是基于 30 秒前的观测算的 → 35.7 度跳变
→ 安全层正确中止。

**为什么 benchmark 没暴露**：`cmd_benchmark` 每组丢弃前 2 次迭代（`前 2 次含 CUDA warmup / 首次分配,丢掉`），
编译成本正好被吃在被丢弃的那两次里。**真机上没有人替它预热。**

**修法**：`deploy_real.py:1121` `_warmup_inference()` —— 在控制环**之外**先跑几次，把编译成本付掉，
并且跑完 `stat_infer_s.clear()`（否则那 17 秒会污染 `_predict_delay` 的分位数窗口，把 `d` 钳到 11）。

### 坑 3：预热被算进 episode 计时

**现象**：`60 控制步 / 1 次规划 / 19.9 s (有效 3.0 Hz)` —— 但 19.9 s 里有 17.0 s 是编译。
真实是 `60 步 / 2.9 s ≈ 20.7 Hz`。

**修法**：`deploy_real.py:1173` `warmup_before_episodes()`，在 `cmd_run` 的 `t0 = perf_counter()`
**之前**调用，且只做一次（5 个 episode 不重复付）。配 `policy._warmed_up` 做幂等。

### 坑 4：只预热了一半的计算图 ★ 最隐蔽的一个

**现象**

```
>>> 预热推理 2 次...  1/2: 17006 ms <- 含编译   2/2: 179 ms   ← 预热明明成功了
[async 诊断] 发起 1 次 / 落地 0 次              ← 但控制环里一次都没落地
async 停顿(chunk 用尽): 28 次
```

**机制**：RTC 的 per-token timestep 有**两条形状不同的路径**，是**两张独立的计算图**：

```
delay == 0  ->  timestep 形状 [1]        ->  图 A（无条件生成，冷启动第一个 chunk）
delay >  0  ->  timestep 形状 [1, H]     ->  图 B（前缀条件）
```

预热不带前缀，所以只热了图 A。控制环里第一次发起就带 `delay=5` → 触发图 B 的编译（约 41 s）
→ 2.5 s 的 episode 里当然落不了地。

**修法**：预热**两条路径都要跑**（`_warmup_inference` 里的 `plans` 列表）。

> 好消息：`delay` 的**数值**不影响图（timestep 形状恒为 `[1, H]`，`d` 只改数值），
> 所以热任意一个 `d>0` 就覆盖了全部延迟取值 —— 两次预热搞定，不是 12 次。
> 这正是设计 per-token timestep 时刻意保证的形状稳定性，在这里派上了用场。

**可推广的教训**：**枚举所有输入形状分支，每个都要预热。** 形状不同 = 不同的图 = 各自一次编译。

### 坑 5：用代理量预测 `d`，产生系统性低估

**现象**

```
elapsed 序列: 7, 6, 6, 6, 7, 6, 6, 6
d(预估)     : 6                        → 两次 elapsed=7 就 overrun
[rtc] 前缀 overrun 2 次
```

**机制**：`_predict_delay` 原本用 `policy.stat_infer_s`，那只是 **`infer_s`（模型耗时）**：

```
infer_s p90 ≈ 197 ms  ->  ceil(197/33.3) = 6
wall    p90 ≈ 233 ms  ->  ceil(233/33.3) = 7     ← 真实 elapsed
差 13~36 ms = 预处理 + 工作线程等 GIL
```

调高 `latency_percentile` 只是打补丁，治不了这个偏差。

**修法**：`deploy_real.py:977` `_elapsed_hist` —— 直接用**观测到的 `elapsed`**（本来就是控制步单位，
自带预处理/GIL/控制环速率的全部影响）作首选依据；样本不足时退回 `infer_s` 估计并 **+1 步**兜底。

**可推广的教训**：**预测你真正关心的量，别用代理量。** 这里真正关心的是「从发起到能用隔了几个控制步」，
那就直接统计它。

---

## 3. 实测数字

### 3.1 延迟（RTX 4090 / 20 核 / bf16）

| `num_inference_steps` | compile | mode | 推理 | 预处理 | 合计 | **d** | async 可用 |
|---|---|---|---|---|---|---|---|
| 5 | ✗ | – | 377.9 ms | 5.6 | 383.5 | **11.5** | 撞上限 |
| 5 | ✓ | `default` | 158.8~169 | 5.5~10.5 | 164~180 | **4.9~5.4** | ✅ |
| 5 | ✓ | `reduce-overhead` | 67.1 ms | 5.7 | 72.8 | **2.2** | ❌ 仅主线程 |
| 4 | ✓ | `default` | 134.5 | 20.2* | 154.7 | 4.6 | ✅ |
| 10 | ✓ | `default` | 288.5 | 8.8 | 297.4 | 8.9 | ✅ |

\* 该值虚高，见第 4.2 节。

**结论：`compile_mode=default` 是决定性的（2.4 倍），线程调优只值几个百分点。**

### 3.2 线程扫描（`--sweep-threads`，20 核）

最优 `torch_threads_pre=8 / torch_threads_infer=4`（默认值 16/0 是 80 核 A800 上调的）。

空载 benchmark 显示 **0% 差别**，但真机上有效：

| | pre=16 / infer=0(=20) | pre=8 / infer=4 |
|---|---|---|
| 节拍超时 | 25.0% | **0.0%** |
| JPEG 解码累计 | 1.0 s | **0.2 s** |
| 有效频率 | 23.7 Hz | **27.5 Hz** |

**原因**：`infer=20` 让推理线程占满全部 20 核，主线程的控制环（TCP + 3 张 JPEG 解码）只能抢。
`infer=4` 给控制环留了 16 核。**空载 benchmark 测不到这个收益 —— 它没有并发的控制环。**

### 3.3 修复全过程

| 指标 | 初始 | 坑2修完 | 坑3修完 | 坑4修完 | 坑5修完 |
|---|---|---|---|---|---|
| 推理均值 | 15017 ms | 185 ms | 178 ms | 184 ms | 184 ms |
| 规划次数 / 60 步 | 3 | 1 | 1 | **9** | 9 |
| 落地次数 | – | 0 | 0 | **8** | 8 |
| async 停顿 | 876 | 28 | 28 | **0** | 0 |
| 节拍超时 | 38.6% | 30.0% | 6.7% | 0.0% | 0.0% |
| 有效频率 | 3.0 Hz | 3.0 Hz | 23.9 Hz | 27.5 Hz | 27.5 Hz |
| 前缀 overrun | 2 | 0* | 0* | 2 | **预期 0** |
| 安全层触发 | abort 35.7° | 0 | 0 | 0 | 0 |

\* 那两次的 0 是「根本没落地所以没得比」，不是真的好。

---

## 4. 三个测量假象（不是 bug，但会把人带偏）

### 4.1 `TCP 均 99.7 ms`

`client.reset()` 服务端 sleep 约 5 s，被算进 TCP 统计。样本少时它主导均值：

```
本次: (6.2 s − 5.0 s) / 61 步 = 19.7 ms/步
上次: (24.8 s − 5.0 s) / 917 步 = 21.6 ms/步     ← 一致
```

TCP 本身约 20 ms，没问题。**小样本上的均值会被一次性成本污染。**

### 4.2 预处理耗时随去噪步数变化（20.2 → 10.5 → 8.8 ms）

预处理是 3 张图 resize + 拼图，**与去噪步数无关**，不可能变化。真因是 `cmd_benchmark` 按
`steps_list` 顺序跑，**第一组承担了全部一次性初始化**，而每组只丢弃 2 次盖不住。

**修法**：进 `steps_list` 循环前先做全局预热 3 次。

### 4.3 `benchmark` 里 `reduce-overhead` 不会崩

`benchmark` 跑在**主线程**，所以它测 cudagraph 类 mode 时给出一个**async 下拿不到的乐观数字**。
`--compile-mode` 的帮助文本里写明了这点；它的意义是「CUDA Graphs 值多少钱」= 做线程重构能换回多少。

---

## 5. 加的诊断工具

| 工具 | 位置 | 用途 |
|---|---|---|
| `benchmark --compile {true,false,both}` | `cmd_benchmark` | 隔离 compile 的贡献 |
| `benchmark --compile-mode {default,reduce-overhead,both}` | 同上 | 隔离 CUDA Graphs 的贡献 |
| `benchmark --sweep-threads` | `deploy_real.py:1435` | 分两阶段扫线程（O(P+I) 而非 O(P×I)，见下） |
| 全局预热 3 次 | `cmd_benchmark` | 消除第一组虚高 |
| `[async 诊断]` 发起/落地台账 | `run_async` | `wall` vs `infer` 分开报，定位「模型慢」还是「线程被饿死」 |
| 启动守卫 | `deploy_real.py:1016` | async × cudagraph mode 直接拒绝 |
| `rtc-check` 第 [4] 组 | `deploy_real.py:1621` | 在真实工作线程上验证两种 mode 的可用性 |
| `inspect.signature` 探测 | `FastWAMPolicy.__init__` | 部署脚本与 `src/` 版本不同步时明确报错而非 `TypeError` |

**线程扫描为什么能分两阶段**：`infer_chunk` 在预处理前后各调一次 `torch.set_num_threads`，
所以 `pre_s` 只受 `_nt_pre` 影响、`infer_s` 只受 `_nt_infer` 影响。20 核上 7 个候选从 49 组降到 14 组。
扫完会用最优组合**实测复核一次**，而不是把两段最小值相加。

---

## 6. 清单：下次遇到「benchmark 很快、真机很慢」

按这个顺序排：

1. **benchmark 丢弃了多少次迭代？** 被丢弃的那几次里藏着真机要付的一次性成本（编译、首次分配、page cache）。
2. **有没有 `torch.compile`？如果有，它在哪个线程上第一次被调用？**
   - cudagraph 类 mode（`reduce-overhead` / `max-autotune`）**只能在 import 它的线程上用**
   - 惰性编译会落在实时环里 → 必须在环外预热
3. **枚举所有输入形状分支，每个都预热了吗？** 形状不同 = 不同的图 = 各自一次编译。
4. **计时的起点在哪？** 预热/初始化被算进业务计时会让「有效 Hz」严重失真。
5. **小样本的均值里有没有一次性大值？**（本例：`reset()` 的 5 s 混进 62 个 TCP 样本）
6. **线程配置是从别的机器抄来的吗？** 核数不同，最优点会整体平移；而且空载 benchmark 测不出并发场景的收益。
7. **预测用的是代理量还是真值？** 能直接观测到目标量（本例的 `elapsed`）就别用代理量（`infer_s`）。

---

## 7. 最终配置（RTX 4090 / 20 核）

```yaml
# configs/deploy/agilex_real_rtc.yaml
inference:
  num_inference_steps: 5
  compile_action_infer: true
  compile_mode: default          # ⚠️ async 下必须；reduce-overhead 会被启动守卫拒绝
  torch_threads_pre: 8           # 默认 16 是 80 核 A800 调的，换机器必须重扫
  torch_threads_infer: 4         # 给控制环留 16 核
rtc:
  delay_mode: measured           # d 由实测 elapsed 的 p90 自动定，无需手填
  latency_percentile: 0.9
control:
  mode: async
```

建议同时开 Inductor 磁盘缓存，避免每次启动付 125 s 编译：

```bash
export TORCHINDUCTOR_CACHE_DIR=<持久盘>/.inductor_cache
```

> 注意：`torch_threads_infer: 4` 也限制了编译的并行度，编译时间从 16.8 s 涨到 83.9+41.3=125 s。
> 磁盘缓存能把后续启动降到几秒。

---

## 8. 尚未解决（与本文无关，记在此以免混淆）

工程链路已通，但**模型精度**问题独立存在：

| | 执行窗口关节误差 @d≈6 |
|---|---|
| 非 RTC 基线 `final_A/step_032000` | 约 4~5° |
| RTC `step_020000` | 约 7.4~7.6° |

RTC 修好了接缝（`seam_ratio` 6.41 → 1.24，真机上安全层零触发也印证了），代价是单步定位精度差约 1.8 倍。
这个差别在真机任务上要不要紧，只能靠成功率判断。详见 `RTC_TRAIN_DESIGN.md` 第 5、6 节。
