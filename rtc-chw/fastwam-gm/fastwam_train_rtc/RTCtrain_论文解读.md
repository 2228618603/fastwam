# Training-Time RTC 论文深度解读

> Training-Time Action Conditioning for Efficient Real-Time Chunking
> Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine — Physical Intelligence
> arXiv:2512.05964v2 [cs.RO], 2025-12-09

## 目录

- [一句话总结](#一句话总结)
- [1. 坐标：这篇论文在 RTC 这条线的什么位置](#1-坐标这篇论文在-rtc-这条线的什么位置)
- [2. 前置：inference-time RTC 及其代价](#2-前置inference-time-rtc-及其代价)
- [3. 它要解决的三个问题](#3-它要解决的三个问题)
- [4. 方法：Training-Time Action Conditioning](#4-方法training-time-action-conditioning)
- [5. Algorithm 1 逐行拆解](#5-algorithm-1-逐行拆解)
- [6. 为什么有效：机制分析](#6-为什么有效机制分析)
- [7. 实验](#7-实验)
- [8. 逐项对比：inference-time vs training-time RTC](#8-逐项对比inference-time-vs-training-time-rtc)
- [9. 局限与批判性阅读](#9-局限与批判性阅读)
- [10. 相关工作的位置关系](#10-相关工作的位置关系)
- [11. 落地清单](#11-落地清单)
- [12. 论文中的一处笔误](#12-论文中的一处笔误)
- [附录：符号表](#附录符号表)

---

## 一句话总结

RTC 为了保证 action chunk 之间的连续性，在**推理时**做 inpainting，而这个 inpainting 用的 pseudoinverse guidance **每个去噪步都要跑一次反向传播**——为了实时性引入的机制自己变成了实时性的负担。这篇论文把这件事**搬到训练时**：训练时随机采样一个模拟的推理延迟 $d$，把 chunk 的前 $d$ 步当作"已知的干净动作前缀"喂给模型，只在后缀上算 loss。于是模型直接学到条件分布 $p(A_{t+d:H} \mid o_t, A_{t:t+d})$，推理时**零额外开销**，前缀直接 clamp 就行。

代价是丢掉了 soft masking 的灵活性，并且需要事先决定训练时的延迟分布。收益是：真机 π0.6 上端到端延迟从 **135ms 降到 108ms**（$d\approx7 \to d\approx5$），成功率和任务时长与 inference-time RTC 打平；仿真中在 $d \ge 2$ 时**反超**，且差距随 $d$ 增大而拉开。

> 关键的实现体量：**不改模型架构、不改机器人 runtime、只需几行代码**。而且可以在一个**没有用前缀条件预训练过**的 base model 上通过微调加进去。这是它最有说服力的地方——不是一个需要重做 pipeline 的方案。

---

## 1. 坐标：这篇论文在 RTC 这条线的什么位置

```
π0 / π0.5 / π0.6          ← action chunking flow-matching VLA 本体
        │
        ├── RTC (arXiv:2506.07339, Black/Galliker/Levine)
        │      异步执行框架 + 推理时 inpainting（pseudoinverse guidance + soft masking）
        │
        └── 本文 (arXiv:2512.05964)
               沿用 RTC 的异步执行框架，替换其中的 inpainting 模块
               → 训练时动作前缀条件（training-time action conditioning）
```

**必须分清的两层**（读这篇论文最容易混淆的地方）：

| 层 | 内容 | 本文是否改动 |
|---|---|---|
| **异步执行框架** | 提前发起推理、queue 管理、chunk 拼接、时间对齐 | ❌ 完全沿用 RTC，**runtime 一行不改** |
| **动作生成模块** | 给定动作前缀 + 延迟 $d$，生成后缀 | ✅ 本文替换这一层 |

论文明确说：新方法"adheres to the same interface as the action generation component of inference-time RTC (see [5], Algorithm 1) and thus acts as a seamless drop-in replacement"。所以这是一次**模块级替换**，不是新框架。

---

## 2. 前置：inference-time RTC 及其代价

### 2.1 问题设定（与 RTC 完全一致）

- 策略 $p(A_t \mid o_t)$，其中 $A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$
- $H$：**prediction horizon**（预测长度）
- $s \le H$：**execution horizon**（每个 chunk 实际执行的步数）
- $d$：**inference delay**，以控制器时间步为单位

时序约束（论文 Fig. 1）：在 $t$ 时刻发起推理，结果在 $t+d$ 才可用，所以新 chunk 的前 $d$ 个动作**无法被执行**——这段时间由上一个 chunk 顶上。上一个 chunk 从 $t-s$ 开始，覆盖到 $t-s+H-1$，因此必须满足

$$t + d \le t - s + H \quad\Longrightarrow\quad \boxed{d \le H - s}$$

这 $d$ 个来自上一个 chunk、与当前 chunk 重叠的动作，就是 **action prefix（动作前缀）**。

```
        t-s              t         t+d              t-s+H          t+H
         │───── 上一个 chunk ──────────────────────────│
                         │────────── 当前 chunk ──────────────────────│
                         │◄── 前缀 ──►│◄─ 重叠剩余 ──►│
                         │   (红,d步)  │   (黄,H-s-d步) │   自由(绿)
                         └── 推理进行中，执行上一chunk ──┘
```

### 2.2 flow matching 约定

$$A_t^\tau = \tau A_t + (1-\tau)\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \tag{1}$$

$$\mathcal{L}(\theta) = \mathbb{E}\left\|v_\theta(A_t^\tau, o_t, \tau) - (\epsilon - A_t)\right\|^2 \tag{2}$$

即 $\tau = 0$ 是纯噪声、$\tau = 1$ 是干净动作，推理时从 $\tau=0$ 积分到 $\tau=1$。

> ⚠️ 式 (2) 的符号与论文 Algorithm 1 的代码不一致，见 [第 12 节](#12-论文中的一处笔误)。

### 2.3 inference-time RTC 干了什么，贵在哪

inference-time RTC 做两件事：

1. **硬约束前缀**（红色，$d$ 步）：新 chunk 的前 $d$ 步必须等于上一 chunk 已承诺的值。
2. **软约束重叠剩余**（黄色）：把**所有** $H-s$ 个重叠动作都用上，超出前缀的部分用**指数衰减权重**——这就是 RTC 论文里的 **soft masking**，用来改善 chunk 间连续性。

实现手段是 **pseudoinverse guidance**（[18] Pokle et al. *Training-free linear image inverses via flows*；[21] Song et al. *Pseudoinverse-guided diffusion models*）。它的好处是极其灵活——任意线性观测算子都能作为约束，所以才能支持任意权重的软掩码。它的代价是：

> **每一个去噪步都需要计算一次 vector-Jacobian product（即一次反向传播）。**

这就是全部问题所在。假设 5 步去噪：
- 普通采样：5 次 forward
- pseudoinverse guidance：5 次 forward + 5 次 VJP ≈ **2~3 倍的计算量**

论文的措辞很直白：这个开销"somewhat defeats the purpose of a real-time execution framework"——为了实时性设计的机制，自己成了延迟的来源。

而且延迟增加会**自我放大**：延迟 $\uparrow$ → $d \uparrow$ → 被冻结的前缀更长 → 机器人被更旧的决策绑得更久 → 反应性 $\downarrow$。同时论文还发现，inpainting 本身"is fundamentally limited in its ability to handle high inference delays"——$d$ 大时它的效果本来就在退化。**两个方向同时恶化**。

---

## 3. 它要解决的三个问题

| # | 问题 | 表现 |
|---|---|---|
| **P1** | pseudoinverse guidance 的 VJP 开销 | 每步一次反向传播 → 端到端延迟 135ms vs 无开销的 108ms |
| **P2** | 延迟增大时 inpainting 质量退化 | 仿真中 $d \ge 2$ 后被 training-time 方法反超，且差距持续拉大 |
| **P3** | P1 与 P2 互相放大 | 开销 → 延迟 → $d$ 变大 → 落进 inpainting 的退化区 → 效果更差 |

注意 P3：这三个问题不是并列的，是一个**正反馈环**。本文一刀切断 P1，环就断了。

---

## 4. 方法：Training-Time Action Conditioning

### 4.1 核心思想

> **既然推理时一定会有 $d$ 步延迟，那就在训练时把这个延迟模拟出来，让模型直接学会"在给定动作前缀的条件下补完后缀"。**

形式化：不再学 $p(A_t \mid o_t)$，而是学

$$p(A_{t+d:H} \mid o_t,\ A_{t:t+d})$$

其中 $A_{t:t+d}$ 是动作前缀（Fig.1 红色），$A_{t+d:H}$ 是动作后缀（Fig.1 黄+绿），**两者都取自同一条 ground-truth chunk**。

这是把"条件生成"从**测试时的近似推断**（雅可比线性化）变成**训练时的分布摊销**（amortized inference）。模型权重里直接编码了这个条件分布，推理时不需要任何额外计算。

### 4.2 关键 trick：用 per-token 的 flow matching timestep 表达"已知"

这是整个方法最漂亮的一笔，也是"只要几行代码"的原因。

观察式 (1)：当 $\tau = 1$ 时，$A_t^1 = A_t$，也就是**干净的真实动作**。所以：

> 把某个 token 的 flow matching timestep 设为 $\tau = 1$、并把它的值设为干净的真实动作，就等价于告诉模型"这一步已经完全确定了"。

于是不需要任何新的 mask embedding、不需要额外的条件分支、不需要新参数——**复用 flow matching timestep 作为 per-token 的"已知/未知"指示器**。

更妙的是：$d$ 这个量**不需要单独作为输入喂进去**。有多少个 token 的 $\tau = 1$，$d$ 就是多少。论文 Fig. 2 的图注说得很清楚——"The flow matching timestep differs between tokens, which indicates the inference delay to the model."

```
                        ┌─────────────── action expert / DiT ───────────────┐
                        │                                                    │
     τ 值:            1.0   1.0    τ     τ     τ     τ     τ                 │
                       │     │     │     │     │     │     │                 │
     token:          [a_t] [a_t+1][ ~  ][ ~  ][ ~  ][ ~  ][ ~  ]             │
                     └─ 动作前缀 ─┘└──────── 动作后缀（加噪）────────┘         │
                       真实值        噪声插值                                 │
                                                                             │
                        │           DiT block × N                            │
                        └────────────────────┬───────────────────────────────┘
                                             │
     loss:            [无 loss][无 loss][ ✓ ][ ✓ ][ ✓ ][ ✓ ][ ✓ ]
                                             └── 只在后缀上算 flow matching loss
```

### 4.3 三处改动（论文原文列举）

**改动 1：让 flow matching timestep 可以逐 token 不同。**

对 DiT 类架构（[16] Peebles & Xie），timestep 是通过 **adaLN-zero** 注入的——即由 $\tau$ 生成 scale / shift / gate 三组调制参数。要支持 per-token 的 $\tau$，只需让这三组参数**按 token 各不相同**。

> 论文特别强调："This does not change the number of learnable parameters." 因为 adaLN 的 MLP 是共享的，只是输入从 $(B,)$ 变成 $(B, H)$，输出从 $(B, C)$ 变成 $(B, H, C)$，broadcast 维度变了而已。

**改动 2：前缀用干净的真实动作，对应的 $\tau$ 设为 1.0。后缀完全不变。**

**改动 3：mask 掉 loss，只在后缀对应的输出上计算。**

为什么必须 mask loss？因为前缀位置的输入已经是干净的真实动作（$\tau=1$），此处的"去噪"任务是平凡的（目标速度恰好对应零剩余噪声），继续算 loss 只会浪费梯度、并且给模型注入一个它在推理时永远不需要执行的任务。

**改动 4（隐含，但很重要）：$d$ 在训练时随机采样。**

论文：*"since we do not know the exact inference delay ahead of time (and inference delays in the real world may vary), we sample $d$ randomly during training."*

这一点被论文轻描淡写了，但**工程价值极大**：
- inference-time RTC 需要你选一个保守的 $d$（比如延迟的 p95），选大了牺牲反应性，选小了接缝跳变。
- training-time RTC 的 $d$ 是**运行时输入**，而模型在训练时见过整个 $d$ 的分布。所以延迟抖动可以被**优雅地按实际值处理**，不需要保守估计。

### 4.4 推理接口

```
输入: 动作前缀 A_{t:t+d}  +  延迟 d
输出: 动作后缀 A_{t+d:H}
```

与 inference-time RTC 的 action generation 组件**接口完全一致**，所以 runtime 不用改。

---

## 5. Algorithm 1 逐行拆解

论文用 JAX 给出了完整实现。下面是带注解的版本（注释中标 ★ 的是相对标准 flow matching 代码的改动）。

### 5.1 训练：`compute_loss`

```python
import jax
import jax.numpy as jnp

def compute_loss(rng, model, observation, action_chunk, max_delay):
    b, ah, ad = action_chunk.shape        # (batch, action_horizon=H, action_dim)
    noise_rng, time_rng, delay_rng = jax.random.split(rng)

    time  = jax.random.uniform(time_rng, (b,))          # 每个样本一个 τ ~ U[0,1)
    noise = jax.random.normal(noise_rng, (b, ah, ad))   # ε ~ N(0, I)

    # ★ 1. 采样模拟的推理延迟。真机实验里用的是 Unif[0, max_delay)
    delay = jax.random.randint(delay_rng, (b,), 0, max_delay)

    # ★ 2. 前缀位置的 τ 强制设为 1.0
    #    time 的形状从 (b,) 变成 (b, ah) —— 这就是 per-token timestep
    prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]   # (b, ah) bool
    time = jnp.where(prefix_mask, 1.0, time[:, None])

    # 构造带噪输入。注意前缀位置 τ=1 → x_t = 1·A + 0·ε = A（干净真实动作）
    #                后缀位置 τ<1 → 正常的噪声插值
    x_t = time[:, :, None] * action_chunk + (1 - time[:, :, None]) * noise

    pred_v_t = model(observation, x_t, time)             # 模型无需任何结构性改动
    loss = (pred_v_t - (action_chunk - noise)) ** 2      # 目标速度 = A - ε

    # ★ 3. loss 只在后缀上求平均
    postfix_mask = jnp.logical_not(prefix_mask)[:, :, None]
    loss = jnp.sum(loss * postfix_mask) / (jnp.sum(postfix_mask) + 1e-8)
    return loss
```

**几个容易踩的点：**

- **`x_t` 的构造是自洽的，不需要特判。** 因为 $\tau=1$ 代入 $\tau A + (1-\tau)\epsilon$ 自动得到 $A$。这就是为什么"改动 2"看起来只是改了 `time` 一个变量——干净前缀是**自动涌现**的，不是手写进去的。
- **`delay` 是 per-sample 的，不是 per-batch。** 同一个 batch 里不同样本有不同的 $d$，梯度信号更均匀。
- **`delay=0` 是合法采样值**，此时退化为标准 flow matching（全部是后缀）。这保证了模型仍然会做无条件生成——冷启动第一个 chunk 需要它。
- **归一化用 `sum / sum(mask)` 而不是 `mean`。** 必须如此，否则 $d$ 大的样本会因为有效 token 少而被稀释。

### 5.2 推理：`sample_actions`

```python
def sample_actions(rng, model, observation, action_prefix, delay, num_steps):
    # action_prefix 被 pad 到 (b, ah, ad)，但只有前 delay 个是有效的
    b, ah, ad = action_prefix.shape
    x_t  = jax.random.normal(rng, (b, ah, ad))     # τ=0：纯噪声
    time = 0.0
    dt   = 1 / num_steps

    prefix_mask = jnp.arange(ah)[None, :] < delay

    for _ in range(num_steps):
        # ★ 每步把前缀位置硬性 clamp 回真实前缀值
        x_t = jnp.where(prefix_mask[:, :, None], action_prefix, x_t)
        # ★ 前缀位置的 τ 同样设为 1.0，与训练时一致
        time_masked = jnp.where(prefix_mask, 1.0, time)

        v_t = model(observation, x_t, time_masked)
        x_t = x_t + dt * v_t                        # 只有 forward，没有 VJP ★
        time = time + dt

    return x_t
```

**关键观察：**

1. **循环里只有 forward，没有反向传播。** 这就是全部的性能收益来源。对比 pseudoinverse guidance 每步一次 VJP。
2. **`clamp` 放在模型调用之前。** 保证模型每次看到的前缀都是干净真实值，与训练时的输入分布严格一致。
3. **返回值 `x_t` 的前缀位置是"脏"的。** 最后一次 `x_t = x_t + dt * v_t` 也更新了前缀位置，且循环结束后没有再 clamp 一次。这**不是 bug**——因为 runtime 只取后缀（索引 $\ge d$）使用。但**重新实现时如果误用了返回值的前缀部分，就会引入错误**。稳妥的做法是在 `return` 前再 clamp 一次，或者干脆 `return x_t[:, delay:]`。
4. **`delay` 是运行时传入的普通参数。** 不需要重新编译、不需要切换 checkpoint，直接把测到的真实延迟传进去。

---

## 6. 为什么有效：机制分析

论文只给了一句解释，值得展开。原文：

> *"This is likely because, as the size of the prefix grows, the inpainting algorithm has to 'work harder' to produce a consistent postfix. In these cases, the training-time algorithm is more robust than the pure inference-time algorithm, which relies on a linearization obtained from the Jacobian of the model."*

### 6.1 两种做法在数学上做的是不同的事

| | inference-time RTC | training-time RTC |
|---|---|---|
| 数学对象 | 用**局部线性化**近似条件得分 $\nabla \log p(A_{\text{post}} \mid A_{\text{pre}})$ | 直接**参数化并拟合**条件分布 $p_\theta(A_{\text{post}} \mid o, A_{\text{pre}})$ |
| 何时求解 | 测试时逐步求解一个逆问题 | 训练时一次性摊销（amortized） |
| 误差来源 | 雅可比线性化的近似误差 | 模型容量 + 训练数据覆盖 |
| 近似何时失效 | **约束越强越失效** | 与约束强度无关 |

### 6.2 为什么 d 越大 inference-time 越吃亏

pseudoinverse guidance 本质上是在每个去噪步用模型在当前点 $x^\tau$ 的雅可比，把"我希望前缀等于某个值"这个约束**线性地**投影回噪声空间的修正方向。这是一个**一阶近似**：

- $d$ 小：被约束的维度少，需要的修正量小，一阶近似在小邻域内准确 → 效果好。
- $d$ 大：前缀强烈约束了后缀的可行集（"我已经承诺往左走了 8 步，后面不可能突然是向右的轨迹"），需要的修正量大且高度非线性 → 一阶近似崩溃。

而 training-time RTC 见过 $d$ 大的样本，它**知道**给定一个长前缀该怎么续，这是学出来的而非算出来的，不存在近似误差随约束强度增长的问题。

Fig. 3 的曲线形状（$d\ge2$ 反超、差距随 $d$ 拉大）与这个机制解释完全吻合。

### 6.3 为什么 d = 0, 1 时略微吃亏

论文的解释：*"training-time RTC does not always receive training supervision for every action — i.e., slightly less training compute is spent learning to generate the first and second actions."*

因为 loss 被 mask 掉了前缀部分，前几个动作位置获得的梯度信号总量少于普通训练。$d=0$ 时（全部是后缀）本该没有差别，但这个 checkpoint 在训练时平均有一部分样本的前几步没被监督，所以对"从零生成前几个动作"这件事练得略少。

差距是"very marginally worse"，属于可接受的代价。论文也指出可以通过给每个 $d$ 单独训 checkpoint 来消除，只是更费算力。

### 6.4 双重收益（论文没明说但很重要）

```
去掉 VJP  →  单步更便宜  →  端到端延迟 135ms → 108ms
                                   ↓
                          d 从 ≈7 降到 ≈5
                                   ↓
              ┌────────────────────┴────────────────────┐
              ↓                                          ↓
    冻结前缀更短，机器人被旧决策                 而且按 Fig.3，d 更小
    绑得更松 → 反应性更好                        本来也在更容易的工作点
```

而 Fig. 3 又说它在 $d$ 大时更强。也就是说：**它在自己的工作点上更轻松，同时在困难工作点上更强壮**——两个方向都顺。

---

## 7. 实验

### 7.1 仿真：dynamic Kinetix

| 项 | 配置 |
|---|---|
| Benchmark | dynamic Kinetix（[15] Matthews et al.），与 RTC 论文同一套 |
| 架构 | 4 层 MLP-Mixer（[25] Tolstikhin et al.） |
| Prediction horizon | $H = 8$ |
| Execution horizon | $s = \max(d, 1)$ |
| 测试延迟 | $d \in \{0,1,2,3,4\}$（$H=8$ 下的最大可行值，受 $d \le H-s$ 约束） |
| 统计量 | 每个数据点 2048 次 rollout，95% Wilson score 置信区间 |
| 数据 | 混合专家策略生成 |

**训练算力严格对齐**（这个对照设计做得很干净）：

- Naive async 与 inference-time RTC：**共用同一个 checkpoint**，正常训练 32 epoch，**不带**前缀条件。
- Training-time RTC：从第 **24** epoch 的 checkpoint 恢复，**再用前缀条件微调 8 epoch**。总计仍是 32 epoch。

> 这个设计同时验证了两件事：① 算力公平；② **可以在没有前缀条件预训练的模型上后期加装**——这正是真机实验能在 π0.6 base model 上做的依据。

**延迟采样分布**：从 $\{0,1,2,3,4\}$ 按**指数衰减权重**采样，理由是"higher delays need less training supervision"（$d$ 大 → 后缀短 → 任务更简单 → 需要的监督更少）。

**结果（Fig. 3）**：

| $d$ | 结论 |
|---|---|
| 0 | training-time 极轻微落后 |
| 1 | training-time 极轻微落后 |
| ≥ 2 | **training-time 胜出，且差距随 $d$ 增大显著拉开** |

### 7.2 真机：π0.6 上的 box building 与 espresso making

| 项 | 配置 |
|---|---|
| Base model | π0.6（[24] Physical Intelligence π0.6 model card） |
| 任务 | 纸箱组装、意式咖啡制作（磨豆 → 压粉 → 萃取 → 倒出），取自 π*0.6 [1] |
| 微调 | 两个 checkpoint 各 8,000 步，batch size 512 |
| 训练延迟分布 | $d \sim \text{Unif}[0, 10]$ → 支持 50Hz 下最大 200ms 延迟 |
| 推理硬件 | **远程** H100 服务器 |
| 去噪步数 | 5 |
| 对照 | 同步基线与 inference-time RTC **共用同一 checkpoint** |

**实测端到端延迟**（核心数字）：

| 方法 | 端到端延迟 | 对应 $d$ |
|---|---|---|
| Training-time RTC | **108 ms** | $\approx 5$ |
| Inference-time RTC | **135 ms** | $\approx 7$ |

→ 省下 **27 ms ≈ 20%**，对应 50Hz 下 **少冻结 2 个控制步**。

**结果（Fig. 5，报告 success rate + duration）：**

- Training-time RTC 与 inference-time RTC 在**成功率和任务时长上都打平**（parity），但**没有计算开销**。
- 两种 RTC 都**显著快于同步推理**基线；同步基线有"visible pauses in between chunks"（chunk 之间可见的停顿）。
- 误差棒：成功率用 68% Wilson score 区间，时长用 ±1 SEM。

**这个实验最重要的一点不是数字，是可行性证明**：training-time RTC 能通过**微调一个未使用前缀条件预训练的 base model** 加装上去。意味着已有的 VLA 资产不需要从头重训。

---

## 8. 逐项对比：inference-time vs training-time RTC

| 维度 | Inference-time RTC | Training-time RTC |
|---|---|---|
| **条件化机制** | pseudoinverse guidance（测试时逆问题求解） | 训练时动作前缀条件（摊销推断） |
| **每步额外开销** | 一次 VJP / 反向传播 | **零** |
| **实测延迟（π0.6, 5 步, H100）** | 135 ms | **108 ms** |
| **利用的重叠动作** | **全部 $H-s$ 个**（红 + 黄），指数衰减权重 | **仅前 $d$ 个**（红） |
| **soft masking** | ✅ 支持 | ❌ 不支持 |
| **高延迟表现（仿真）** | $d\ge2$ 起退化 | **更强，差距随 $d$ 拉大** |
| **低延迟表现（$d=0,1$）** | 略优 | 极轻微落后 |
| **训练成本** | 无（可用普通 checkpoint） | 需带前缀条件的训练/微调 |
| **架构改动** | 无 | per-token flow matching timestep（**不增加参数**） |
| **Runtime 改动** | — | **无**（接口一致） |
| **延迟抖动处理** | 需保守选一个 $d$ | $d$ 是运行时输入，训练时已覆盖整个分布 |
| **超参负担** | soft mask 衰减率 | **训练时延迟分布的选择** |
| **代码量** | guidance 实现 + 每步 VJP | **几行** |

**一句话取舍**：如果你的延迟很低（$d \le 1$）且不想动训练，用 inference-time RTC；如果延迟在几十到几百毫秒（真实的大 VLA 场景），用 training-time RTC。

---

## 9. 局限与批判性阅读

### 9.1 论文自己承认的两条

**L1：灵活性更差。** 只支持对应推理延迟的"硬"前缀，无法像 inference-time RTC 那样软性地纳入前缀之外的重叠动作。

这一条的实际影响需要注意：上一轮我们讨论过，**soft 过渡区是 chunk 接缝处平滑性的来源**。硬前缀在 $\tau=1$ 位置保证了 $C^0$ 连续（第一个被执行的新动作 = 上一 chunk 对该时刻的预测），但**前缀之后的过渡是模型学出来的，不是显式约束的**。论文的真机结果说这在实践中够用（成功率和时长都打平），但这是一个**经验结论而非结构保证**。如果你的任务对接缝平滑性极端敏感，这一点值得单独验证（看关节加速度曲线）。

**L2：需要谨慎选择训练时的延迟分布。** 必须基于预期的推理延迟来定。

真机用的是 $\text{Unif}[0,10]$（支持到 200ms @ 50Hz），实际工作点 $d\approx5$。所以他们训练的范围比实际需要的**宽一倍**。这是明智的（覆盖抖动），但也说明：**换硬件、换网络拓扑、换去噪步数，都可能需要重新考虑这个分布**。相比之下 inference-time RTC 是即插即用的。

### 9.2 论文没讨论但真实存在的问题

**训练/测试的前缀分布不匹配（exposure bias）。**

这是我认为最值得追的一点：

- **训练时**，前缀 $A_{t:t+d}$ 取自**数据集里的 ground-truth chunk**（专家动作）。
- **推理时**，前缀来自**上一个 chunk，是模型自己生成的动作**。

这是经典的 exposure bias / DAgger 式失配。如果模型自身分布与专家分布有偏差（几乎必然），那推理时喂进去的前缀就是**轻度 OOD 的**。而且这个偏差会沿着 chunk 链条**递归累积**——chunk $k+1$ 的前缀来自 chunk $k$，chunk $k$ 的前缀来自 chunk $k-1$……

有意思的是，inference-time RTC 在这一点上**没有这个问题**，因为它从不训练在前缀条件下，guidance 是纯测试时的。所以这是本文引入的一个新失配面。

论文完全没提。可能的缓解方向：训练时对前缀加噪 / 用 rollout 出来的前缀做 scheduled sampling / 混合专家前缀与自生成前缀。这是一个明确的后续工作点。

**Intro 与 Results 的口径不一致。**

- Introduction：*"we show **improved performance** over inference-time RTC on two highly complex tasks: box building and espresso making."*
- §V-B / Fig. 5：*"training-time RTC maintains both performance and speed **parity** with inference-time RTC"*
- Abstract：*"maintains both task performance and speed **parity** ... while being computationally cheaper"*

Abstract 和 Results 都说 **parity**，只有 Intro 说 **improved**。以 Results 和 Fig. 5 为准：**真机上是打平 + 更省算力**，不是性能提升。仿真里才有性能反超。读的时候别被 Intro 带偏。

**其他没覆盖的**：

- 只在 flow matching / DiT 类架构上验证。对自回归离散 token 的 VLA（π0-FAST 那种 DCT 频域 tokenize）如何适配，没有讨论。
- 只有 5 去噪步这一个设置。去噪步数越多，VJP 的绝对开销越大，training-time RTC 的优势应该越明显——但没有这条 ablation。
- 没有报告训练时的额外开销（per-token adaLN 会让 adaLN 的调制参数计算量从 $O(B)$ 变成 $O(BH)$，虽然通常可忽略）。
- 两个真机任务，样本量不大（Wilson 68% 区间说明每个条件的 trial 数有限），"parity"的结论本身有统计不确定性。

---

## 10. 相关工作的位置关系

论文把自己放在一张相当清楚的地图上：

```
                        VLA 实时性问题
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
  ① 让模型更快            ② 分层解耦              ③ 异步执行 + 解决不连续
      │                       │                       │
  MiniVLA [2]          Gemini Robotics [23]      SmolVLA [20]  ← 异步但不解决不连续
  SmolVLA [20]         GR00T N1 [3]                            → chunk 间 OOD "jerks"
                       (System 1 / System 2)
                                                 RTC [5]       ← 推理时 inpainting
   ↑ 论文明确说这两类                                  │
     "orthogonal to ours"                              ├── A2C2 [19]  轻量校正头
     且各有代价（改架构、改训练流程）                    ├── VLASH [22] 只条件于**单个**未来动作
                                                       └── 本文      条件于**完整前缀**
```

**几个精确的定位陈述：**

- **SmolVLA [20]**：异步执行算法与 RTC 类似，但**不解决 chunk 间不连续问题**，导致 chunk 之间出现 OOD 的"jerks"。
- **A2C2 [19]**（Sendai et al., *Leave no observation behind*）：**并发工作**，通过加一个**轻量校正头**解决不连续。
- **VLASH [22]**（Tang et al., Song Han 组，*Real-time VLAs via future-state-aware asynchronous inference*）：**并发工作**，通过条件于**单个**未来动作解决不连续。论文明确对比：*"In contrast to VLASH, we condition on a full prefix of future actions."*
- **分层 VLA / 小模型**：正交路线，可以叠加使用，但各自有代价（改网络架构、改训练配方）。

> 三个并发工作（本文、A2C2、VLASH）在 2025 年 9–12 月集中出现，都在解决"异步执行的 chunk 间不连续"。这说明这个问题已经是 VLA 落地的公认瓶颈，值得持续跟踪。

---

## 11. 落地清单

如果要在自己的 flow-matching VLA 上加装 training-time RTC：

### 训练侧

- [ ] **确认架构支持 per-token timestep。** DiT + adaLN-zero：让 scale/shift/gate 的 broadcast 从 `(B, C)` 变 `(B, H, C)`。其他架构（如 MLP-Mixer、U-Net）需要确认 timestep 注入点能否逐 token 区分。
- [ ] **测出真实推理延迟分布**，再定训练时的 $d$ 分布。建议覆盖到 p99 的 1.5～2 倍（论文：实际 $d\approx5$，训练覆盖到 10）。
- [ ] **$d$ 的采样分布**：均匀（论文真机）或指数衰减（论文仿真，理由是大 $d$ 需要更少监督）。**必须包含 $d=0$**，否则冷启动第一个 chunk 无法生成。
- [ ] **$d$ 按样本采样，不是按 batch。**
- [ ] **loss 归一化用 `sum/sum(mask)`**，不要用 `mean`。
- [ ] **校验约束 $d_{\max} \le H - s$。**
- [ ] **可以从现成 base model 微调**，不必从头训（论文真机：8,000 步 @ batch 512；仿真：32 epoch 里最后 8 epoch）。

### 推理侧

- [ ] **runtime 不用改**——如果你已经跑着 RTC 的异步框架，只替换动作生成函数。
- [ ] **把测到的真实延迟作为 `delay` 传入**，不要用固定保守值。这是相对 inference-time RTC 的主要增益之一。
- [ ] **前缀 clamp 放在模型调用之前**，每个去噪步都做。
- [ ] **只使用返回值的后缀部分**（索引 $\ge d$）。稳妥起见在 `return` 前再 clamp 一次或直接切片返回。
- [ ] **前缀必须是绝对动作表示**（关节位置目标 / 绝对末端位姿）。delta 动作要先积分成绝对量，否则"同一时刻的动作"在两个 chunk 间不可比。

### 验证侧

- [ ] **对比 VJP 前后的端到端延迟**，确认省下的时间符合预期（论文：135→108ms）。
- [ ] **看 chunk 接缝处的关节速度/加速度曲线**，这是判断连续性是否真的保住了最直接的信号（因为丢了 soft masking，这一条必须实测）。
- [ ] **扫 $d$ 做 ablation**，确认在自己的工作点上没有落进退化区。
- [ ] **监控 queue 欠载次数**，确认异步框架的时序假设成立。

---

## 12. 论文中的一处笔误

正文式 (2) 与 Algorithm 1 的代码**符号不一致**：

| 来源 | 速度目标 |
|---|---|
| 式 (2) | $\epsilon - A_t$ |
| Algorithm 1 | `action_chunk - noise` 即 $A_t - \epsilon$ |

**代码是对的**，正文式 (2) 的符号写反了。推导：式 (1) 给出 $A^\tau = \tau A + (1-\tau)\epsilon$，对 $\tau$ 求导得 $\frac{dA^\tau}{d\tau} = A - \epsilon$。而 `sample_actions` 从 `time=0.0` 起、以 `x_t = x_t + dt * v_t` **正向**积分到 $\tau=1$，要落在数据上，速度必须是 $A - \epsilon$。

> 这大概是 π0 系列惯用约定（$A^\tau = \tau\epsilon + (1-\tau)A$，从 $\tau=1$ 反向积分到 $\tau=0$，目标 $\epsilon - A$）与本文改用的正向约定混写导致的。**实现时以 Algorithm 1 为准。**

---

## 附录：符号表

| 符号 | 含义 | 论文中的取值 |
|---|---|---|
| $H$ | prediction horizon，chunk 预测长度 | 仿真 8 |
| $s$ | execution horizon，每 chunk 实际执行步数 | 仿真 $\max(d,1)$ |
| $d$ | inference delay，以控制步为单位 | 仿真 0–4；真机训练 Unif[0,10]，实测 ≈5 |
| $\tau$ | flow matching timestep，0=噪声 1=数据 | 前缀强制为 1.0 |
| $A_t$ | 从 $t$ 开始的动作 chunk，$[a_t,\ldots,a_{t+H-1}]$ | — |
| $A_{t:t+d}$ | action prefix，动作前缀（Fig.1 红） | — |
| $A_{t+d:H}$ | action postfix，动作后缀（Fig.1 黄+绿） | — |
| $\epsilon$ | 高斯噪声 | $\mathcal{N}(0,I)$ |
| $v_\theta$ | 速度场网络 | — |
| 约束 | $d \le H - s$ | — |

---

## 参考

- **本文**：Black, Ren, Equi, Levine. *Training-Time Action Conditioning for Efficient Real-Time Chunking.* arXiv:2512.05964v2, 2025.
- **[5] RTC**：Black, Galliker, Levine. *Real-Time Execution of Action Chunking Flow Policies.* arXiv:2506.07339, 2025.
- **[24] π0.6**：Physical Intelligence. *π0.6 model card*, 2025.
- **[1] π*0.6**：Amin et al. *π*0.6: a VLA that learns from experience.* arXiv:2511.14759, 2025.
- **[18]** Pokle et al. *Training-free linear image inverses via flows.* arXiv:2310.04432, 2023.
- **[21]** Song et al. *Pseudoinverse-guided diffusion models for inverse problems.* ICLR 2023.
- **[19] A2C2**：Sendai et al. *Leave no observation behind: Real-time correction for VLA action chunks.* arXiv:2509.23224, 2025.
- **[22] VLASH**：Tang et al. *VLASH: Real-time VLAs via future-state-aware asynchronous inference.* arXiv:2512.01031, 2025.
- **[15] Kinetix**：Matthews et al. arXiv:2410.23208, 2024.
- **[16] DiT**：Peebles & Xie. ICCV 2023.
- **[13] Flow Matching**：Lipman et al. arXiv:2210.02747, 2022.
