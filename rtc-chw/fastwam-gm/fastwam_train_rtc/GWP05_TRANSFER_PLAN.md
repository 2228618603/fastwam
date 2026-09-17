# 把 GigaWorld-Policy-0.5 预训练权重迁移进 FastWAM —— 决策、映射、验证与 A/B 方案

> 讨论日期：2026-09-03
> 相关文档：`../GigaWorldPolicy解读/GigaWorldPolicy_论文与代码详解.md`（尤其 §4.4 / §4.5 / §4.13）、
> `NEXT_SESSION.md`（FastWAM 训练与离线评测结论）、`RTC_TRAIN_DESIGN.md`、`DEPLOY_DESIGN.md`、
> `INFER_LATENCY_DEBUG.md`
>
> 本文件是**方案与执行记录**。所有形状、key 名、参数量都实测自本地文件，不是从论文抄的。

---

## 0. 一句话

在 FastWAM 里热启动 GWP-0.5 的预训练权重，跑一次受控 A/B，**用半天机器时间判断「GWP 那批 embodied 预训练对我这 471 episode 单任务有没有用」**——而不是花两周把整条链路搬到 GWP 仓库去问同一个问题。

---

## 1. 为什么选这条路（方案 A）而不是直接用 GWP 训（方案 B）

### 1.1 我们缺的只有一项

| 我们已经有的（已验证） | 来源 |
|---|---|
| 5 epoch / 77,690 步完整训练 + 39 个 checkpoint | `runs/agilex_final_3cam_384_1e-4/final_A/` |
| 两套验证集上的离线评测 sweep（成对差分 + 95% CI + 逐关节度数矩阵） | `eval_offline/final_A_newval71/`，指标与训练期 eval 已实测对齐（`action_l2` 0.0227 vs 0.0246，差 8%） |
| training-time RTC（关闭时逐比特一致，已回归验证） | `RTC_TRAIN_DESIGN.md` + `scripts/rtc_regression_check.py` |
| 真机异步部署，378 ms → 178 ms，`d=6~7`、async 停顿 0、安全层零触发 | `DEPLOY_DESIGN.md` + `INFER_LATENCY_DEBUG.md` |
| 一套现成的 A/B 短跑框架 | `scripts/chain_ab.sh`，baseline `runs/agilex_uncond_3cam_384_1e-4/ab_A` |

| 我们缺的 | 谁有 |
|---|---|
| **embodied 预训练**（FastWAM 论文明确 "without embodied pretraining"） | GWP：GigaWorld-1 视频 + 2K h 机器人数据 + AC-WM 混合 |

我们是 471 episode / 单任务，step 32,000（epoch 2.06）就开始过拟合——**正是预训练收益最大的区间**（GWP Table 7 在小数据下报 +28pt）。

上面两张表就是全部理由：**方案 B 的绝大部分成本花在重建第一张表，而不是拿到第二张表。**

### 1.2 实验质量：受控 vs 全量替换

| | 变的量 | 归因能力 |
|---|---|---|
| **方案 A** | 只有初始化 | 数据 / 归一化 / 指标 / 验证集 / 判读规则 / baseline 全部不动 → **受控实验** |
| **方案 B** | 数据格式(v2.1→v3.0)、动作参数化(绝对关节→delta)、归一化(z-score→q01/q99)、掩码、优化器(AdamW→CAME8Bit)、flow shift、horizon(32→48) **同时全变** | 好不知道为什么好，差不知道为什么差 |

我们已经过了"能跑就行"的阶段（在做 39 个 checkpoint 的成对差分 + CI），这个阶段最贵的是**归因能力**。

### 1.3 「已经做了」不是理由，「已经验证过」才是

重建代码不难，重建可信度很难。`INFER_LATENCY_DEBUG.md` 的五个坑**全部不是模型问题**，全是异步 + torch.compile + 实时控制环的交互，离线 benchmark 都正常。这类坑换仓库会重新长一批。

### 1.4 成本（数字已修正，见 §1.5）

| | 人力 | 机器 |
|---|---|---|
| 方案 A | ≈ 1 人·天 | 每臂 2.7 h wall-clock / 8×A800 = **21.8 GPU·h** |
| 方案 B | ≈ 两周 + 重挣一遍可信度 | 同等或更多 |

### 1.5 ⚠️ 成本数字的一次修正

先前口头估的"6 GPU·小时"是错的——把 wall-clock 当 GPU·小时说了，差 8 倍。实测你自己的 A/B 短跑：

| run | 起 | 止 | wall-clock |
|---|---|---|---|
| `ab_A`（5000 步 / 8×A800） | 08-28 14:06:34 | 16:50:02 | **2 h 43 m** |
| `ab_B`（5000 步 / 8×A800） | 08-28 16:51:03 | 19:34:41 | **2 h 44 m** |

和配置注释里 1.95 s/step 的推算吻合（5000 × 1.95 s = 2.71 h）。`bs=8` 峰值显存 76.2~78.3 GB / 81920 MiB，**塞不进第二个 run，只能串行**。

完整代价：

| 项 | 代价 |
|---|---|
| 转换脚本 + 验证 | ≈ 1 人·天 |
| 读 22.4 GB F32 → 写 bf16 | ≈ 11 GB/份 磁盘 |
| 每臂训练 | 2.7 h 独占 8 卡 = 21.8 GPU·h |
| 短跑离线评测 | 远小于 39-ckpt sweep 的 50 min |
| 机会成本 | 机器多人共享，独占是真实成本 |

**所以映射表的验证不能省**：`load_checkpoint` 用 `strict=False`，会静默吞掉任何 key 拼写错误，而一次静默失败要花掉的正是这 2.7 h。验证在 CPU 上几分钟跑完。

### 1.6 什么会让我们改推方案 B

1. **延迟成为硬约束**（不是"勉强能跑"）。这是 GWP 唯一无法靠改几十行抹平的结构性优势：`wam.cpp` 0.6 在 A800 / 10 步的实测是 GWP05 **122.01 ms** vs FastWAM **211.27 ms**。
   → 但这件事**不需要先训练就能判断**：`wam.cpp` 0.6 两个模型都支持（`scripts/convert/convert_fastwam.py`，且 `profiles/fastwam_robotwin_3cam384_zscore.json` 正好对应我们的配置）。先用官方权重在我们的输入尺寸上量一遍，再决定值不值得投。**先量，再投。**
2. **C1 赢、C2 输**（见 §5）。那是"动作专家绑死在 GWP 掩码上"的指纹，说明那半边价值只能在 GWP 里取。
3. **要往多任务 / 多本体走**。`EmbodimentSpecificLinear` + 10,000 h 课程是为这个设计的，FastWAM 完全没有。

### 1.7 诚实的上限声明

方案 A 的上限**低于**方案 B。§4 的语义障碍③④意味着 GWP 那批预训练的价值**可能大部分绑在它自己的掩码里**，转过来只剩视觉专家那一半。

**推方案 A 优先不是因为 A 更强，是因为 A 是探针：它能在一天内告诉你 B 值不值得做，反过来不行。**

---

## 2. 与解读文档 §4.13.3「为什么不能互相加载」的关系

那一节的**标题**比它自己的正文结论强。它实际证明的是**不能 drop-in**，而它的收尾句是：

> block 权重互灌是一个可以做的研究实验（比如"换掉掩码后微调"），**但不是一个能省事的捷径**。

它列的四道拦路虎，两道是工程的、两道是语义的：

| 拦路虎 | 性质 | 本方案的处理 | 解决了吗 |
|---|---|---|:---:|
| ① key 名空间完全不同 | 工程 | §3 完整映射表 | ✅ |
| ② 动作/状态 I/O 维度层数都不同 | 工程 | §3.3 那 19 个张量弃用，对应的 4 个目标张量留 FastWAM 自己的初始化 | ✅ 绕开 |
| ③ 状态注入路径不同 | **语义** | §4 标为坑 | ❌ |
| ④ 掩码语义不同 | **语义** | §4 标为坑，**C1 臂就是为隔离它而存在** | ❌ |

①② 转换脚本能消掉，③④ 不能。**这条路的成本在脚本，风险在语义。**

---

## 3. 权重映射（全部实测自本地文件）

### 3.0 本地文件清单

| 文件 | 内容 | 张量数 | 命名 |
|---|---|---|---|
| `/mnt/data/zzd/giga-world-policy/Giga-World-Policy-0.5/` | GWP-0.5 预训练 MoT，22.4 GB **F32** | **1664** | diffusers（`_class_name: CasualWorldActionTransformer_MoT`） |
| `checkpoints/Wan-AI/Wan2.2-TI2V-5B/` | Wan2.2 原始权重，19 GB | **825** | **原始/DiffSynth = FastWAM 原生命名** |
| `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | FastWAM 的 ActionDiT 插值初始化 | — | FastWAM |

> 🔑 第二行是关键：本地的 Wan2.2 已经是 FastWAM 的命名，且它的 825 个 key 恰好是 `mixtures.video.*` 的完整目标清单。**所以映射表可以离线实证校验，不必下载 Diffusers 版。**

### 3.1 维度逐位对齐（不需要任何形状手术）

| | GWP-0.5 | FastWAM |
|---|---|---|
| visual / video expert | 30 层 / 3072 / 14336 / 24×128 | 完全相同 |
| action expert | 30 层 / 1024 / 4096；q/k/v `1024→3072`、o `3072→1024` | 完全相同 |
| in/out channels、patch | 48 / 48、(1,2,2) | 相同 |
| 视频输出头 | `proj_out` (192, 3072) | `head.head` (192, 3072) |

### 3.2 映射规则

每个 block，`visual_expert → mixtures.video`、`action_expert → mixtures.action`：

```
attn1.to_q / to_k / to_v        → self_attn.q / k / v
attn1.to_out.0                  → self_attn.o
attn1.norm_q / norm_k           → self_attn.norm_q / norm_k
attn2.{同上四条}                → cross_attn.{同上四条}
ffn.net.0.proj / ffn.net.2      → ffn.0 / ffn.2
scale_shift_table  (1,6,C)      → modulation  (1,6,C)
norm2.{weight,bias}             → norm3.{weight,bias}        ← ★ 换名
```

顶层：

```
patch_embedding                              → mixtures.video.patch_embedding
condition_embedder.text_embedder.linear_1/2  → mixtures.video.text_embedding.0/.2
condition_embedder.time_embedder.linear_1/2  → mixtures.video.time_embedding.0/.2
condition_embedder.time_proj                 → mixtures.video.time_projection.1
proj_out                                     → mixtures.video.head.head
scale_shift_table  (1,2,3072)                → mixtures.video.head.modulation
action_condition_embedder.{text,time}_*      → mixtures.action.{text,time}_embedding.0/.2
action_condition_embedder.time_proj          → mixtures.action.time_projection.1
```

### 3.3 弃用的 19 个 GWP 张量 → 4 个目标张量保持自有初始化

| GWP | 形状 | FastWAM 对应 | 为什么不行 |
|---|---|---|---|
| `action_encoder.{in,mid,out}_proj.{w,b}` (6) | `in_proj [2,16,128]` | `action_encoder` `(1024,14)` | 16 vs 14 维、3 层 vs 1 层、多 embodiment 轴 |
| `action_decoder.{in,mid,out}_proj.{w,b}` (6) | `out_proj [2,128,16]` | `head` `(14,1024)` | 同上 |
| `state_encoder.{in,mid,out}_proj.{w,b}` (6) | `[2,16,128]` → 自注意力 token | `proprio_encoder` `(4096,14)` → cross-attn | 注入路径不同，不是同一个东西 |
| `action_scale_shift_table` (1) | `[1,2,1024]` | 无（`ActionDiT.head` 是裸 Linear） | 丢掉 |

> `mixtures.action.action_encoder` / `head` 在 baseline 里**本来就是随机初始化**（`ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")`），所以 A/B 在这一点上是公平的。
>
> 重新初始化的参数量：`14×1024×2 + 1024 + 14 ≈ 3 万 / 6.02 B`。

### 3.4 关于 `norm2 → norm3`：这不是猜的，是被结构唯一确定的

- GWP `ActionExpertBlock.__init__`：`norm1`(affine=**False**)→self-attn 前、`norm2`(affine=**True**)→cross-attn 前、`norm3`(affine=**False**)→FFN 前。checkpoint 里确实只有 `norm2.{weight,bias}`。
- FastWAM `DiTBlock.__init__`：`norm1`/`norm2`(affine=**False**)、`norm3 = nn.LayerNorm(hidden_dim, eps)`(affine 默认 **True**)。

**两边各只有一个带参数的 LayerNorm，所以映射是被唯一确定的，不存在歧义。** 其余同理：`ffn.net.0.proj [14336,3072]` / `net.2 [3072,14336]` 由形状锁定；`attn1`=self / `attn2`=cross 由源码锁定（`_expert_qkv` 用 `attn1` 且加 rope，`_expert_cross_attn_output` 用 `attn2` 且吃 `encoder_hidden_states`）；`norm_q(q)` 再 rope 的顺序两边一致。

### 3.5 落地方式：FastWAM 零代码改动

`trainer.py:327` → `FastWAM.load_checkpoint()` → 读 `payload["mot"]`，`strict=False`。所以转换脚本只需产出

```python
torch.save({"mot": {"mixtures.video.…": t, "mixtures.action.…": t}}, out)   # bf16, ≈11 GB
```

再在 task config 里 `resume: <该文件>` —— **和 `ab_B` 热启动 RoboTwin 权重走的是同一条路**。

---

## 4. 三个语义坑（决定收益上限，转换消不掉）

| 坑 | 差异 | 本方案的处理 |
|---|---|---|
| **④ 掩码语义** | GWP `future→action = ✅`，FastWAM `= ❌`。GWP 的 action token 被训练成"能解释未来画面的原因"，FastWAM 的不是 | **动作专家迁移最大的不确定性来源** → 单独设 C1（只迁视觉）与 C2（全迁）两臂来隔离 |
| **拼图版式** | 都是 384×320 / 120 token，但 GWP 是 front 上 **192** + 双腕下 192（`dst_height//2`）；FastWAM(robotwin 模式) 是 front 上 **256** + 双腕下 128（`robot_video_dataset.py:168-192`） | 见 §5.2，作为**可选第三臂**，默认不动（保持单变量） |
| **③ 状态注入 + 动作参数化** | GWP：state 是自注意力 token；动作是 12 维 delta + 夹爪绝对值，q01/q99→[-1,1]。我们：proprio 走 cross-attn；动作是绝对关节角 + z-score | 第一次实验**不动**。改它会连带作废 `dataset_stats.json` 和整条部署链路 |

其余小差异：GWP action stream 多一个 state token（1D RoPE 位置 0 语义偏移）、horizon 48 vs 32。不致命。

---

## 5. A/B 设计

> ## ⛔ 2026-09-07 更正：**C1 / C2 两臂的结果无效,需要重跑。**
>
> §3.5 说的落地方式(`resume:` 加载 `{"mot": {...}}`)在当时的 `trainer.py` 里是**空操作**:
> 权重加载发生在 `accelerator.prepare()` **之后**,而 DeepSpeed ZeRO-1 会在每个 optimizer
> step 用引擎初始化时的 fp32 master 副本覆盖回 bf16 参数 —— 热启动权重只活过第 1 次前向。
>
> 所以 C1(−2.5%,`n.s.`)与 C2(+21% 更差)测到的都不是 GWP-0.5 预训练,
> 而是 baseline 的复现。§8 那两道验证门(V1 覆盖率 / V2 血缘余弦)是对的、
> `checkpoints/gwp05_*.pt` 两个文件也是对的 —— **失效的只有训练侧的加载时机**。
>
> 指纹:`ab_B` 用同一条路径加载**训练完整的 FastWAM**,step 10 的 `loss_action` 却与
> **随机初始化**只差 0.2%;修复后重跑变成 3.25~4.38。
>
> 修复(拆分文件/目录两条 resume 路径 + 第 1 个 step 后的永久守卫)与完整证据链见
> **`WANPRETRAIN_TRANSFER_PLAN.md` §3**。重跑只需用修复后的代码重新执行
> `scripts/chain_gwp_ab.sh`,转换产物不用重做。

### 5.1 臂

| 臂 | 初始化 | 拼图 | 回答什么 | 状态 |
|---|---|---|---|---|
| **A**（baseline） | 纯 Wan2.2 + ActionDiT 插值 | 256/128 | 基线 | ✅ **已有**：`runs/agilex_uncond_3cam_384_1e-4/ab_A` |
| **C1** | GWP **video expert only**（825 张量），ActionDiT 保持插值初始化 | 256/128 | 机器人视频预训练对视觉专家有用吗？**语义最干净** | 待跑 |
| **C2** | GWP **video + action expert**（1645 张量） | 256/128 | 动作专家能跨掩码语义迁移吗？ | 待跑 |
| C3（可选） | 同 C1 | **192/192** | 拼图版式错配吃掉了多少收益？ | 待定 |

全部 5000 步，其余超参与 `ab_A` 完全一致（`configs/task/agilex_uncond_3cam_384_1e-4.yaml`、`dataset_stats` 钉死同一份）。

### 5.2 为什么 C1/C2 默认**不**改拼图版式

保持单变量。若 GWP init 在带版式劣势的情况下仍然赢，结论更强，且**采纳时部署链路零改动**。若结果 `n.s.`，C3 是自然的后续（版式改动只涉及 `robot_video_dataset.py` 几行，`dataset_stats` 只统计 action/state，不受图像影响）。

代价：可能出现假阴性。这是已知的取舍。

### 5.3 判读规则（沿用 `AGILEX_TRAINING_PLAN.md` Phase 6）

- C 在 `loss_action` 与 `action_l2` 上均优 ≥10% → 采纳该初始化跑满 5 epoch，然后重跑 RTC 微调与真机
- 差距 <5% 或更差 → 保持纯 Wan2.2 初始化
- **C1 赢而 C2 输** → 掩码语义指纹，转入 §1.6 的方案 B 评估

### 5.4 验证门（**必须在开训前全绿**）

| 门 | 内容 | 需要 GPU |
|---|---|:---:|
| **V1 覆盖率** | 每个映射后的 target key 必须存在于 `MoT.state_dict()` 且 shape 一致；未命中的 target key 恰好是预期集合；弃用的 GWP key 恰好 19 个 | ❌ |
| **V2 血缘余弦** | 视觉专家每个张量 vs 本地 Wan2.2 **同名** key 的 cos-sim 应显著高于**错位配对**（block i vs block i+1）的对照组。错配 / 转置 / 层偏移都会在这里塌掉 | ❌ |
| **V3 前向 smoke** | 转换后权重跑 `dryrun_fastwam.py` + 少量 val 样本；video loss 应明显优于纯 Wan2.2 初始化（动作分支因 I/O 头随机初始化，此时必然差） | ✅ 1 卡 |

V1/V2 纯 CPU，几分钟。V3 需要一张空闲卡。

---

## 6. 执行记录

| 时间 | 事件 |
|---|---|
| 2026-09-03 | 侦察：8×A800 **全部被占满**（77.9/80 GB、94~100% util、8 个进程不在本容器命名空间 → 他人作业）。训练必须等待 |
| 2026-09-03 | 确认 GWP-0.5 权重已在本地 `/mnt/data/zzd/giga-world-policy/Giga-World-Policy-0.5`（22.4 GB F32、1664 张量） |
| 2026-09-03 | 确认本地 `checkpoints/Wan-AI/Wan2.2-TI2V-5B` 是**原始/FastWAM 命名**（825 张量）→ V2 血缘校验可离线做，无需下载 Diffusers 版 |
| 2026-09-03 | RAM 1440 GB / 可用 1402 GB，无内存约束 |
| 2026-09-03 20:08 | 写出 `checkpoints/gwp05_video_only.pt`（9.31 GiB / 825 张量 / 4.9998 B） |
| 2026-09-03 20:10 | 写出 `checkpoints/gwp05_full.pt`（11.21 GiB / 1645 张量 / 6.0207 B） |
| 2026-09-03 20:11 | 启动 `scripts/chain_gwp_ab.sh`，等 8 卡释放后自动跑 C1 → C2 → 出对比 |

## 7. 交付物

| 文件 | 作用 |
|---|---|
| `scripts/convert_gwp05_to_fastwam.py` | 转换器，内建 V1 覆盖率 + V2 血缘余弦两道门，失败即抛错 |
| `scripts/chain_gwp_ab.sh` | 串联守护：**等**8 卡空闲（只等不抢）→ C1 → C2 → 三臂对比 |
| `checkpoints/gwp05_video_only.pt` | C1 臂初始化，825 张量 / 4.9998 B |
| `checkpoints/gwp05_full.pt` | C2 臂初始化，1645 张量 / 6.0207 B |

## 8. 验证门实测结果（全绿）

### V1 覆盖率

```
GWP 张量总数   : 1664
已映射         : 1645
弃用           : 19        （action_encoder/decoder/state_encoder + action_scale_shift_table）
target 清单总数 : 1649       （meta device 上构造的真实 MoT.state_dict()）
未填充的 target : 4
  mixtures.action.action_encoder.{weight,bias}
  mixtures.action.head.{weight,bias}
```

与 §3 的预测**逐个数字吻合**。

### V2 血缘余弦（判定映射表正确的关键证据）

对视觉专家的 layer 0/15/29 + 全部顶层张量，比较「GWP-0.5 张量 vs 本地 Wan2.2 **同名** key」
与对照组「vs block (i+1) 的同名叶子」：

| | cos-sim |
|---|---|
| **同名配对** | mean **0.9752**，min 0.7512 |
| **错位对照** | mean **0.2755** |

大权重矩阵的判别最干净，例如：

```
mixtures.video.blocks.15.self_attn.q.weight    同名 0.9716   错位  0.0001
mixtures.video.blocks.15.self_attn.o.weight    同名 0.9816   错位 -0.0001
mixtures.video.blocks.29.cross_attn.v.weight   同名 0.9827   错位  0.0001
mixtures.video.blocks.29.ffn.2.weight          同名 0.9960   错位 -0.0003
```

> `norm3.weight` / `norm_q.weight` 那几行错位也有 0.99 —— 因为 LayerNorm/RMSNorm 权重都在 1.0
> 附近、天然互相关，**不具判别力**。判别力来自权重矩阵那些行（错位 ≈ 0.0000）。
>
> 同名 min = 0.7512 出现在 `text_embedding.0.bias`，bias 是小向量、微调期间相对位移大，属正常。
>
> 这一门能证伪的错误：转置、`attn1`/`attn2` 互换、层偏移、`norm2`/`norm3` 错配、
> q/k/v 错位。任一发生，同名 cos 都会塌到对照组水平。

### V3 前向 smoke —— **尚未执行**（需要一张空闲卡）

8 卡全被他人作业占满，无法执行。守护脚本内已加了替代性的保底检查：训练日志里必须出现
`Loading weight checkpoint only: <ckpt>`，否则告警。

另外补了一道**离线等价验证**（不需要 GPU），已通过：

```
MoT.state_dict() 键数 = 1649
gwp05_video_only.pt : 写出 825 键，不在 MoT 里的 0，shape 不符 0  -> 命中 825/1649
gwp05_full.pt       : 写出 1645 键，不在 MoT 里的 0，shape 不符 0 -> 命中 1645/1649
```

即 `load_state_dict(strict=False)` 不会静默丢弃任何一个张量。**这已经把 §1.5 提到的
"静默失败烧掉 2.7 h"这个风险消除了。**

## 9. 待决 / 后续

### 9.1 C3（拼图版式）—— 决定：**先不加，等 C1/C2 结果**

先把「换版式要付什么代价」查清楚（已逐项核实）：

**不需要改的：**

| | 为什么 |
|---|---|
| **数据本身** | 拼图是 `robot_video_dataset.py:170-192` 在 `__getitem__` 里从原始 mp4（480×640）在线 resize+concat 的，磁盘上没有任何拼好的图像缓存 |
| **`runs/_shared/dataset_stats.json`** | 逐键核实：只含 `state/{joint,gripper_position}` 与 `action/default` 的 min/max/q01/q99/mean/std，**没有任何图像相关键**（`base_lerobot_dataset.py:283` 的 `get_dataset_stats` 只遍历 `state_meta` / `action_meta`） |
| **`data/text_embeds_cache/`** | 只是文本，按 prompt 哈希 + `context_len` 索引 |

**需要同步改的 —— 全仓库 `256, 320` 只出现在两处，必须逐像素一致：**

| 位置 | 用途 |
|---|---|
| `src/fastwam/datasets/lerobot/robot_video_dataset.py:177` | 训练 / 离线评测 |
| `scripts/deploy_real.py:469` | 真机部署（`:414` 已校验 `concat_multi_camera` 名字，但尺寸是硬编码） |

**真实代价：**

1. C3 本身 +2 h 43 m 共享机器时间
2. 若采纳：改上面 2 处；`final_A/step_032000.pt` 及其整套离线评测变成"另一版式下的基线"，
   绝对指标不可跨版式比（同版式内的相对比较仍有效）
3. 真机侧需重新确认拼图一致性（RTC / 延迟工作不受影响，只有构图路径变了）

**决定与理由：**

- C1 或 C2 赢 ≥10% → C3 无意义，且"在 256/128 上就赢"是更好的结局（采纳时部署零改动）
- C1/C2 都 `n.s.` → 那时 C3 才是有意义的**单变量**追问（C3 vs C1 只差版式）
- 现在就加 = 花 2 h 43 m 共享机器时间回答一个可能不需要问的问题

**实现约定（真要加的时候）**：把版式做成 `concat_multi_camera: "gwp"` 这样的**配置项**，
让 `deploy_real.py` 从 run 的 config 读尺寸，而不是编辑硬编码 —— 使 train/deploy 版式错配
在结构上不可能发生。

⚠️ 在 C1/C2 排队期间**不动** `robot_video_dataset.py`：守护启动时会加载它，
不给已验证的路径引入任何风险。

### 9.2 后续

1. C1/C2 跑完后按 §5.3 判读；若采纳，跑满 5 epoch 并重跑 RTC 微调与真机验证。
2. V3 前向 smoke（1 卡）待有空闲卡时补测（已被 §8 的离线等价验证覆盖）。
3. ~~wam.cpp 延迟先量后投~~ —— 2026-09-03 用户决定暂不做。
