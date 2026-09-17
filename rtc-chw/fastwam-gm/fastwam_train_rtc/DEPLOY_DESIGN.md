# FastWAM Agilex 真机部署设计与踩坑

> 代码:`scripts/deploy_real.py`(主体)、`scripts/deploy_robot_client.py`(线协议)、
> `configs/deploy/agilex_real.yaml`(配置)。
> 训练与离线评测的结论见 `NEXT_SESSION.md`;离线评测的设计见 `OFFLINE_EVAL_DESIGN.md`。
> 生成:2026-09-01

---

## 1. 链路与职责划分

```
GPU 机(8xA800)                                机器人 PC
┌────────────────────────────────────────┐    ┌──────────────────────────┐
│ scripts/deploy_real.py                 │    │ start_robot_arm_service   │
│  ├─ FastWAM 6.02B, bf16, 单卡           │    │  ├─ ROS2: 3 相机 + 2 臂   │
│  ├─ ObsPipeline(与训练逐算子对齐)       │◄──►│  ├─ Piper SDK JointCtrl  │
│  ├─ infer_action -> 32x14 归一化动作     │TCP │  └─ obs: JPEG + state    │
│  ├─ denorm_action -> 物理量              │9900│                          │
│  ├─ SafetyFilter                        │    └──────────────────────────┘
│  └─ 控制环 sync / async                  │
└────────────────────────────────────────┘
```

模型只吃**单帧图像 + 单帧 proprio**(`infer_action` 的 `input_image` 是 `[1,3,H,W]`、
`proprio` 是 `[1,14]`),所以部署侧**不需要历史缓冲**,这一点比多数 VLA 简单。

### 为什么不复用 openpi 的 `AgilexEnv`

| 原因 | 细节 |
|---|---|
| numpy 大版本冲突 | fastwam env 是 numpy **2.2.6**,openpi 钉死 `numpy>=1.22,<2.0` |
| cv2 没装 | fastwam env 无 opencv,而 `AgilexEnv._decode_images` 依赖 `cv2.imdecode` |
| 依赖面 | 还要拖进 `geekrl.core.env`,为 4 个命令不值得 |

所以 `deploy_robot_client.py` 自带实现:`msgpack_numpy` 那 40 行**逐字搬自**
`openpi_client/msgpack_numpy.py`(键名 `__ndarray__`/`__npgeneric__` 是线协议契约,不能改),
JPEG 解码改用 PIL。已实测**双向互操作**通过(我方编码 ↔ openpi 解码,两个方向都对)。

---

## 2. 六个必须对齐的契约

错了大多**不会报错**,只表现为动作乱、抓不稳 —— 所以每一条都在 `preflight` 或 `selfcheck` 里查死。

### 2.1 两套 14 维排布不同(最容易写错)

```
proprio = [ 12 关节(左 j0..j5, 右 j0..j5) , 2 夹爪(左, 右) ]
action  = [ 左 j0..j5 , 左夹爪 , 右 j0..j5 , 右夹爪 ]      <- 夹爪插在下标 6 和 13
```

- proprio 的顺序由 `ConcatLeftAlign` 按 `shape_meta.state` 的书写顺序决定(joint 然后 gripper_position)
- action 的顺序来自数据集 `meta/info.json` 的 `action.names`
- service 的 `step` 用的正是 **action 排布**(`robot_arm_service.py:467-468`:
  `action[:6]`→左臂、`action[6]`→左夹爪、`action[7:13]`→右臂、`action[13]`→右夹爪)

代码里用 `ACT_JOINT_IDX` / `ACT_GRIP_IDX` 显式表达,不写裸切片。

### 2.2 夹爪单位:训练 0~1 归一化,service 用米

| | 训练数据 | RobotArmService |
|---|---|---|
| proprio 夹爪 | `gripper_position`,归一化,实测 min **-0.031** / max **0.956** | `msg.position[6]`,**米**,0~0.07 |
| action 夹爪 | `action[6]/[13]`,归一化,实测 min **-0.088** / max **1.020** | `round(m*1e6)` clip 到 `[0,70000]` |

**满开度 = 0.07 m**(已确认),故 `frac = m / 0.07`、`m = frac * 0.07`。
配置项 `units.gripper_travel_m`。`preflight` 会打出「实测米值 → 换算分数 → 训练统计区间」
三方对照,越界默认**拒绝启动**(`preflight.abort_on_gripper_out_of_range`)。

### 2.3 相机改名 + 顺序

service 给 `cam_top` / `cam_left_wrist` / `cam_right_wrist`;训练要
`cam_high` / `cam_left_wrist` / `cam_right_wrist`。映射写在 `robot.cameras`。

**顺序不能变**:`concat_multi_camera="robotwin"` 把 `video[0]` 缩到 256x320 作上半屏、
`video[1]/video[2]` 各缩到 128x160 左右并排作下半屏(`robot_video_dataset.py:170-194`)。
顺序错了 = 顶部相机被塞进腕部画面的位置,PSNR 看着还行,动作全错。

### 2.4 JPEG 通道:用 PIL **不要**再翻

service 做的是 `cv2.imencode(".jpg", rgb[:, :, ::-1])`。cv2 把入参当 BGR 处理,
而入参正好是真图的 BGR,所以**落盘 JPEG 的颜色是正确的**。于是:

- `cv2.imdecode()` 返回 BGR → openpi 的 `AgilexEnv` 必须 `[:, :, ::-1]` 翻回 RGB
- `PIL.Image.open()` 直接返回 **RGB** → **不能再翻**

翻错了模型看到的是蓝红互换的世界,不报任何错。已用红/绿/蓝三色块图实测验证。

### 2.5 归一化统计量

必须是训练那一份 `runs/_shared/dataset_stats.json`。用错/缺失会让动作完全失真且不报错。

另外 `SingleFieldLinearNormalizer.forward` 会把 z-score **clamp 到 ±5**
(`normalizer.py:129`),所以真机状态严重 OOD 时信息会被截断 ——
`preflight` 会打 `max|z|` 并在 >5 时告警。

### 2.6 权重要真加载上

`load_state_dict(strict=False)`(`fastwam.py:1214`)键名不匹配会被**静默忽略**,
跑出来是随机初始化的分数还看着挺"正常"。本模型应为 **1649/1649**,不匹配直接退出
(逻辑与 `offline_eval.load_and_verify` 一致)。

---

## 3. 时间尺度与执行模式(核心取舍)

`fps=30` → 1 个动作步 = **33.3 ms**;`action_horizon=32` = 1.067 s。

### 3.0 实测延迟(`benchmark` 子命令,A800 / bf16 / 本机)

| num_inference_steps | 推理 | 预处理 | 合计 | = 控制周期 |
|---|---|---|---|---|
| 10 | 413 ms | 4 ms | **416 ms** | **12.5** |
| 5 | 247 ms | 4 ms | **250 ms** | **7.5** |

**⚠️ `OFFLINE_EVAL_DESIGN.md:246` 的 0.42 s 是只算 `infer_action`、不含预处理的数**
—— 离线评测的拼图在 dataloader worker 里做,从不计入。部署必须把预处理算进预算。
本机 10 步的 `infer_s` = 413 ms,与那个 0.42 s 吻合,所以两边并不矛盾。

**CPU 线程数影响很大,而且两个阶段要的值相反**:

| 配置 | 推理 | 预处理 | 合计(10 步) |
|---|---|---|---|
| 定 16 线程 | 564 ms | 58 ms | 622 ms |
| 定 80 线程(不限) | 428 ms | 290 ms | 718 ms |
| **分阶段 pre=16 / infer=80** | **413 ms** | **4 ms** | **416 ms** |

预处理是 CPU 小张量(3 相机 resize + 拼图),线程过订阅时同步开销吃掉一切;
去噪循环相反,线程多的快。所以 `inference.torch_threads_pre` / `torch_threads_infer`
分开设置(在 `infer_chunk` 里逐阶段 `torch.set_num_threads`)。

> 分阶段那一行的预处理 4 ms 比「定 16 线程」的 58 ms 还好一个数量级,**这个差距的完整机制没查清**
> (推测与 intraop 线程池的重建时机有关)。数值可重复,但**属于机器/负载相关的经验值** ——
> 换机器或负载变了请重跑 `benchmark`,别照抄这里的数。

动作误差随 chunk 步号近似线性增长(离线实测,`figs/14_angle_error_deg.png`):

```
error(k) ≈ 1.37 + 0.232 * (k-1)  度      k=1 -> 1.37 度,  k=32 -> 8.55 度
```

### 3.1 sync(默认)

执行 `replan_steps` 步 → 停下驻留 `settle_steps` → 取新观测 → 阻塞推理 → 下一 chunk。

按实测 416 ms(10 步去噪)、`settle_steps=3`:

| replan_steps | 执行时长 | 占空比 | 有效频率 | 用到的 chunk 步 | 末步误差 |
|---|---|---|---|---|---|
| 5 | 0.167 s | 24% | 7.3 Hz | 1–5 | 2.3 度 |
| **10(默认)** | **0.333 s** | **39%** | **11.8 Hz** | **1–10** | **3.5 度** |
| 16 | 0.533 s | 51% | 15.2 Hz | 1–16 | 4.9 度 |

换成 5 步去噪(250 ms)时,replan=10 的占空比升到 **49%**。

- ✅ 严格遵守交接文档「`replan_steps` ≤ 10」,只用误差最小的前 10 步
- ✅ 每次都从**静止、已稳定**的状态重新规划,延迟误差不累积。
  「空箱子里挑货」本来就是准静态任务,这个 regime 最稳
- ❌ 每 0.33 s 冻结 0.42 s,整体放慢到约 39% 速度,动作一顿一顿

**⚠️ 规划前必须重取观测。** `step` 返回的 obs 是"刚下发指令、手臂还没走到"时刻采的 ——
这与训练语义一致(实测状态 → 未来 32 步指令)。但推理阻塞的 0.4 s 里手臂仍在走向上一个目标,
所以要先用**保持指令**(重发上一条 `cmd`)驻留 `settle_steps` 步再取 obs,
否则规划输入是过期状态。默认 3 步 = 100 ms。

### 3.2 async(`--mode async`)

后台线程推理,算完按**实际经过步数**对齐拼接:新 chunk 的 t=0 对应发起推理那一刻,
所以要从第 `elapsed` 步接入。按实测:

| 去噪步数 | 接入步号 | 该处误差 |
|---|---|---|
| 10 | 13 | 4.2 度 |
| 5 | **8** | **3.0 度** |

- ✅ 连续 30 Hz,不停顿
- ❌ 10 步去噪时要用 chunk 第 13 步之后,**违反**交接文档的「≤10 步」建议;
  5 步去噪时接入点回到第 8 步,就**不再违反**了

chunk 用尽而新的还没算完时**保持上一条指令**(不外推)并计数告警;
若 `elapsed >= action_horizon`(整个 chunk 都过期)会单独报警。

### 3.3 推荐的演进路径

1. 先按默认 **sync + 10 步**跑通真机,确认方向、单位、抓取都对
2. 离线复测 5 步去噪的精度:
   ```bash
   python scripts/offline_eval.py --config configs/eval/final_A_newval71.yaml \
       --num-inference-steps 5 --out-dir /tmp/eval_ns5
   python scripts/visualize_eval.py --sweep /tmp/eval_ns5
   ```
   与 `eval_offline/final_A_newval71` 的 10 步结果做**逐样本成对差分**;标 `n.s.` 就是精度没掉
3. 精度没掉就上 **async + 5 步**:第 8 步接入(误差约 3.0 度)、连续 30 Hz,
   既不违反「≤10 步」也没有顿挫 —— 这是终态
4. 还想更快:`inference.compile_action_infer=true`(已透出,**尚未实测收益**)

---

## 4. 安全层

按顺序作用在**反归一化后的 14 维绝对指令**上:

1. **首步跳变检查** —— chunk 第 1 步与实测状态的关节差:
   超 `warn_first_step_jump`(0.15 rad)计数告警(由下一步的钳位吸收);
   超 `abort_first_step_jump`(0.60 rad)**直接中止 episode**。
   这个量级的跳变基本只可能来自:单位换算错、相机映射错、权重不对、初始位姿严重 OOD。
2. **逐步 delta 钳位** —— **累积式**:第 k 步以第 k-1 步的钳位结果为基准。
   如果模型想跑得比上限快,执行轨迹会落后于预测轨迹且本 chunk 内追不回来 ——
   这没关系,每 `replan_steps` 步就从实测状态重新规划,误差不累积。
3. **绝对限位** —— 钳到训练 `action` 的 `global_min/max` 再放 `limit_margin_*`。
   模型没见过的关节区域不去。
4. **夹爪物理钳位** —— 分数 clip 到 `[0,1]` 再换米(`to_service_action`)。

### delta 上限怎么定的

不是拍的数。统计训练数据 **60 个 episode / 129,762 个转移**的步间动作差分:

| | mean | p99 | p99.9 | max |
|---|---|---|---|---|
| 12 关节(rad / 33.3ms) | 0.008 | 0.059 | **0.097** | 0.815(离群) |
| 2 夹爪(分数 / 33.3ms) | 0.007 | 0.124 | **0.198** | 0.689 |

取 **p99.9**:`max_delta_joint=0.10`(≈172 度/s)、`max_delta_gripper=0.20`。
放过示教里 99.9% 的真实速度,挡住 0.8 rad(46 度)的单步跳变。

---

## 5. 已查出的三处与 openpi service 的不一致

### 5.1 初始位姿在训练分布边缘之外(**决定:只告警,不干预**)

service 的 `_DEFAULT_INIT_JOINTS_*` 与训练 episode 起始位姿(40 个 episode 实测)对比:

| 维 | 训练起始 mean±std | 训练起始 [min, max] | service init | z |
|---|---|---|---|---|
| 左 j3 | -0.296 ± 0.149 | [-0.558, +0.022] | **+0.092** | **+2.6σ,越界** |
| 左 j5 | +0.362 ± 0.142 | [+0.036, +0.595] | **-0.112** | **-3.3σ,越界** |
| 右 j1 | +0.269 ± 0.090 | [+0.003, +0.456] | +0.457 | +2.1σ,压线 |
| 左夹爪 | 0.636 | [0.412, 0.849] | **0.000 m → 0.0** | **完全闭合,越界** |

姿态家族是对的(明显同一套设备),但**左腕滚转与左夹爪开合在训练里从没出现过**。
第一帧 OOD 会污染整条 rollout。

当前实现:`preflight` 统计训练数据第 0 帧的分布(缓存到 `runs/_shared/start_pose_stats.json`),
逐维打 z-score 与越界告警,**不自动干预**。

若要修:两条路 —— (a) 改机器人 PC 上 `RobotArmService` 的 `_DEFAULT_INIT_JOINTS_*`;
(b) 在本脚本里加一段慢速斜坡,用 `step` 走到训练起始位姿。当前都没做。

### 5.2 协议没有"只读观测"命令

只有 `ping / reset / step / stop`:`reset` 会以 20% 速度走初始位姿并 sleep 5s(**真实运动**),
`step` 必然下发 `JointCtrl`。所以「连上去看一眼观测但不碰手臂」**做不到**。

`RobotArmClient.try_obs()` 已预留:它发 `{"cmd":"obs"}`,服务端不支持就返回 `None`。
想启用只需在机器人 PC 的 `robot_arm_service.py` 的 `_handle()` 里加两行(纯读,零风险):

```python
if cmd == "obs":
    return {"obs": self._get_obs()}
```

### 5.3 退出时不归零

`_do_stop()` 会 `go_zero()` 把双臂打到全零位 —— 从弯曲姿态出发可能扫过工作区撞箱子。
默认**不发 `stop`**,只断开连接(手臂保持当前位姿通电驻留)。
要归零显式设 `robot.go_zero_on_exit: true`。

---

## 6. 联调顺序

| 级别 | 命令 | 需要硬件 | 验证什么 |
|---|---|---|---|
| 1 | `deploy_real.py selfcheck` | ❌ | 部署侧预处理与训练**逐元素一致** |
| 2 | `deploy_real.py benchmark` | ❌ | 推理延迟 → 定 `replan_steps` |
| 3 | `deploy_fake_service.py` + `run` | ❌ | 整个 run 路径:preflight / 控制环 / 安全层 / 落盘 |
| 4 | `python scripts/deploy_robot_client.py --endpoint ...` | ✅ | 连通性、单位、相机名(有 `obs` 命令时纯读、不动手臂) |
| 5 | `deploy_real.py run --max-steps 60 --confirm` | ✅ | 真机 2 秒小步试 |
| 6 | `deploy_real.py run --num-episodes N --record-dir ...` | ✅ | 正式 rollout |

第 3 级用 `scripts/deploy_fake_service.py`:它实现同一套线协议、模拟一阶跟踪动力学,
并且喂**真实数据集帧 + 真实起始位姿**(所以模型输出是合理动作,安全层才测得出东西)。

```bash
python scripts/deploy_fake_service.py --extract          # 抽一帧,只做一次
python scripts/deploy_fake_service.py --port 19912 & FAKE_PID=$!
python scripts/deploy_real.py run --config configs/deploy/agilex_real.yaml \
    --endpoint tcp://127.0.0.1:19912 --max-steps 40 --record-dir /tmp/deploy_rec
kill $FAKE_PID
# 想看 OOD 起始位姿会不会被抓到:加 --init openpi 起服务
```

> ⚠️ **别用 `pkill -f deploy_fake_service.py` 收尾。** `-f` 匹配整条命令行,
> 如果你的命令行里本来就含这个文件名(比如上面几行写在同一条 shell 命令里),
> pkill 会把**你自己的 shell** 一起杀掉 —— 表现是命令静默退出、后面的 echo 都不打印。
> 用 `$!` 记下 PID 再 `kill` 最稳。

`selfcheck` 是最有价值的一步,它做四件事:

1. 用训练数据集的一个样本,把**原始逐相机图**过部署侧 `ObsPipeline`,
   与数据集自己产出的拼图**逐元素比对**(临时摘掉 processor 拿 preprocess 之前的原始 sample,
   见 `base_lerobot_dataset.py:269`)
2. proprio 归一化逐元素比对
3. 量化 **JPEG q95 传输往返**对拼图和动作的影响(这是传输的固有代价,不是 bug)
4. 端到端跑一次 `infer_action`,报对 GT 的首步/末步关节误差,与离线评测的参考值(1.4 度 / 8.6 度)对照

---

## 7. 落盘内容(`--record-dir`)

| 文件 | 内容 |
|---|---|
| `episode_XXX.npz` | 逐步:时间戳、chunk 编号、chunk 内步号、物理指令、service 单位指令、实测状态;逐 chunk:完整的归一化/物理量/钳位后动作、proprio、推理耗时、钳位计数 |
| `episode_XXX_summary.json` | 步数/chunk 数/有效频率/推理延迟/安全层计数/权重信息/instruction |
| `episode_XXX_model_view.mp4` | 模型每次规划时**真正看到的** 384x320 拼图,fps = 规划频率 |

`model_view.mp4` 是最值得先看的 —— 它能立刻暴露相机映射错、通道翻转、拼图错位这类问题,
也可以和 `eval_offline/*/ckpt/*/videos/*.mp4` 的 GT 部分做人眼比对。

---

## 8. 已知坑

| 坑 | 说明 |
|---|---|
| **两套 14 维排布不同** | proprio `[12 关节, 2 夹爪]` vs action `[左6, 左爪, 右6, 右爪]`。用 `ACT_*_IDX` 别写裸切片 |
| **夹爪单位** | 训练 0~1、service 米,满开 0.07 m。错了不报错,只是抓不稳 |
| **PIL 解 JPEG 不翻通道** | 与 openpi 用 cv2 的写法相反,详见 §2.4 |
| **相机顺序硬编码** | `robotwin` 拼图假定 `video[0]` 是顶部相机 |
| **sync 模式要重取观测** | 推理阻塞期间手臂还在动,不驻留取新 obs 就是拿过期状态规划 |
| **`ToTensor` 断言 uint8** | `transforms/image.py:11`,所以逐相机变换前必须保持 uint8,不能提前转 float |
| **normalizer 会 clamp ±5** | `normalizer.py:129`,严重 OOD 时信息被截断且不报错 |
| **`_crop` 断言 ndim==3** | `action_state_merger.py:56-57`,反归一化时 action 和 state **都要带 batch 维** |
| **`load_state_dict(strict=False)`** | 键名不匹配静默丢权重,应为 1649/1649 |
| **text embedding 按 sha256 找** | `instruction` 与训练 `tasks.jsonl` 差一个字符就找不到缓存;缓存目录里只有那一条 |
| **`infer_action` 无 `num_video_frames`** | 那是 `infer_joint` 的参数,传了会 TypeError |
| **`control.hz` 必须 = 30** | 数据集 fps=30,改了动作步长就与训练不符 |


---

## 9. 已做过的验证(2026-09-01,均无硬件)

| 验证 | 结果 |
|---|---|
| 线协议 + msgpack 编解码 | ✅ 与真实 `openpi_client.msgpack_numpy` **双向互操作**通过 |
| JPEG 通道 | ✅ 红/绿/蓝三色块图往返后主色仍是 R/G/B(证明 PIL 路径不该翻通道) |
| `try_obs()` 降级 | ✅ 服务端不支持 `{"cmd":"obs"}` 时返回 `None` 而非抛异常 |
| **拼图与训练一致** | ✅ **max\|Δ\| = 0.000e+00**(fp32 逐元素) |
| **proprio 归一化一致** | ✅ **max\|Δ\| = 0.000e+00** |
| 权重加载 | ✅ 1649/1649 键全匹配 |
| 文本条件 | ✅ 命中缓存,`context=(128,4096)` |
| 夹爪单位往返 | ✅ `0.693 frac → 0.0485 m → 0.693 frac`;落盘里 action[6] 确为米 |
| 模型视角拼图(人眼) | ✅ 上=俯视双箱、下=双腕;**蓝箱子是蓝的**(通道没翻) |
| sync 控制环 | ✅ 40 步 / 4 次规划,首步跳变 1.0~3.8 度,安全层 0 次钳位 |
| async 控制环 | ✅ 45 步,有效 17.3 Hz,节拍 0 次超时 |
| OOD 起始位姿告警 | ✅ `--init openpi` 时正确报出 `R_j1` / `L_grip` / `R_grip` 越界 |
| 落盘 | ✅ npz(逐步 + 逐 chunk)、summary.json、model_view.mp4 齐全 |
| 延迟 | ✅ 见 §3.0 |

上表用 `step_077690.pt`(手边现成的)跑的 —— 流水线一致性与权重无关。
**换成真正要部署的 `step_032000.pt` 后,第 1/2/3 级全部重跑并通过**:

| | 结果 |
|---|---|
| 权重校验 | 1649/1649 |
| 拼图 / proprio | 均 `max\|Δ\| = 0.000e+00` |
| 延迟 | 10 步 446 ms(7.7~13.4 个控制周期);5 步 258 ms |
| 假 service run | 40 步 / 4 次规划,首步跳变 **1.2~2.2 度**,安全层 0 次钳位,0 warning |

**仍未验证的**(必须真机):真实 ROS2 相机的分辨率/FOV 是否与训练数据一致、
Piper 实际跟踪能力与 `max_delta_joint` 是否匹配、以及最终的**任务成功率**。

### 验证时顺带发现并修掉的问题

1. `selfcheck` 最初报拼图差 `1.95e-3` —— **不是流水线错**,是我拿 bf16 转换后的张量
   去和 fp32 参考比(bf16 在 1.0 附近间隔 2^-8)。逐级 bisect 显示每一级都严格相等。
   已改成 `compose_image(cast=False)` 走 fp32 比对,阈值收紧到 1e-6。
2. msgpack 解出的 ndarray 是用 `np.ndarray(buffer=<bytes>)` 构造的 → **只读**,
   `torch.as_tensor` 会告警且原地写是未定义行为。已在 `_decode_obs` / `_decode_jpegs` 里一律 copy。
3. sync 模式下 `Pacer` 把**故意的推理冻结**记成"控制周期超时"(报 935 ms 超时),
   节拍统计失去意义。已在规划后 `pacer.reset()`。
4. async 的 `n_stall`(chunk 用尽只能保持上一条指令)原本只打印不汇总。
   已提升为 runner 属性,写进 episode 报告与 `summary.json` 的 `async_stalls`。
5. 假 service 与推理**同机**时会抢 CPU(它要 JPEG 编码 3 路 480x640),
   实测推理从 416 ms 涨到 600~730 ms,async 因此出现 chunk 用尽停顿。
   真机上编码在机器人 PC,不会有这个问题 —— 但这也说明 async 在延迟接近 horizon 时很脆。


---

## 10. 最小部署集(搬到真机时用)

部署目标机:**RTX 4090 / 24 GB**(实测峰值 15.1 GiB,余量约 8 GB,够)。

### 10.1 为什么不是"训练出的那一个 pt"就够

`step_032000.pt` 里只有这些:

| 键 | 内容 |
|---|---|
| `mot` | 1649 个张量,**6.021B 参数**,bf16 → 正好 12.0 GB |
| `proprio_encoder` | 2 个张量,约 0 |
| `step` / `torch_dtype` | 元信息 |

**没有 VAE,也没有文本编码器** —— 训练时这两个是冻结的,不参与梯度更新,所以 trainer 不存
(存了就是每 2000 步白写十几 GB)。但推理时:

| 组件 | 训练时 | 在 checkpoint 里 | 部署是否需要 |
|---|---|---|---|
| `mot`(video expert + action expert) | 训练 | ✅ | 从 checkpoint 加载 |
| **VAE** | 冻结 | ❌ | **必需** —— 相机图要经它编码成 latent |
| T5 文本编码器 | 冻结 | ❌ | 不需要 —— 用缓存好的 embedding 绕开 |

所以"要搬好几个文件"的唯一实质原因是:**VAE 不在 checkpoint 里,而推理离不开它**(1.4 GB)。

### 10.2 底座 DiT 的 20.8 GB 也不必搬(已实测)

原加载顺序是「先读 Wan2.2 底座填进 DiT → 再用 checkpoint 覆盖」。既然 `mot` 是
**1649/1649 全覆盖**,底座那一步纯属浪费。`inference.skip_base_dit_load: true`(默认开)
会把 `model.skip_dit_load_from_pretrain=true` 传进 hydra,于是:

- `loader.py:170-176` 走随机初始化分支,**连 `download_if_necessary()` 都不调** → 文件不必存在
- 同一个开关也管 `ActionDiT.from_pretrained`(`fastwam.py:145-151`)→ 2.0 GB 一并跳过
- `vae_config.download_if_necessary()`(`loader.py:165`)是**无条件**的 → VAE 不受影响

实测(`/tmp/test_skip_dit.py`,同一 seed 同一输入):

| | 开 vs 不开 |
|---|---|
| `mot` 1649 个张量 | **逐比特相同**(不一致 0 个) |
| 动作输出(物理量) | **`max\|Δ\| = 0.000e+00`** |
| 建模耗时 | 71.6s → **54.6s** |

万一将来换的 checkpoint 没能全覆盖,`load_and_verify` 会因缺键**直接退出**,不会静默跑随机权重。

### 10.3 清单

**要搬(13.4 GB)**

| 内容 | 大小 |
|---|---|
| `runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_032000.pt` | 12.0 GB |
| `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | 1.4 GB |
| `runs/_shared/dataset_stats.json` + `start_pose_stats.json` | 92 KB |
| `data/text_embeds_cache/agilex/36a916de….t5_len128.wan22ti2v5b.pt` | 1.0 MB |
| `src/` + `configs/` + `pyproject.toml` + `env.sh` + 三个 `scripts/deploy_*.py` | 2 MB |

**不搬(省 72 GB)**

| 内容 | 大小 | 为什么 |
|---|---|---|
| `Wan-AI/Wan2.2-TI2V-5B/*.safetensors` | 18.8 GB | `skip_base_dit_load: true`,见 §10.2 |
| `ActionDiT_linear_interp_….pt` | 2.0 GB | 同上 |
| `models_t5_umt5-xxl-enc-bf16.safetensors` | 11 GB | `load_text_encoder: false`,走 embedding 缓存 |
| `fastwam_release/robotwin_uncond_3cam_384.pt` | 12 GB | 只被 `resume:` 用,而它是 `null` |
| `data/agilex_empty_the_box_fastwam/` | ~30 GB | **`run` 完全不用数据集**;起始位姿分布带上已生成的 `start_pose_stats.json` 即可(`auto` 会直接命中缓存,数据集缺失时该检查只是跳过,不会崩)。只有 `selfcheck` 需要数据集 |
| 其余 checkpoint、`eval_offline/`、`third_party/` | 36 GB+ | 用不到 |

**搬的命令**

```bash
DST=user@真机IP:/path/to/FastWAM
rsync -av src/ pyproject.toml configs/ env.sh DEPLOY_DESIGN.md \
      scripts/deploy_real.py scripts/deploy_robot_client.py scripts/deploy_fake_service.py $DST/
rsync -avR runs/_shared/ data/text_embeds_cache/agilex/ $DST/
rsync -avR --progress \
  runs/agilex_final_3cam_384_1e-4/final_A/checkpoints/weights/step_032000.pt \
  checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors $DST/
```

### 10.4 搬过去要改的地方

1. **`env.sh`** —— 三处硬编码:`cd /home/gaomeng/FastWAM`、`FASTWAM_ENV=/root/miniforge3/envs/fastwam`、
   `CUDA_HOME`。`DIFFSYNTH_MODEL_BASE_PATH` / `HF_HOME` / `MODELSCOPE_CACHE` 用的是 `$(pwd)`,自动跟着走。
   **这三个变量必须设** —— VAE 是靠 `DIFFSYNTH_MODEL_BASE_PATH` 找到的,不设会尝试联网重新下载。
2. **`robot.endpoint` 改成 `tcp://127.0.0.1:9900`** —— 同机了,走回环。
3. **Python 环境**别手挑包(`deploy_real.py` 为了 `DEFAULT_PROMPT` 会 import 到 lerobot 数据集那一套,
   连带拉进 `accelerate`/`torchcodec`/`datasets`,容易漏)。用
   `conda env export -n fastwam --no-builds > fastwam_env.yml`,或 `conda-pack` 整包搬。
4. **`inference.torch_threads_pre/infer` 与 `control.replan_steps` 必须重新定** —— 见下。

### 10.5 4090 上必须重测的两件事

**(1) 延迟大概率变化,方向不确定。** A800 显存带宽 2039 GB/s、4090 只有 1008 GB/s;
batch=1 的扩散推理偏带宽瓶颈,所以 **4090 可能比 A800 慢**(尽管算力更高)。
本机 446 ms 的数直接搬过去用是不安全的 —— `replan_steps` 与占空比全依赖它:

```bash
python scripts/deploy_real.py benchmark --config configs/deploy/agilex_real.yaml
```

这一步不需要硬件也不需要数据集,搬完立刻能跑,是最快的验收。

**(2) CPU 争抢是新出现的风险。** 原架构里 service 在机器人 PC、模型在 GPU 机,互不干扰。
搬到同一台机器后,**ROS2 图像回调 + 三路 480x640 的 JPEG 编码 + Piper CAN 通信**
与推理抢同一份 CPU。本项目实测过这个效应:把假 service 与推理放同机时,
推理从 416 ms 涨到 600~730 ms,async 因此出现 chunk 用尽停顿。

所以搬过去后:
- 先单独跑 `benchmark`(没有 service)拿到基线
- 再起 service 后跑 `run`,对比 `summary.json` 里的 `infer_s_mean` 涨了多少
- 真机核数远少于 80,`torch_threads_pre: 16` / `torch_threads_infer: 0`(=用满)要重新扫
