# 离线批量评测：设计文档

> 涉及文件：`scripts/offline_eval.py`（推理+存盘）、`scripts/visualize_eval.py`（出图）
> 配置：`configs/eval/*.yaml`
> 创建：2026-08-28　|　升级：2026-08-31　|　配套：`AGILEX_TRAINING_PLAN.md`、`AGILEX_FINETUNE_GUIDE.md`

---

## 0. 当前状态（先看这个）

| 组件 | 状态 |
|---|---|
| `offline_eval.py` | ✅ **已在真实数据上跑通并验证** —— 见下方"验证记录" |
| `visualize_eval.py` | ✅ 9 张图逐张渲染看过 |
| `configs/eval/smoke.yaml` | 链路自检：1 ckpt / 2 样本 / action 模式 / 单卡，约 2 分钟 |
| `configs/eval/final_A_sweep.yaml` | 全量：39 ckpt / 64 样本 / joint 模式 / 8 卡分片 |

### 那个致命 bug（2026-08-28 版从未跑通的原因）

`run_final.sh:149-165` 在 final_A 训练前对 ab_A/ab_B 各跑过一次，两次都在**第一个样本**就崩：

```
.run_final.log:43-54
  offline_eval.py:68 denorm_action → action_state_merger.py:57 _crop → assert x.ndim == 3
```

`denorm_action` 只给 `action` 补了 batch 维，`state`（proprio）还是 `[T,D]`，而 merger 的
`_crop` 要求 3 维。`trainer.py:496` 传的本来就是 `[1,T,D]`，"照抄"时漏了这一层。
因为外面包了 `|| true`，训练完全没受影响，日志里只留下两行 `⚠️ 离线评测失败 rc=1`；
`eval_offline/{ab_A,ab_B}/` 里只有 `dataset_stats.json`，`samples/`、`videos/` 全空。

### 验证记录（2026-08-31）

| 检查 | 结果 |
|---|---|
| 链路 | smoke config 跑通，`samples/*.npz` + `metrics.csv` + `summary.json` 全部落盘 |
| **指标可比性**（关键闸门） | `step_074000` / 32 固定样本，离线 `action_l2` = **0.0227**；训练日志 ±2500 步窗口（11 个 eval 点 = 88 样本）= **0.0246**。差 8%，远小于样本间跨度（单样本 0.0002~0.0687）→ 反归一化与统计量路径正确 |
| 权重校验 | `mot` 键 **1649/1649** 全匹配（与上一会话核对热启动时的数字一致） |
| val split | 自动解出 episode `[36, 97, 449, 461, 136]`，共 9,558 帧 —— 与 `val_set_proportion=0.01` + `rng(42)` 的复算结果一致 |
| episode 指定 | `--episodes 36` → `dataset_total=2437`（该 episode 真实长度），`metrics.csv` 的 `episode_index` 全为 36 |
| 多卡分片 | 3 ckpt × 3 卡，三份 `sample_idxs` **完全相同** → 严格可比 |
| joint 模式 | mp4 = h264 / 320×1152 / 9 帧 / **7.5 fps**（真实播放速率） |
| 换数据集目录 | 指向 `agilex_empty_the_box_all_470` 时 preflight 在 2 秒内报出缺哪 5 个列（该目录列名未转换） |

速度实测：**action 模式 0.42 s/样本，joint 模式 1.36 s/样本**（10 步去噪，A800）。
ckpt 从 OSS 加载约 13 s（918 MB/s × 12 GB）+ 反序列化。

---

## 1. 为什么需要它

仓库原本**没有**"在自己的数据上批量推理并定量看效果"的能力：

| 现有入口 | 为什么不够 |
|---|---|
| `trainer.evaluate()` | 每 `eval_every` 步只采 **1 个样本/卡**，且**每步换一批随机样本**（`trainer.py:436-437` 按 `global_step+rank` 播种）→ **checkpoint 之间不可比**；只存拼接 mp4 + 标量指标，`pred_action` 用完就丢 |
| `experiments/libero/run_libero_manager.py` | 依赖 LIBERO + mujoco 仿真器 |
| `experiments/robotwin/run_robotwin_manager.py` | 依赖 RoboTwin + SAPIEN |
| `experiments/libero/summarize_results.py` | 只是把仿真评测的成功率 json 汇总成表，与真机数据无关 |
| `scripts/dryrun_fastwam.py` | 只测**推理速度**，用合成输入，不看精度 |

真机数据（Agilex）两个仿真器都用不上，所以必须自建。

**最关键的一点**：上一会话为了绕过"每步换样本"带来的噪声（单点 `action_l2` 在
0.0082~0.0654 之间跳 8 倍），只能做 ±2500 步窗口平均。本脚本让**所有 checkpoint 评
完全相同的一批样本**，把 checkpoint 选择从统计规避变成直接比较，并进一步支持
**成对差分**（见 §4.7）。


---

## 2. 架构：两阶段分离

```
   ┌──────────────────────┐        ┌────────────────────────┐
   │  offline_eval.py     │        │  visualize_eval.py     │
   │  推理 + 存盘          │───────▶│  读盘 + 出图            │
   │  需要 GPU，慢         │  文件   │  纯 CPU，秒级           │
   └──────────────────────┘        └────────────────────────┘
```

**为什么分开**：joint 模式推理一个样本约 1.55 s（10 步去噪 + VAE 编解码 + PSNR/SSIM），
64 个样本约 3 分钟且占满 GPU；而调图（改配色、换图形、加指标）会反复迭代。
分开后改图**不用重跑推理**。

这也意味着 `offline_eval.py` 存的是**原始结果**而不是图 —— 你后续想画任何新图，
数据都已经在盘上了。

`offline_eval.py` 内部还分成 **launcher / worker** 两个角色（同一个文件）：
config 里给了多张卡且有多个 ckpt 时，launcher 把 ckpt 列表**轮转分片**到各卡，
每卡起一个 worker 子进程；worker 只建一次模型，然后循环换权重。
样本下标只由 `(split, episodes, num_samples, per_episode, stride, seed)` 决定，
所以分片**不影响可比性**。

---

## 3. 数据契约（改代码前必读）

### 3.1 目录结构

`--config` 模式（可多 ckpt）：

```
<out_dir>/
├── config.resolved.yaml          实际生效的配置快照（worker 读的就是它）
├── ckpt/<tag>/                   每个 ckpt 一个子目录，内部结构见下
├── compare.csv                   每 ckpt 一行的聚合指标（含 step / epoch）
├── sweep.json                    扫描级元信息 + 各 ckpt 聚合指标
├── logs/gpu*.log                 各分片日志
└── figs/                         visualize_eval.py 产出
```

单 ckpt 目录内部（两种模式一致）：

```
<dir>/
├── samples/sample_0000.npz ...   每个样本一个，原始数组
├── videos/sample_0000.mp4  ...   仅 save_video 时；pred/VAE重建/GT 竖向拼接，7.5 fps
├── metrics.csv                   每样本一行的标量指标
└── summary.json                  聚合指标 + 运行元信息 + 时间标注参数
```

> 旧式调用（`--ckpt` + `--out-dir`，不带 `--config`）仍**直接写 out-dir**，
> 不套 `ckpt/<tag>/` —— `run_final.sh:159` 的调用不受影响。

### 3.2 `samples/sample_XXXX.npz`

`T` = action horizon（当前 32），`D` = action 维度（当前 14）。

| key | shape | 含义 |
|---|---|---|
| `pred_action_norm` | `[T, D]` | 模型直接输出（归一化空间） |
| `gt_action_norm` | `[T, D]` | 真值（归一化空间） |
| `pred_action_phys` | `[T, D]` | **反归一化后的物理量**（关节 rad / 夹爪开度） |
| `gt_action_phys` | `[T, D]` | 同上，真值 |
| `proprio_norm` | `[T, D]` | 本体感知（归一化）；反归一化时必需 |
| `abs_err_per_dim` | `[D]` | 该样本逐维平均绝对误差 |
| `abs_err_per_step` | `[T]` | 该样本逐预测步平均绝对误差 |

> 注意 `sample_XXXX` 的编号是**评测序号**（0..N-1），不是数据集下标。
> 数据集下标存在 `metrics.csv` 的 `idx` 列。

### 3.3 `metrics.csv`

| 列 | 含义 |
|---|---|
| `idx` | 数据集内的样本下标 |
| `episode_index` | 该样本属于哪个 episode（原始编号） |
| `frame_in_episode` | 在该 episode 内的起始帧号 |
| `pad_frac` | 该样本的 action 有多少比例是 padding（复制填充）。>0 说明窗口越过了 episode 末尾，误差会被人为压低 |
| `infer_s` | 该样本推理耗时（秒） |
| `action_l2` | **MSE**（不是 RMSE），物理量。与训练日志 `eval/action_l2` **同定义**，可直接对比 |
| `action_l1` | 平均绝对误差，物理量 |
| `action_rmse` | `sqrt(MSE)`，更好解读（关节维单位 rad，0.1 ≈ 5.7°） |
| `action_max_abs` | 单点最大绝对误差（看极端失败） |
| `psnr_rg` / `ssim_rg` | 生成 vs 真实 —— 仅 `--mode joint` |
| `psnr_rd` / `ssim_rd` | 生成 vs VAE 重建 —— **剔除 VAE 损失后的纯生成质量** |
| `psnr_dg` / `ssim_dg` | VAE 重建 vs 真实 —— **这套 VAE 的天花板** |

### 3.4 `summary.json`

```
ckpt, ckpt_tag, step, epoch,            # epoch = step / annotate.steps_per_epoch
weight_check: {mot_total, mot_loaded, mot_missing, mot_unexpected, step_in_payload},
task, split, dataset_dir, episodes[], num_episodes_in_split,
mode, num_samples, num_inference_steps, sigma_shift, action_crosscheck,
dataset_total, action_dim_names[D], num_padded_samples,
annotate: {fps, video_fps, action_dt_s, horizon_steps, horizon_s, steps_per_epoch},
eval_started_at, eval_finished_at, eval_wall_s,
metrics: { <指标名>: {mean, std, sem, min, max, p50, p90, n} },
abs_err_per_dim_mean[D],
abs_err_per_step_mean[T],
abs_err_per_step_p90[T],
sample_idxs[N]                          # 所有 ckpt 必须一致，否则对比不可信
```

`action_dim_names` 从数据集 `meta/info.json` 的 `features.action.names` 读，
拿不到时退化成 `dim0..dimD-1`。`annotate` 是**时间标注**的唯一来源，出图全靠它。

### 3.5 `compare.csv` / `sweep.json`（多 ckpt）

`compare.csv` 每 ckpt 一行：`ckpt_tag, step, epoch, num_samples`，
外加每个指标的 `<key>_mean` / `<key>_sem` / `<key>_p90`。

`sweep.json` 额外带 `reference`（成对差分的基准）、`highlight`（轨迹叠加要画的 ckpt）、
`sample_idxs`、`missing`（哪些 ckpt 没产出），以及各 ckpt 的完整 `metrics`。
合并时会**核对所有 ckpt 的 `sample_idxs` 是否一致**，不一致就打警告 ——
那种情况下整个对比失去意义。

---

## 4. 关键设计决策

### 4.1 反归一化必须走 processor，不能直接乘 std 加 mean

`normalizer` 是按 `shape_meta` 里的**子 key** 分别统计的（本项目是
`action.default` 一个键，但 state 是 `joint` + `gripper_position` 两个），
所以不能对拼接后的 14 维直接反归一化。必须走这一圈：

```python
merger.backward  →  normalizer.backward  →  merger.forward
```

实现见 `offline_eval.py::denorm_action()`，逻辑**照抄 `trainer.py:503-535`**。
这样保证本脚本的 `action_l2` 与训练日志的 `eval/action_l2` 数值可比 —— 这是
本脚本存在价值的一半（另一半是覆盖样本量与可比性）。

⚠️ **`action` 和 `state` 都必须带 batch 维**：`_crop` 断言 `ndim == 3`
（`action_state_merger.py:56-57`）。第一版只给 `action` 补了，`state` 漏了，
结果两次实跑都在第一个样本崩（见 §0）。改这个函数时务必同步对照 trainer。

已实测对齐：`step_074000` / 32 固定样本，离线 `action_l2` = 0.0227 vs
训练日志 ±2500 窗口 0.0246（差 8%，远小于样本间跨度）。

### 4.2 指标一律算在物理量上，不是归一化空间

归一化空间的误差没有物理意义（不同维的 std 差 3 倍）。物理量的 RMSE 可以直接
读成"平均差几度"。trainer 也是这么做的。

### 4.3 视频质量给三组而不是一组

`生成vs真实` 这一个数字会被 VAE 的重建损失污染 —— 即使模型完美，也达不到 100%。
所以同时给：

- `rd`（生成 vs VAE 重建）→ 剔除 VAE 后的**纯生成质量**
- `dg`（VAE 重建 vs 真实）→ **天花板**
- `rg`（生成 vs 真实）→ 端到端观感

判断模型好坏看 `rd`，判断这套 VAE 够不够看 `dg`。

### 4.4 样本选取默认等间隔而不是随机

`np.linspace` 在整个 split 上等间隔取样，保证覆盖各 episode 的不同阶段，且
**完全可复现** —— 这是多 ckpt 可比性的前提。想按固定步长取用 `stride`；
想每个 episode 取一样多用 `per_episode`（覆盖更均衡，长 episode 不会占更多配额）。

**`skip_padded`**：每个 episode 末尾 `num_frames-1 = 32` 帧起头的窗口，GT action 是
被复制填充的（`sliding_window_with_replication`），误差会被人为压低。默认在 sweep
config 里开着，排除的候选数会打印出来（本数据 5 个 episode × 32 = 160 个）。
不开的话 `metrics.csv` 的 `pad_frac` 列可以事后过滤。

### 4.5 `mode: action` 与 `mode: joint`

| | 调用 | 实测速度 | 产出 |
|---|---|---|---|
| `action` | `infer_action` | **0.42 s/样本** | 只有动作指标 |
| `joint` | `infer_joint` | **1.55 s/样本** | 动作 + 视频质量 + 可存 mp4 |

`infer_action` 的参数里**没有** `num_video_frames`，`infer_joint` 有且必填 ——
脚本里按 mode 分别处理（写的时候踩过：参数名不是 `num_frames`）。

⚠️ `infer_joint` 默认 `test_action_with_infer_action=True`（`fastwam.py:788`）：
会**先单独跑一遍 `infer_action`** 做交叉校验，然后返回 joint 通道的 action。
trainer eval 走的也是这条路（`infer` → `infer_joint`），所以默认保持 True 以对齐；
`action_crosscheck: false` 能省约 40% joint 耗时。

另外 trainer eval 会把 GT `action` 传进去做视频条件，但本项目
`video_dit_config.action_conditioned = false`，那个参数被忽略，所以离线不传是等价的。

### 4.6 多卡分片按 ckpt 而不是按样本

ckpt 之间彼此独立，分片零通信；而且样本下标是确定性的，所以**每个 worker 算出来的
`sample_idxs` 完全一样**，分片不影响可比性（合并时会核对）。
按样本分片反而要处理"同一个 ckpt 的结果跨进程拼接"。

⚠️ **必须限每个 worker 的 CPU 线程数**。不限的话 8 个进程各起 80 个 torch 线程，
80 核机器 load average 冲到 265，每样本从 1.55 s 掉到 16 s（**慢 12 倍**）。
瓶颈在 CPU 侧：VAE 解码后的 PIL 转换、PSNR/SSIM、h264 编码都在 CPU 上。
launcher 会自动设成 `总核数 / 卡数`（本机 = 10）。

### 4.7 多 ckpt 对比要做成对差分，不是比均值

所有 ckpt 评的是同一批样本，所以可以做**配对比较**：逐样本相减再求均值。
样本难度带来的方差在相减时被消掉了，灵敏度高出一个量级。

实测（3 个晚期 ckpt / 8 样本）：不配对时 `action_rmse` 的 95% CI 是 ±0.13，
配对后 Δ 的 CI 只有 ±0.002 —— 差 60 倍。图 `12_paired_delta` 就是干这个的，
CI 不跨 0 才标为显著，否则打 `n.s.`。

### 4.8 可视化用 matplotlib 而不是 HTML/SVG

| | matplotlib PNG | HTML/SVG |
|---|---|---|
| 远程查看 | Jupyter/VS Code 直接打开 | 要下载，或被 Jupyter 受限 iframe 拦掉脚本 |
| **能否自检** | ✅ 可读图检查 | ❌ 本机无无头浏览器 |
| 放论文 | 直接用，可出 PDF | 要截图 |
| hover 读数 | ❌（用 CSV） | ✅ |

**决定性的是第二条**：渲染出来逐张看，抓到过 4 个真实缺陷（见 §7）。

### 4.9 时间标注：x 轴用秒，顶部挂步数

动作步是 30 fps 的帧，1 步 = 33.3 ms，32 步 = 末步在 **1.033 s**（跨度按 32 个
间隔算是 1.067 s，`annotate.horizon_s` 存的是后者）。

- 底部 x 轴 = **秒**（物理意义明确）
- 顶部 = **动作步**，是同一个轴的单位换算（`secondary_xaxis`），**不是第二个数据尺度**
  —— 双 y 轴始终禁止
- 顶部刻度必须显式给（`{1, n/4, n/2, 3n/4, n}`），否则 matplotlib 会自己选到
  step=0，而第 0 步并不存在
- 多 ckpt 图同理：底部 = 训练步，顶部 = epoch
- mp4 用 **7.5 fps** 导出（`fps / action_video_freq_ratio = 30/4`），即真实播放速率；
  trainer 存的是 8 fps，略快于真实
- 每张图页脚都盖上 ckpt 步数 / epoch / split 与 episode / 样本量 / 去噪步数 / 评测时间戳

---

## 5. 使用方法

### 5.1 首次验证链路（约 2 分钟）

```bash
cd /home/gaomeng/FastWAM && source env.sh
python scripts/offline_eval.py --config configs/eval/smoke.yaml
```

1 个本地 ckpt / 2 样本 / action 模式 / 单卡。数据集与模型构建、preflight、
权重校验、反归一化、落盘全都会走一遍。

### 5.2 全量扫描（39 ckpt × 64 样本 × 8 卡，约 20 分钟）

```bash
python -u scripts/offline_eval.py --config configs/eval/final_A_sweep.yaml
python scripts/visualize_eval.py --sweep eval_offline/final_A_sweep
```

进度看 `eval_offline/final_A_sweep/logs/gpu*.log`。
`python -u` 是为了 launcher 自己的 stdout 也别被缓冲（worker 的已经强制 unbuffered）。

推理挂了但结果还在的话，`--merge-only` 可以只重做合并：

```bash
python scripts/offline_eval.py --config configs/eval/final_A_sweep.yaml --merge-only
```

### 5.3 config 字段

见 `configs/eval/final_A_sweep.yaml`，每一项都有注释。要点：

| 字段 | 说明 |
|---|---|
| `task` | 决定模型结构与数据配置，**必须与训练时一致**（不一致时 preflight 的权重校验会报 mot 键不匹配） |
| `checkpoints` | 路径 / 路径列表 / `{dir, glob, every}` / 以上混写。会按 step 排序去重 |
| `highlight` / `reference` | 前者是 `13_traj_overlay` 要叠加的 ckpt（≤3），后者是 `12_paired_delta` 的基准 |
| `dataset.dataset_dir` | 换一份 LeRobot 数据集 |
| `dataset.episodes` | `null`=沿用 `val_set_proportion` 划分（训练时那 5 个）；`[36,97]`=显式指定；`"all"`=全部 471 个 |
| `dataset.norm_stats` | ⚠️ **必须与训练时同一份**，用错会让指标完全失真 |
| `sampling.per_episode` | 每个 episode 等间隔取 N 个，覆盖更均衡 |
| `sampling.skip_padded` | 跳过 episode 末尾 32 帧的 padding 窗口 |
| `inference.mode` | `joint`=动作+视频；`action`=只动作（快 3.7 倍） |
| `inference.action_crosscheck` | 关掉省约 40% joint 耗时，代价是失去与 trainer 的严格一致 |
| `annotate.steps_per_epoch` | 用来把 step 换算成 epoch。本次训练 = 15,538 |
| `runtime.gpus` | `[0..7]`；多于 1 张且多于 1 个 ckpt 时自动分片 |

命令行可以覆盖任何一项（`--num-samples` / `--mode` / `--episodes` / `--gpus` …），
`--help` 有完整列表。旧式单 ckpt 调用（`--ckpt` + `--out-dir`，不给 `--config`）
行为不变，仍直接写 out-dir。

> `dataset_stats` 会自动使用 `runs/_shared/dataset_stats.json`（若存在），
> 与训练保持一致。**用错统计量会让指标完全失真且不报错。**

### 5.4 出图

```bash
python scripts/visualize_eval.py --sweep eval_offline/final_A_sweep   # 多 ckpt（9 张图）
python scripts/visualize_eval.py DIR                                 # 单 ckpt 目录（5 张）
python scripts/visualize_eval.py DIR_A DIR_B DIR_C                   # 几个独立目录横向对比
python scripts/visualize_eval.py DIR --dpi 200 --pdf
```

目录下有 `sweep.json` 时会**自动识别**为多 ckpt 模式，`--sweep` 只是显式声明。
多 ckpt 模式下的单 ckpt 图取自 `reference` 那一份。

单 ckpt 图：

| 图 | 内容 | 形式选择理由 |
|---|---|---|
| `01_overview` | KPI + 3 个误差分布直方图（带 95% CI） | 单一数字用文字不用图；分布用直方图 |
| `02_per_dim_error` | 逐维平均绝对误差，**最差那维强调** | 14 个类别量级比较；维度名长所以横向条形 |
| `03_error_vs_horizon` | 误差随**预测秒数** + p90 带，顶部挂动作步 | 随时间变化 → 折线；标出末步/首步倍数 |
| `04_video_quality` | PSNR/SSIM 三组分布 | small multiples，各组独立可读 |
| `05_traj_best/worst` | 14 维轨迹 pred vs GT，x 轴用秒 | 14 维塞一张图不可读 → small multiples |

多 ckpt 图：

| 图 | 内容 | 形式选择理由 |
|---|---|---|
| `10_ckpt_sweep` | 每指标一格，x=训练步（顶轴 epoch），均值 + 95% CI 带，最优点强调+直接标注 | 训练步是**连续有序**变量 → 折线。给 39 个 ckpt 各配一个颜色会直接违反配色上限；单色 + emphasis 反而更清楚 |
| `11_ckpt_table` | 排名表（按 action_rmse 升序，逐列加粗最优） | 39 个类别全都带意义 —— 这种情况就该用**表**，不是更多颜色 |
| `12_paired_delta` | 相对 `reference` 的**逐样本成对** Δrmse，均值 + 95% CI，CI 跨 0 标 `n.s.` | 样本完全相同 → 配对差分，见 §4.7 |
| `13_traj_overlay` | `highlight` 的 ≤3 个 ckpt 在**分歧最大**的样本上叠加 14 维轨迹 | all-pairs 形式的 3 色槽上限；GT 是参考不是系列，用 ink 色不占色槽 |

⚠️ `10_ckpt_sweep` 的 CI 带**很可能大面积重叠**。重叠就意味着那个差异不显著，
别读成"越训越好"。要判断两个具体 ckpt 的优劣，看 `12_paired_delta`。

---

## 6. 扩展点（改代码看这里）

### 6.1 加一个新的标量指标

在 `offline_eval.py` 的 `rec = {...}`（约 186 行）里加一项即可 —— 它会自动
进 `metrics.csv`、自动进 `summary.json` 的 `metrics` 聚合（`agg()` 遍历所有 key）。
若要在概览图里显示，再到 `visualize_eval.py::fig_overview` 的 `kpis` 列表加一行。

### 6.2 加一个新的逐维/逐步数组

在 `npz = {...}`（约 196 行）里加。若要聚合进 `summary.json`，仿照
`per_dim` / `per_step` 那两行 `np.stack(...)` 的写法。

### 6.3 换成按 episode 聚合而不是按样本

当前一个"样本"是一个 33 帧窗口，但 `metrics.csv` 已经有 `episode_index` /
`frame_in_episode` 列（`EpisodeIndex` 类按 `episode_data_index` 二分得出），
所以"哪条 episode 最差"直接对 CSV 做 groupby 就行，不用改推理代码。
想每个 episode 取一样多的样本用 `sampling.per_episode`。

### 6.4 换成开环 vs 闭环评测

当前是**开环**：给第 0 帧，一次预测 32 步，与真值比。
要做闭环（模拟真机的 replan），需要一个能"执行动作并返回新观测"的东西 ——
真机数据里没有，只能真机上做（见 `experiments/robotwin/fastwam_policy/deploy_policy.py`
的 `replan_steps` 逻辑）。**离线数据无法做真闭环**，这是原理限制不是实现缺失。

### 6.5 加新图表

`visualize_eval.py` 里每张图是一个独立函数（单 ckpt 的是 `fig_xxx(S, rows, ...)`，
多 ckpt 的是 `fig_xxx(sw, ...)`，`sw` 就是 `sweep.json` 的内容 + `runs`），
在 `draw_single()` / `main()` 里调用。加图就加一个函数 + 一行调用。

配色用文件顶部的常量：
- 分类色 `S1/S2/S3` —— **只用这 3 个**。该调色板只有前 3 槽通过 all-pairs 色盲
  安全门槛；需要更多系列时改成分面、归并到"其他"，或换成"有序变量当 x 轴"，
  **不要自造第 4 个色**
- 有序量用 `SEQ`（单色 light→dark），双向量用 `DIV_LO/DIV_HI` + 中性灰中点
- 文字一律用 ink 色（`INK/INK2/MUTED`），**不要用系列色写字**
- 时间轴用 `step_twin(ax, dt, n_steps)`，多 ckpt 图用 `_xaxis_steps(ax, sw, runs)`
- 页脚用 `footer(fig, run_footer(S))` 或 `footer(fig, sweep_footer(sw))`

### 6.6 换模型变体（IDM / Optional IDM / Joint）

`task` 换成对应的 task 配置即可，脚本用 hydra 实例化模型，不写死变体。
但注意：`fastwam_optional_idm` 只有 `infer_action`（没有 `infer_joint`），
所以那个变体必须用 `mode: action`；Optional IDM 还支持
`action_infer_mode=idm|first_frame`，当前脚本未暴露该参数，需要时加进 `kw`。

---

## 7. 已踩过的坑

### 7.1 致命的（会给出错的数或直接崩）

| 坑 | 现象 | 处置 |
|---|---|---|
| **反归一化漏了 state 的 batch 维** | `_crop` 的 `assert x.ndim == 3` 挂掉，**两次实跑都在第一个样本崩** | `action` 和 `state` 都要 `unsqueeze(0)`。见 §0 / §4.1 |
| `mot.load_state_dict(strict=False)` | 键名不匹配会被**静默忽略**，跑出来是随机初始化的分数，看起来还挺"正常" | `load_and_verify()` 比对键集合，`missing>0` 直接退出。本模型应为 **1649/1649** |
| 数据集列名不匹配 | `robot_video_dataset.__getitem__:283-292` 会静默换随机样本重试 5 次，最后在很深的地方崩 —— 而那已经烧掉 40 多秒建模型 | preflight 拿 `meta/info.json` 的 features 对 `shape_meta` 逐项查，2 秒内报出缺哪些列 |
| 用错 `norm_stats` | 指标完全失真且**不报错** | preflight 检查存在性并打印用的是哪份；换数据集时必须重新确认 |
| text-embed 缓存缺条目 | 推理第一个样本才崩 | preflight 按 `sha256(DEFAULT_PROMPT.format(task=...))` 逐条查，缺了就打印 precompute 命令 |
| episode 末尾的 padding 窗口 | GT action 是复制填充的，误差被**人为压低** | `sampling.skip_padded`，或事后用 `pad_frac` 列过滤 |

### 7.2 性能

| 坑 | 现象 | 处置 |
|---|---|---|
| **多卡 worker 不限 CPU 线程** | 8 进程 × 80 torch 线程，load average 265，每样本 1.55s → **16s（慢 12 倍）** | launcher 自动设 `OMP/MKL_NUM_THREADS` 与 `torch.set_num_threads` = 总核数/卡数 |
| 子进程日志一直是空的 | stdout 是文件时 Python 按块缓冲，跑 20 分钟看不到任何进度 | 子进程强制 `PYTHONUNBUFFERED=1` + `python -u`；launcher 自己也建议用 `python -u` |

### 7.3 API 细节

| 坑 | 现象 | 处置 |
|---|---|---|
| `infer_joint` 的参数名 | 是 `num_video_frames`（必填），不是 `num_frames` | 按 mode 分别构造 kwargs |
| `infer_action` 无视频参数 | 传 `num_video_frames` 会 TypeError | 只在 joint 分支加 |
| `infer_joint` 会多跑一遍 `infer_action` | 默认 `test_action_with_infer_action=True`，joint 约 2× action 成本 | 与 trainer 一致所以默认保留；`action_crosscheck: false` 可关 |

### 7.4 渲染（都是"渲染出来看"才发现的）

| 坑 | 现象 | 处置 |
|---|---|---|
| matplotlib 网格压在数据上 | 条形被切成好几段 | `axes.axisbelow=True`（默认把网格画在 patch 之上） |
| 直方图均值标注压柱子 | 文字与柱重叠 | 均值移进子图标题 |
| **顶部步数轴出现 step 0** | matplotlib 自选刻度，映射回去是 -33ms，而第 0 步不存在 | `step_twin` 显式给 `{1, n/4, n/2, 3n/4, n}` |
| **表头文字互相压** | `action_rmse` 右对齐时伸进左邻列 | 列宽按**英寸**定死再换算，表头与数值都对齐到列右边界，指标名缩写 |
| 成对差分图读数飞出去 | 标签跟着条形末端走，被误差棒拉得七零八落 | 读数放坐标区右侧固定一列（blended transform） |
| 顶部轴标签压住正中刻度 | `动作步` 与中间那个数字重叠 | `labelpad=9` |
| 少数几个 ckpt 时条形像色块 | 槽位高，条形显得又粗又实 | `len<6` 时条形高度收到 0.24 |
| 页脚压住 x 轴标签 | `fig.text` 不被 constrained_layout 计入 | 用 `fig.supxlabel` 当页脚，它会被预留空间 |
| 中文变豆腐块 | matplotlib 不自动加载系统字体 | 主动 `fm.fontManager.addfont()` 注册 `/usr/share/fonts` |
| CJK 字体缺减号字形 | 负号显示异常 | `axes.unicode_minus=False` |
| Markdown 星号原样渲染 | 标题里写 `**同一批**` 真的显示星号 | matplotlib 不解析 markdown，别写 |

---

## 8. 与训练期 eval 的关系

| | `trainer.evaluate()` | `offline_eval.py` |
|---|---|---|
| 何时跑 | 训练中每 `eval_every` 步 | 训练后按需，可一次扫 39 个 ckpt |
| 样本量 | 1/卡 = 8 个 | 任意（默认 64） |
| 样本选取 | **每步换一批随机样本**（`global_step+rank` 播种）→ **ckpt 之间不可比** | 等间隔、确定性 → **所有 ckpt 评同一批**，可做成对差分 |
| 存什么 | 拼接 mp4 + 标量（进 wandb/日志） | **原始 pred/gt 数组** + 标量 + mp4 |
| 能否事后分析 | ❌ `pred_action` 用完就丢 | ✅ |
| 验证集 | 固定是 `val_set_proportion` 划出来的那份 | 可换数据集目录 / 指定 episode / 全量 |
| 指标定义 | 同一套 | 同一套（有意对齐，已实测差 8%） |
| 视频 fps | 8（略快于真实） | **7.5**（真实播放速率） |

所以两者互补：训练期 eval 看**趋势**（尤其 `val_loss` 有没有转升），
离线评测做**定量比较与 checkpoint 选择**。

> 但离线指标终究只是筛选。按 `NEXT_SESSION.md` 原则 5：验证集只有 5 个 episode，
> 且是同一任务的不同 episode，离线 `action_l2` 低不等于真机成功率高。
> **最终必须真机 rollout 定胜负。**
