# FastWAM AgileX Eval Metrics Explained

本文解释这次 50 条快速评测里的几个核心指标：

```text
count_windows: 50
norm_mse_all: 0.134778
norm_mae_all: 0.252965
raw_mse_all: 0.028360
raw_mae_all: 0.108028
raw_all_joints_mae_deg: 6.9481
mean latency: 0.193s/window
p95 latency: 0.272s/window
```

## 1. 一个 window 是什么

FastWAM 在这个 AgileX 任务里，每条样本预测一个 action chunk：

```text
action shape = [32, 14]
```

含义是：

- `32`: 未来 32 个动作步。
- `14`: 每步动作有 14 个维度。
- 前 6 维: 左臂关节。
- 第 6 维: 左夹爪。
- 第 7-12 维: 右臂关节。
- 第 13 维: 右夹爪。

所以一个 window 里一共有：

```text
32 * 14 = 448
```

个数值会参与整体误差计算。

`count_windows: 50` 表示这次评测一共取了 50 个这样的 action window。

总共参与整体误差统计的动作数值数量是：

```text
50 * 32 * 14 = 22400
```

## 2. norm 指标是什么

`norm_*` 指标是在训练用的归一化动作空间里算的。

训练时，原始动作会经过 normalizer 变成模型更容易学习的尺度。对 z-score 归一化来说，形式类似：

```text
action_norm = (action_raw - mean) / std
```

评测时：

- `gt_norm`: 数据集里的归一化 ground truth action。
- `pred_norm`: 模型直接预测出的归一化 action。

`norm_mse_all` 和 `norm_mae_all` 都是在这两个张量上直接算。

## 3. norm_mse_all

MSE 是 mean squared error，平均平方误差。

公式：

```text
norm_mse_all =
  mean((pred_norm[t, d] - gt_norm[t, d])^2)
```

其中：

- `t` 是 action 步，范围是 `0..31`。
- `d` 是动作维度，范围是 `0..13`。
- `mean` 覆盖所有 window、所有步、所有 14 维动作。

展开到这次 50 条评测，就是：

```text
norm_mse_all =
  sum((pred_norm - gt_norm)^2) / (50 * 32 * 14)
```

这次结果：

```text
norm_mse_all = 0.134778
```

通俗理解：模型在归一化空间里的平均平方误差是 `0.134778`。平方误差会放大大误差，所以 MSE 对明显预测偏差更敏感。

小例子：

```text
gt_norm   = [1.0,  0.0, -1.0]
pred_norm = [0.8,  0.3, -1.4]
diff      = [-0.2, 0.3, -0.4]

mse = ((-0.2)^2 + 0.3^2 + (-0.4)^2) / 3
    = (0.04 + 0.09 + 0.16) / 3
    = 0.0967
```

## 4. norm_mae_all

MAE 是 mean absolute error，平均绝对误差。

公式：

```text
norm_mae_all =
  mean(abs(pred_norm[t, d] - gt_norm[t, d]))
```

展开到这次 50 条评测：

```text
norm_mae_all =
  sum(abs(pred_norm - gt_norm)) / (50 * 32 * 14)
```

这次结果：

```text
norm_mae_all = 0.252965
```

通俗理解：平均每个归一化动作数值，模型和 GT 相差约 `0.253`。

继续用上面的例子：

```text
gt_norm   = [1.0,  0.0, -1.0]
pred_norm = [0.8,  0.3, -1.4]
diff      = [-0.2, 0.3, -0.4]

mae = (abs(-0.2) + abs(0.3) + abs(-0.4)) / 3
    = (0.2 + 0.3 + 0.4) / 3
    = 0.3
```

MSE 和 MAE 的区别：

- MAE 更直观，表示平均差多少。
- MSE 会平方误差，所以更惩罚大偏差。

## 5. raw 指标是什么

`raw_*` 指标是把归一化动作反归一化回原始动作单位后再算。

反归一化形式类似：

```text
action_raw = action_norm * std + mean
```

评测脚本里会对 `gt_norm` 和 `pred_norm` 都做同样的 normalizer backward：

```text
gt_raw   = denormalize(gt_norm)
pred_raw = denormalize(pred_norm)
```

然后在 `gt_raw` 和 `pred_raw` 上计算 raw 指标。

raw 指标更接近真实机器人动作尺度，但要注意：

- 关节维度通常是弧度。
- 夹爪维度可能是开合位置或归一化位置。
- `raw_mse_all` / `raw_mae_all` 把关节和夹爪 14 维一起平均，所以不是纯关节角误差。

## 6. raw_mse_all

公式：

```text
raw_mse_all =
  mean((pred_raw[t, d] - gt_raw[t, d])^2)
```

展开到这次 50 条评测：

```text
raw_mse_all =
  sum((pred_raw - gt_raw)^2) / (50 * 32 * 14)
```

这次结果：

```text
raw_mse_all = 0.028360
```

通俗理解：在原始动作单位上，所有动作维度混在一起后的平均平方误差是 `0.028360`。

小例子：

```text
gt_raw   = [0.10, 0.50, -0.20]
pred_raw = [0.20, 0.40, -0.50]
diff     = [0.10, -0.10, -0.30]

mse = (0.10^2 + (-0.10)^2 + (-0.30)^2) / 3
    = (0.01 + 0.01 + 0.09) / 3
    = 0.0367
```

## 7. raw_mae_all

公式：

```text
raw_mae_all =
  mean(abs(pred_raw[t, d] - gt_raw[t, d]))
```

展开到这次 50 条评测：

```text
raw_mae_all =
  sum(abs(pred_raw - gt_raw)) / (50 * 32 * 14)
```

这次结果：

```text
raw_mae_all = 0.108028
```

通俗理解：回到原始动作单位后，平均每个动作维度的绝对误差约 `0.108`。

注意：这个值混合了关节和夹爪，所以如果想看机器人手臂动作准不准，更建议看下面的关节角指标。

## 8. raw_all_joints_mae_deg

这个指标只看 12 个关节维度，不看 2 个夹爪维度。

关节维度是：

```text
left_arm_joints  = 0..5
right_arm_joints = 7..12
all_joints       = 0..5 + 7..12
```

先在原始单位里计算关节绝对误差。原始关节单位是弧度：

```text
joint_mae_rad =
  mean(abs(pred_raw[t, joint_dim] - gt_raw[t, joint_dim]))
```

然后把弧度换算成角度：

```text
joint_mae_deg = joint_mae_rad * 180 / pi
```

这次结果：

```text
raw_all_joints_mae_deg = 6.9481
```

通俗理解：平均到每个 window、每个未来动作步、每个关节维度，模型预测的关节角和 GT 相差约 `6.95 度`。

这是比 `raw_mae_all` 更直观的机械臂动作误差指标。

小例子：

假设只看 3 个关节，误差分别是：

```text
diff_rad = [0.10, -0.20, 0.05]
```

先算弧度 MAE：

```text
mae_rad = (abs(0.10) + abs(-0.20) + abs(0.05)) / 3
        = 0.1167 rad
```

换算成角度：

```text
mae_deg = 0.1167 * 180 / pi
        = 6.69 deg
```

所以 `raw_all_joints_mae_deg` 可以直接理解成平均关节角度偏差。

## 9. mean latency

`mean latency` 是平均每个 window 的推理耗时。

公式：

```text
mean_latency =
  sum(latency_i) / count_windows
```

这次结果：

```text
mean latency = 0.193s/window
```

通俗理解：模型平均每次预测一个 `[32, 14]` action chunk 需要 `0.193` 秒。

注意这里不是单个动作步的耗时，而是一次性预测 32 步动作的耗时。

如果粗略换算到每个动作步：

```text
0.193 / 32 = 0.0060 s/step
```

但真实部署时通常不会简单按这个数线性理解，因为策略可能按 chunk 方式滚动执行。

## 10. p95 latency

`p95 latency` 是 95 分位延迟。

含义是：把 50 个 window 的推理耗时从小到大排序，取靠近 95% 位置的值。

这次结果：

```text
p95 latency = 0.272s/window
```

通俗理解：这 50 次推理里，大约 95% 的 window 推理耗时不超过 `0.272` 秒。

为什么看 p95：

- `mean latency` 看平均速度。
- `p95 latency` 看尾部慢请求。
- 部署时，尾部延迟通常比平均延迟更影响体感稳定性。

小例子：

```text
latency = [0.10, 0.11, 0.12, 0.13, 0.50]
```

平均值：

```text
mean = (0.10 + 0.11 + 0.12 + 0.13 + 0.50) / 5
     = 0.192
```

这里有一次明显慢的 `0.50s`。p95 会更接近尾部耗时，因此能暴露偶发慢推理。

## 11. 这几个指标怎么一起看

建议按这个顺序理解：

```text
1. count_windows
   先看样本数够不够。50 条只是快速测试，不是最终结论。

2. norm_mse_all / norm_mae_all
   看模型在训练空间里的整体误差，适合模型间横向比较。

3. raw_mae_all / raw_mse_all
   看回到原始动作尺度后的整体误差，但混合了关节和夹爪。

4. raw_all_joints_mae_deg
   看机械臂关节预测偏差，最直观。

5. mean latency / p95 latency
   看推理速度和尾部延迟。
```

这次快速评测可以概括为：

```text
50 个 window 上，模型平均关节角误差约 6.95 度；
每次预测 32 步动作平均耗时约 0.193 秒；
95% 的推理在 0.272 秒以内完成。
```
