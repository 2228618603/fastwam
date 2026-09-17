# RTC Python 部署（工程优化版）

支持 aligned RTC step30000 / step50000 **transformer BF16** 权重。
支持同步 closed-loop 和真正异步 `rtc-async`，协议传递 `action_prefix + delay`。

完整修改说明、安装更新、预热、自测、启动、A/B 和回退步骤：

[README_RTC_ENGINEERING_OPTIMIZATIONS.md](README_RTC_ENGINEERING_OPTIMIZATIONS.md)

```bash
# 服务端环境，先自测再正式启动
conda activate gwp_infer
SELF_TEST=1 bash start_rtc30k_python_server.sh
bash start_rtc30k_python_server.sh

# 测试 RTC50k 时改用：
SELF_TEST=1 bash start_rtc50k_python_server.sh
bash start_rtc50k_python_server.sh

# 客户端另一个终端
source /opt/ros/humble/setup.bash
conda activate gwp_client
bash start_rtc_client_log_only.sh
# 确认机械臂状态后再启动运动
SPEED=50 HZ=10 REPLAN=16 RTC_PREFIX=8 DURATION=30 bash start_rtc_client_async.sh
```

SSD 权重路径保持不变：

- `weights/rtc/giga_470_aligned_l6g_r6g_rtc_step30000_transformer_bf16`
- `weights/rtc/giga_470_aligned_l6g_r6g_rtc_step50000_transformer_bf16`
- `weights/base/Wan2.2-TI2V-5B-Diffusers`
- `weights/norm/norm_stats_aligned_l6g_r6g.json`

以上均位于 `/media/geekplus/PortableSSD/zzd_test/` 下。服务端默认完整编译；
首次启动先编译和预热，见到 `serving on ...` 后才连接客户端。

旧 README 中“未实现异步 RTC / EMA 权重”等说明已过时，原文保存在108备份目录
`/home/zzd/project/rtc_engineering_optimization_20260917/before/bundle/README.md`。
