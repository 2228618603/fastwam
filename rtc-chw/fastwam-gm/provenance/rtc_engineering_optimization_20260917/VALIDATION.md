# 2026-09-17 RTC 工程优化验收

- 108 CPU/loopback：5 项通过，CUDA 项按设计跳过。
- 113 微型随机 MoT FP32 CUDA Graph：6 项通过。
- 113 微型随机 MoT BF16 CUDA Graph：6 项通过。
- CUDA 用例覆盖多次请求和 delay=0/8/10 切换，不加载真实权重。
- 108/113 main 与 RTC bundle 的16个同步文件 SHA256 一致。
- 两个113代码包的16个对应文件均与 manifest 一致。
- eager、action-only compile、full compile 的 shell 参数传递检查通过。
- Python 语法与启动 shell 语法检查通过。
- 113旧主项目文件与旧部署包/构建脚本已备份；机器人端尚未更新。
- 未运行完整 RTC30k 权重稳态测速或真机动作；200ms不是本次测试结果。

相关日志：cpu_tests.log、cuda_tests.log、cuda_bf16_tests.log、shell_checks.log、sync_verification.log。
