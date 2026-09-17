# fastwam-gm 迁移清单

本目录只放复现 FastWAM RTC 训练和 GWP/RTC 部署工程加速需要参考的代码、小配置、脚本和说明文档；不包含大权重、数据集、训练输出和缓存。

## 目录

- `fastwam_train_rtc/`：来自 `/home/gaomeng/FastWAM` 的 FastWAM 精简代码树。
  - 重点看 `src/fastwam/models/wan22/rtc.py`
  - 训练入口：`scripts/train.py`、`scripts/train_zero1.sh`、`scripts/train_zero2.sh`
  - RTC 配置：`configs/task/agilex_rtc_3cam_384_1e-4.yaml`
  - 说明：`RTC_TRAIN_DESIGN.md`、`DEPLOY_DESIGN.md`
- `gwp_rtc_deploy/`：来自 `/home/zzd/project/robot1_zzd_test_bundle/rtc_python_optimized_20260917.tar.gz` 的部署优化包。
  - 原始包和 sha256 保留在该目录下。
  - 已解包到 `gwp_rtc_deploy/rtc_python/`，方便直接看代码。
  - 重点看 `README_RTC_ENGINEERING_OPTIMIZATIONS.md`。
- `provenance/rtc_engineering_optimization_20260917/`：工程加速补丁、校验日志和同步校验记录。

## 未包含的大文件

- FastWAM 的 `checkpoints/`
- FastWAM 的 `data/`
- FastWAM 的 `runs/`
- FastWAM 的 `eval_offline/`
- 缓存、日志、`__pycache__`、`.git`

如果要真正训练或推理，需要在目标机器单独准备对应数据、norm stats、Wan/FastWAM 权重和 RTC/GWP 权重。
