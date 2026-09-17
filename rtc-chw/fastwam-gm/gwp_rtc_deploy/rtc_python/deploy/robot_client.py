"""
Giga World Policy 机器人客户端 (Agilex Cobot Magic 双臂, Piper SDK + ROS2).

运行在工控机上。无 torch/CUDA 依赖 — 与 deploy/robot_server.py (GPU 服务器端)
通过 websocket + msgpack 通讯。协议沿用此前已验证的 FastWAM 部署形式，
核心安全模型保留；旧参考实现已随 legacy 部署目录归档。

=============================================================================
                          SAFETY MODEL - READ FIRST
=============================================================================
本脚本拒绝移动机器人, 除非你显式逐级 opt-in, 不存在一步跳到闭环自主的开关。

    --mode log-only     (默认) 只推理并打印动作, 什么都不发给机械臂。
                        先用它核对单位/量程/方向。
    --mode replay       回放数据集 episode 的已录动作 (不做模型推理)。
                        这是验证单位换算和爪量程的经典方法 — 你已知动作应该
                        是什么样子。
    --mode step         以低速度执行 LIMITED 数量的模型动作 (--max-steps,
                        默认 1) 然后停止。
    --mode closed-loop  连续推理/执行。必须 --i-am-watching 且显式
                        --max-duration。

所有会动的模式还额外要求:
    --confirm-safety    你确认: 急停在手边, 工作区清空, 机械臂包络内无人。

运动模式始终生效的防护:
  * 关节幅值限制          (--joint-limit-rad, 默认 3.14)
  * 轨迹平滑度            (--max-joint-delta-rad, 默认 0.20) — 限制相邻两条
                          命令之间的步长, 即捕获模型输出的不连续
  * 跟踪误差              (--max-tracking-error-rad, 默认 0.50) — 限制命令 vs
                          实测, 即捕获机械臂跟不上。解决方法是提高
                          --speed-percent / 降低 --action-hz, 不是放宽阈值
  * 爪钳制到硬件量程      (0 .. 70000 raw)
  * 速度上限              (--speed-percent, 默认 70, 硬上限 100)
  * 陈旧观测 watchdog     (--obs-timeout-s, 默认 0.5)
  * 连续违规即中止        (不盲目重试)

单位约定 (与 lingbot_deploy_client.py 在同一台机器上验证过):
  * 关节: 弧度 -> Piper raw = round(rad * 1000 * 180/pi)
  * 爪: 模型输出 0~1 开合分数 -> 米 = 分数 * --gripper-scale (默认 0.105)
       -> Piper raw = round(m * 1e6), 然后钳制到 [0, 70000]。
  ⚠️ 0.105 这个比例是从这台机械臂上的前一次部署继承的, 原代码里仍标注
     "待重新验证"。在 --mode replay 里验证后再在推理模式中信任它。

模型输出布局 (14D, 绝对关节角, 弧度):
    action[0:6]  左臂关节      action[6]   左爪 (0~1)
    action[7:13] 右臂关节      action[13]  右爪 (0~1)

发给服务器的 state (14D, 与动作序一致):
    state[0:6] 左关节, state[6] 左爪 (0~1),
    state[7:13] 右关节, state[13] 右爪 (0~1)

典型开机流程
------------
 1) python robot_client.py --server ws://<gpu-host>:8000 --mode log-only
 2) python robot_client.py --mode replay --episode-parquet <...>.parquet \
        --confirm-safety --speed-percent 10
 3) python robot_client.py --server ... --mode step --max-steps 1 \
        --confirm-safety --speed-percent 10
 4) python robot_client.py --server ... --mode closed-loop \
        --confirm-safety --i-am-watching --max-duration 30 --speed-percent 10

起始姿态
--------
模型输出绝对关节角, 每条训练 episode 都从任务合适的工作姿态开始, 从不从
机械零位开始。从零位推理等于让模型外推它从未见过的状态, 所以加 --goto-start
先把双臂插值到数据集的起始姿态 (内置为全部 471 条训练 episode 首帧的均值;
--start-pose-parquet 可用某条 episode 覆盖)。该接近过程本身也是运动, 即使
与 --mode log-only 组合也需要 --confirm-safety。
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------- optional deps
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    import cv_bridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object  # type: ignore

try:
    from piper_sdk import C_PiperInterface_V2
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False

# ------------------------------------------------------------------- constants
# 单位换算 - 与同一台机器上已验证的关节空间客户端 (lingbot_deploy_client.py)
# 完全相同。不要"简化"这些数字。
RAD_TO_DEG_001 = 1000.0 * 180.0 / math.pi   # 弧度 -> Piper 0.001 度
METER_TO_GRIPPER_RAW = 1_000_000.0          # 米  -> Piper 0.001 mm
GRIPPER_RAW_MIN, GRIPPER_RAW_MAX = 0, 70000  # 硬件行程极限 - 永远不要超出

ACTION_HZ_DEFAULT = 30.0  # 数据集 fps == 部署控制频率

CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
# ROS2 topic -> 服务器相机 key。
# ⚠️ 默认 topic 名取自参考部署 (FastWAM), 必须按你们工控机的实际 topic 修改:
#   ros2 topic list | grep image  看真实 topic 名, 用 --topic-top / --topic-left
#   / --topic-right 或直接改这个字典。
DEFAULT_TOPIC_MAP = {
    "/camera/top/camera/color/image_raw": "top",
    "/camera/left/camera/color/image_raw": "left_wrist",
    "/camera/right/camera/color/image_raw": "right_wrist",
}

SPEED_PERCENT_HARD_MAX = 100  # 真机验证后放开 (2026-08-05)

# 默认起始姿态, 14D 动作布局
# ([0:6] 左关节, [6] 左爪, [7:13] 右关节, [13] 右爪)。
#
# 由 agilex_empty_the_box_all_470 全部 471 条 episode 首帧的均值算出。
# 每条训练 episode 都从任务合适的工作姿态 (双臂伸向箱子) 开始, 从不从机械
# 零位开始, 所以推理必须从分布内的状态起步 — 否则模型在从未见过的状态上外推。
#
# 各 episode 间的关节分布 (std): 大多数关节 0.05–0.19 rad, 即起始姿态相当
# 一致, 单个代表值即可。离零位最远的是 j5 关节 0.851 rad (~49 度)。
START_POSE_ACTION14 = np.array([
    -0.2187, +0.1383, -0.4933, -0.2940, +0.8513, +0.2763, +0.5086,   # left:  j1..j6, gripper
    +0.3524, +0.1806, -0.4801, +0.0619, +0.7892, -0.0104, +0.5402,   # right: j1..j6, gripper
], dtype=np.float32)


def _warn_gripper_scale(gripper_scale: float) -> None:
    """把已知的 scale 与硬件极限之间的矛盾摆到台面上, 而不是藏起来。

    ``--gripper-scale`` 默认 0.105 是从这台机械臂上前一次部署原样继承的 (那里
    就已经标注"待重新验证")。完全张开 (模型输出 1.0) 时换算为 105000 raw,
    超过硬件行程极限 70000。send_command 里的钳制保证硬件安全, 但也意味着
    所有 ~0.667 以上的模型输出都塌缩成同一个"完全张开"命令, 爪的顶段分辨率
    丢失。刻意保持原样 (与前次部署一致); 在真机上用 --mode replay 验证后再改。
    """
    raw_at_full = round(gripper_scale * METER_TO_GRIPPER_RAW)
    if raw_at_full <= GRIPPER_RAW_MAX:
        return
    saturates_at = GRIPPER_RAW_MAX / (gripper_scale * METER_TO_GRIPPER_RAW)
    print("\n" + "!" * 78)
    print("[GRIPPER SCALE WARNING - 已知未解决问题, 按前次部署保留]")
    print(f"  --gripper-scale {gripper_scale} 使完全张开 1.0 映射到 {raw_at_full} raw,")
    print(f"  但硬件行程极限是 {GRIPPER_RAW_MAX} raw。")
    print(f"  => 命令会被钳制 (硬件安全), 但所有 ~{saturates_at:.3f} 以上的模型输出")
    print(f"     都会变成同一个'完全张开'命令, 顶段爪分辨率丢失。")
    print(f"  自洽的值应该是 {GRIPPER_RAW_MAX / METER_TO_GRIPPER_RAW:.3f},")
    print(f"  但这是物理参数 - 先在真机上 --mode replay 验证 (命令 0.0 -> 1.0 并")
    print(f"  测量实际爪行程) 再改。")
    print("!" * 78 + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=None, help="ws://host:port of robot_server.py (replay 不需要)。")
    p.add_argument("--mode", default="log-only",
                   choices=["log-only", "replay", "step", "closed-loop", "rtc-async", "goto-start"],
                   help="见 SAFETY MODEL。默认 log-only 什么都不发。"
                        " goto-start: 仅归位到起始姿态后退出, 不需要 --server。")

    # 显式安全 opt-in
    p.add_argument("--confirm-safety", action="store_true",
                   help="任何运动都需要。确认急停可及且工作区清空。")
    p.add_argument("--i-am-watching", action="store_true",
                   help="closed-loop/rtc-async 需要。确认有人全程目视监督。")

    # 运动限制
    p.add_argument("--speed-percent", type=int, default=70,
                   help=f"Piper 速度档 (1..{SPEED_PERCENT_HARD_MAX})。默认 70。")
    p.add_argument("--joint-limit-rad", type=float, default=3.14,
                   help="拒绝任何 |角度| 超过此值的关节命令。")
    p.add_argument("--max-joint-delta-rad", type=float, default=0.2,
                   help="轨迹平滑度上限: 相邻两条命令间某关节移动超过此值即拒绝。"
                        "真实 30Hz 训练数据 p99.9 = 0.103 rad, 0.2 留了余量同时仍能抓住野输出。")
    p.add_argument("--max-tracking-error-rad", type=float, default=0.5,
                   help="跟踪误差上限: 命令关节与实测位置差超过此值即中止, 即机械臂跟不上"
                        "命令轨迹。解决方法是提高 --speed-percent 或降低 --action-hz, 不是放宽阈值。")
    p.add_argument("--gripper-scale", type=float, default=0.105,
                   help="模型 0~1 爪分数 -> 米。先 REPLAY 验证!。")
    p.add_argument("--max-steps", type=int, default=1, help="--mode step: 执行多少条动作。")
    p.add_argument("--max-duration", type=float, default=None,
                   help="--mode closed-loop: 硬性墙钟停止 (秒)。必填。")
    p.add_argument("--replan-steps", type=int, default=8,
                   help="每条推理 chunk 先执行这么多步再重新观测。")
    p.add_argument("--rtc-prefix-steps", type=int, default=8,
                   help="--mode rtc-async: 发起下一次异步推理时承诺/前缀的动作步数。"
                        "建议 6..10，且不要超过 RTC 训练的 max_delay-1=11。")
    p.add_argument("--action-hz", type=float, default=ACTION_HZ_DEFAULT)
    p.add_argument("--obs-timeout-s", type=float, default=0.5,
                   help="相机/状态数据比这个年龄还旧就中止。")
    p.add_argument("--max-violations", type=int, default=1,
                   help="达到多少次安全限制违规后中止 (默认: 第一次就中止)。")

    # 硬件
    # can0 = 左臂, can1 = 右臂 — 与这台机器人上已有的客户端一致
    p.add_argument("--can-left", default="can0")
    p.add_argument("--can-right", default="can1")
    p.add_argument("--enable-timeout-s", type=float, default=20.0,
                   help="等待每只机械臂 EnablePiper 成功的超时时间。右臂偶发慢启时可设 30。")
    p.add_argument("--no-ros2", action="store_true",
                   help="跳过 ROS2 相机采集 (log-only/replay 无相机调试)。")

    # 相机 topic 覆盖 (默认见 DEFAULT_TOPIC_MAP, 按工控机实际修改)
    p.add_argument("--topic-top", default=None)
    p.add_argument("--topic-left", default=None)
    p.add_argument("--topic-right", default=None)

    # replay
    p.add_argument("--episode-parquet", default=None, help="--mode replay: 数据集 episode parquet。")
    p.add_argument("--replay-max-frames", type=int, default=60, help="--mode replay: 最多执行的帧数。")

    # 推理前移动到数据集起始姿态
    #
    # 模型输出绝对关节角, 训练数据的 episode 全部从任务合适的工作姿态开始
    # (双臂伸向箱子), 从不从机械零位开始。从零位推理等于让模型外推它从未见过
    # 的状态, 所以先把双臂带到录制的起始姿态是部署的常规部分, 不是特殊模式。
    p.add_argument("--goto-start", action="store_true",
                   help="推理前先把双臂插值到数据集的起始姿态 (内置为 471 条"
                        "episode 首帧均值; --start-pose-parquet 可覆盖)。这会动机械臂, "
                        "即使 log-only 模式也需要 --confirm-safety。")
    p.add_argument("--start-pose-parquet", default=None,
                   help="可选: 用这条 episode parquet 的首帧作为起始姿态, 替代内置均值。")
    p.add_argument("--goto-step-rad", type=float, default=0.02,
                   help="接近起始姿态时每步最大关节变化。默认 0.02 rad 远低于真实 30Hz 运动的 0.035 rad p95。")
    p.add_argument("--goto-settle-s", type=float, default=1.0,
                   help="到达起始姿态后、推理开始前的停顿。")

    p.add_argument("--instruction", default=None, help="覆盖发送给服务器的任务文本。")
    p.add_argument("--log-file", default=None, help="把动作日志追加到这个文件 (JSON lines)。")
    return p.parse_args()


# ============================================================ observation side
class ObservationBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._images: Dict[str, np.ndarray] = {}
        self._stamps: Dict[str, float] = {}

    def update_image(self, cam_key: str, image_rgb_hwc: np.ndarray):
        with self._lock:
            self._images[cam_key] = image_rgb_hwc
            self._stamps[cam_key] = time.time()

    def snapshot(self) -> Tuple[Dict[str, np.ndarray], float]:
        """返回 (images, 最旧一帧的年龄秒)。"""
        with self._lock:
            if len(self._images) < len(CAMERA_KEYS):
                missing = [k for k in CAMERA_KEYS if k not in self._images]
                raise RuntimeError(f"cameras not ready: missing {missing}")
            imgs = {k: v.copy() for k, v in self._images.items()}
            oldest = min(self._stamps[k] for k in imgs)
        return imgs, time.time() - oldest


class ROS2ObservationCollector(Node):
    def __init__(self, buf: ObservationBuffer, topic_map: Dict[str, str]):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 not available")
        super().__init__("gwp_obs_collector")
        self.buf = buf
        self.bridge = cv_bridge.CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=2)
        for topic, key in topic_map.items():
            self.create_subscription(Image, topic, lambda m, k=key: self._on_image(m, k), qos)
        self.get_logger().info(f"subscribed: {list(topic_map)}")

    def _on_image(self, msg, cam_key: str):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.buf.update_image(cam_key, img)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"image convert failed {cam_key}: {e}")


# ================================================================ arm wrapper
class PiperArmJoint:
    """关节空间 Piper 封装, 复刻已验证的实现。"""

    def __init__(self, can_name: str, name: str):
        if not HAS_PIPER:
            raise RuntimeError("piper_sdk not installed")
        self.name = name
        self.can_name = can_name
        self.piper = C_PiperInterface_V2(can_name)
        self._enabled = False

    def connect(self):
        print(f"  [{self.name}] connecting {self.can_name} ...")
        self.piper.ConnectPort()
        time.sleep(0.2)

    def enable(self, timeout: float = 5.0):
        print(f"  [{self.name}] enabling ...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.piper.EnablePiper():
                self._enabled = True
                print(f"  [{self.name}] enabled")
                return
            time.sleep(0.01)
        raise RuntimeError(f"[{self.name}] enable timeout")

    def read_state(self) -> Tuple[np.ndarray, float]:
        """-> (6 个关节角, 弧度; 爪开合, 米)

        注意 API: 关节和爪来自两次不同的 SDK 调用。
        GetArmJointMsgs() 没有 gripper_state 属性 — 爪在
        GetArmGripperMsgs().gripper_state.grippers_angle 上。
        刻意不用 try/except 包住: 静默回退到全零会把全零本体向量喂给模型,
        看着正常实则更危险, 不如大声失败。
        """
        if not self._enabled:
            return np.zeros(6, dtype=np.float32), 0.0
        js = self.piper.GetArmJointMsgs().joint_state
        joints = np.array([
            js.joint_1 / RAD_TO_DEG_001, js.joint_2 / RAD_TO_DEG_001,
            js.joint_3 / RAD_TO_DEG_001, js.joint_4 / RAD_TO_DEG_001,
            js.joint_5 / RAD_TO_DEG_001, js.joint_6 / RAD_TO_DEG_001,
        ], dtype=np.float32)
        gripper_m = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / METER_TO_GRIPPER_RAW
        return joints, gripper_m

    def send_command(self, joints_rad: np.ndarray, gripper_m: float, speed_percent: int):
        if not self._enabled:
            raise RuntimeError(f"[{self.name}] not enabled; refusing to send")
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=int(speed_percent), is_mit_mode=0x00)
        self.piper.JointCtrl(*[round(float(j) * RAD_TO_DEG_001) for j in joints_rad])
        raw = int(round(float(gripper_m) * METER_TO_GRIPPER_RAW))
        raw = max(GRIPPER_RAW_MIN, min(GRIPPER_RAW_MAX, raw))
        self.piper.GripperCtrl(raw, 1000, 0x01, 0x00)

    def hold(self):
        """用当前实测姿态作为命令来停止运动。"""
        if not self._enabled:
            return
        try:
            joints, gripper_m = self.read_state()
            self.send_command(joints, gripper_m, speed_percent=5)
        except Exception as e:  # noqa: BLE001
            print(f"  [{self.name}] hold failed: {e}")

    def disconnect(self):
        print(f"  [{self.name}] disconnect")


# ============================================================== safety checker
class SafetyGuard:
    """两个不同的检查, 针对两种不同的失败模式。

    原本把"命令 vs 实测"合并成一个 delta 检查, 产生过误导性拒绝: 以 30Hz 流式
    下发绝对位置目标而机械臂以 speed_percent=15 运动时, 机械臂会落后于命令, 即使
    模型轨迹完全平滑, 命令-实测的差距也会增长。(实测: 相邻命令只差 ~0.05 rad,
    而命令-实测差距已增长到 0.23 rad。)

      1. 轨迹平滑度 (命令 vs 上一条命令) — 判断"模型是否发出野跳变"的正确信号。
         保持紧: 真实 30Hz 训练数据 p99.9 步长 0.103 rad。
      2. 跟踪误差 (命令 vs 实测) — 检测"机械臂跟不上"。需要更松的阈值, 正确
         的解决方法是提高 --speed-percent 或降低 --action-hz, 不是中止。分开
         报告, 以免再次混淆两种原因。
    """

    def __init__(self, joint_limit_rad: float, max_joint_delta_rad: float,
                 max_tracking_error_rad: float, max_violations: int):
        self.joint_limit = float(joint_limit_rad)
        self.max_delta = float(max_joint_delta_rad)
        self.max_tracking_error = float(max_tracking_error_rad)
        self.max_violations = int(max_violations)
        self.violations = 0
        self._prev_cmd: Optional[np.ndarray] = None
        self.worst_tracking_error = 0.0
        self._tracking_warned = False

    def check(self, action14: np.ndarray, current_state14: Optional[np.ndarray]) -> Optional[str]:
        """动作必须被拒绝时返回错误串, 否则返回 None。"""
        a = np.asarray(action14, dtype=np.float32).reshape(-1)
        if a.shape[0] != 14:
            return f"action must be 14D, got {a.shape[0]}D"
        if not np.all(np.isfinite(a)):
            return f"action contains non-finite values: {a.tolist()}"

        left_j, right_j = a[0:6], a[7:13]
        lg, rg = float(a[6]), float(a[13])

        # --- 绝对关节极限
        for label, joints in (("left", left_j), ("right", right_j)):
            over = np.abs(joints) > self.joint_limit
            if np.any(over):
                return (f"{label} joint magnitude exceeds {self.joint_limit} rad at "
                        f"{np.where(over)[0].tolist()}: {joints.tolist()}")
        for label, g in (("left", lg), ("right", rg)):
            if not (-0.5 <= g <= 1.5):  # 爪命令下发前会 clip 到 0~1; 这里只拦截明显野值
                return f"{label} gripper fraction far out of 0~1 range: {g}"

        # --- 检查 1: 轨迹平滑度 (这条命令 vs 上一条命令)
        if self._prev_cmd is not None:
            p = self._prev_cmd
            for label, cmd, prev in (("left", left_j, p[0:6]), ("right", right_j, p[7:13])):
                step = np.abs(cmd - prev)
                if np.any(step > self.max_delta):
                    idx = np.where(step > self.max_delta)[0].tolist()
                    return (f"{label} trajectory jump exceeds {self.max_delta} rad at {idx}: "
                            f"step={np.round(step, 4).tolist()} "
                            f"(model emitted a discontinuous command)")

        # --- 检查 2: 跟踪误差 (命令 vs 实测) — 提示性, 更松的阈值
        if current_state14 is not None:
            cur = np.asarray(current_state14, dtype=np.float32).reshape(-1)
            # state 布局: [0:6] 左关节, [6] 左爪, [7:13] 右关节, [13] 右爪
            worst = 0.0
            for label, cmd, now in (("left", left_j, cur[0:6]), ("right", right_j, cur[7:13])):
                err = np.abs(cmd - now)
                worst = max(worst, float(err.max()))
                if np.any(err > self.max_tracking_error):
                    idx = np.where(err > self.max_tracking_error)[0].tolist()
                    return (f"{label} TRACKING ERROR exceeds {self.max_tracking_error} rad at {idx}: "
                            f"err={np.round(err, 4).tolist()} — the arm is falling behind the "
                            f"commanded trajectory. Raise --speed-percent or lower --action-hz "
                            f"rather than loosening this bound.")
            self.worst_tracking_error = max(self.worst_tracking_error, worst)
            # 远在中止阈值之前就警告一次, 让滞后尽早可见。
            if not self._tracking_warned and worst > 0.5 * self.max_tracking_error:
                print(f"  [SAFETY] note: tracking error reached {worst:.4f} rad "
                      f"(abort at {self.max_tracking_error}); the arm is lagging behind commands. "
                      f"Consider a higher --speed-percent or lower --action-hz.")
                self._tracking_warned = True

        self._prev_cmd = a.copy()
        return None

    def reset_trajectory(self):
        """忘掉上一条命令 (动作队列被丢弃时调用)。"""
        self._prev_cmd = None

    def register_violation(self, reason: str) -> bool:
        """-> True 表示必须中止。"""
        self.violations += 1
        print(f"  [SAFETY] REJECTED ({self.violations}/{self.max_violations}): {reason}")
        return self.violations >= self.max_violations


# =============================================================== model client
class ServerClient:
    def __init__(self, url: str):
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect
        msgpack_numpy.patch()
        self._msgpack = msgpack
        self._request_lock = threading.Lock()
        print(f"[client] connecting to {url} ...")
        self._ws = connect(url, open_timeout=30, max_size=None)
        info = self._request({"ping": True})
        print(f"[client] server info: {json.dumps(info.get('info', {}), indent=2, ensure_ascii=False)}")
        self.info = info.get("info", {})
        self.last_infer_s = 0.0
        self.last_infer_wall_s = 0.0
        self.wire_targets = self.info.get("wire_target_sizes")
        if self.wire_targets:
            try:
                import PIL.Image  # 不能覆盖 sensor_msgs.msg.Image 的 ROS2 类型。
            except ImportError:
                self.wire_targets = None
                print("[client] pillow unavailable; sending full-size images (pip install pillow)")
            else:
                print(f"[client] 上行预处理已开: {self.wire_targets}")

    def _shrink(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """移植 gaomeng 的 PIL resize/crop；与服务端训练预处理逐像素一致。"""
        from PIL import Image as PILImage
        out = {}
        for key, arr in images.items():
            target = self.wire_targets.get(key)
            if target is None:
                out[key] = arr
                continue
            w, h = map(int, target)
            img = PILImage.fromarray(arr)
            sw, sh = img.size
            if (sw, sh) == (w, h):
                out[key] = arr
                continue
            if h / sh < w / sw:
                nw, nh = w, int(round(w / sw * sh))
            else:
                nw, nh = int(round(h / sh * sw)), h
            # cv2.INTER_LINEAR 和 PIL BILINEAR 的像素不同，不可替换。
            img = img.resize((nw, nh), PILImage.Resampling.BILINEAR)
            x, y = (nw - w) // 2, (nh - h) // 2
            out[key] = np.asarray(img.crop((x, y, x + w, y + h)))
        return out

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._request_lock:
            # 原两行串联 pack/send 和 recv/unpack，无法分辨网络与模型耗时：
            # self._ws.send(self._msgpack.packb(payload, use_bin_type=True))
            # resp = self._msgpack.unpackb(self._ws.recv(), raw=False)
            t0 = time.perf_counter()
            blob = self._msgpack.packb(payload, use_bin_type=True)
            t1 = time.perf_counter()
            self._ws.send(blob)
            t2 = time.perf_counter()
            raw = self._ws.recv()
            t3 = time.perf_counter()
            resp = self._msgpack.unpackb(raw, raw=False)
            t4 = time.perf_counter()
            self.last_timing = {"pack_ms": (t1 - t0) * 1000, "send_ms": (t2 - t1) * 1000,
                                "recv_ms": (t3 - t2) * 1000, "unpack_ms": (t4 - t3) * 1000,
                                "payload_mib": len(blob) / 1024**2}
            if not resp.get("ok"):
                raise RuntimeError(f"server error: {resp.get('error')}")
            return resp

    def reset(self):
        self._request({"reset": True})

    def infer(self, images: Dict[str, np.ndarray], state14: np.ndarray,
              instruction: Optional[str], action_prefix: Optional[np.ndarray] = None,
              delay: int = 0) -> np.ndarray:
        t0 = time.perf_counter()
        # 原 payload 直接携带全尺寸 images；模型最终只用 320x384 的合成图。
        # payload = {"obs": images, "state": np.asarray(state14, dtype=np.float32)}
        if self.wire_targets:
            images = self._shrink(images)
        shrink_ms = (time.perf_counter() - t0) * 1000
        payload: Dict[str, Any] = {"obs": images, "state": np.asarray(state14, dtype=np.float32)}
        if instruction:
            payload["instruction"] = instruction
        delay = int(delay or 0)
        if delay > 0:
            if action_prefix is None:
                raise ValueError("action_prefix is required when delay > 0")
            payload["delay"] = delay
            payload["action_prefix"] = np.asarray(action_prefix, dtype=np.float32)[:delay]
        resp = self._request(payload)
        self.last_infer_s = float(resp.get("infer_s", 0.0))
        self.last_infer_wall_s = time.perf_counter() - t0
        self.last_timing["shrink_ms"] = shrink_ms
        print(f"[timing] request={self.last_infer_wall_s * 1000:.1f}ms "
              f"server={self.last_infer_s * 1000:.1f}ms "
              + " ".join(f"{k}={v:.2f}" for k, v in self.last_timing.items()))
        return np.asarray(resp["action"], dtype=np.float32)

    def close(self):
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


# ==================================================================== helpers
def assemble_state14(left_joints: np.ndarray, left_gripper_m: float,
                     right_joints: np.ndarray, right_gripper_m: float,
                     gripper_scale: float) -> np.ndarray:
    """按 action-compatible 顺序拼 14D 本体向量 (L6, Lg, R6, Rg)。"""
    lg = float(np.clip(left_gripper_m / gripper_scale, 0.0, 1.0))
    rg = float(np.clip(right_gripper_m / gripper_scale, 0.0, 1.0))
    # Legacy wrong order kept for reference:
    #   [left_joints, right_joints, left_gripper, right_gripper]
    # It matched the old broken training state but not the action layout.
    return np.concatenate([
        np.asarray(left_joints, dtype=np.float32).reshape(6),
        np.array([lg], dtype=np.float32),
        np.asarray(right_joints, dtype=np.float32).reshape(6),
        np.array([rg], dtype=np.float32),
    ])


def split_action14(action14: np.ndarray, gripper_scale: float):
    """-> (left_joints, left_gripper_m, right_joints, right_gripper_m)"""
    a = np.asarray(action14, dtype=np.float32).reshape(-1)
    left_j, right_j = a[0:6], a[7:13]
    lg_m = float(np.clip(a[6], 0.0, 1.0)) * gripper_scale
    rg_m = float(np.clip(a[13], 0.0, 1.0)) * gripper_scale
    return left_j, lg_m, right_j, rg_m


def describe_action(action14: np.ndarray, gripper_scale: float) -> str:
    lj, lg, rj, rg = split_action14(action14, gripper_scale)
    return (f"L_joints={np.round(lj, 4).tolist()} L_grip={lg:.5f}m | "
            f"R_joints={np.round(rj, 4).tolist()} R_grip={rg:.5f}m")


class ActionLogger:
    def __init__(self, path: Optional[str]):
        self._f = open(path, "a") if path else None

    def log(self, record: Dict[str, Any]):
        if self._f:
            self._f.write(json.dumps(record, default=float) + "\n")
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


# ==================================================== start-pose approach
def load_start_pose(parquet_path: str) -> np.ndarray:
    """读一条 episode 的第一条动作 -> 14D 动作布局向量。"""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    col = "action" if "action" in table.column_names else "actions"
    if col not in table.column_names:
        raise ValueError(f"no action column in {parquet_path}; columns={table.column_names}")
    first = np.asarray(table[col].to_pylist()[0], dtype=np.float32).reshape(-1)
    if first.shape[0] != 14:
        raise ValueError(f"expected 14D action, got {first.shape[0]}D")
    return first


def goto_start_pose(arms, target14: np.ndarray, args, logger: "ActionLogger") -> bool:
    """从当前实测姿态插值到 target14。成功返回 True。

    用小步长 (--goto-step-rad) 移动而不是一次性发大跳, 使双臂平滑行进。
    每一步命令天然有界, 任何关节超过 --joint-limit-rad 就中止整个过程。
    """
    tgt_lj, tgt_lg_m, tgt_rj, tgt_rg_m = split_action14(target14, args.gripper_scale)

    cur_lj, cur_lg_m = arms[0].read_state()
    cur_rj, cur_rg_m = arms[1].read_state()

    for label, joints in (("left target", tgt_lj), ("right target", tgt_rj)):
        over = np.abs(joints) > args.joint_limit_rad
        if np.any(over):
            print(f"[ABORT] {label} exceeds --joint-limit-rad at {np.where(over)[0].tolist()}: {joints.tolist()}")
            return False

    max_travel = float(max(np.abs(tgt_lj - cur_lj).max(), np.abs(tgt_rj - cur_rj).max()))
    n_steps = max(1, int(np.ceil(max_travel / max(args.goto_step_rad, 1e-6))))

    print("\n" + "-" * 78)
    print("APPROACHING DATASET START POSE — THE ROBOT WILL MOVE")
    print(f"  pose source      : {args.start_pose_parquet or 'built-in mean of 471 episodes'}")
    print(f"  current  L joints: {np.round(cur_lj, 4).tolist()}")
    print(f"  target   L joints: {np.round(tgt_lj, 4).tolist()}  grip={tgt_lg_m:.5f}m")
    print(f"  current  R joints: {np.round(cur_rj, 4).tolist()}")
    print(f"  target   R joints: {np.round(tgt_rj, 4).tolist()}  grip={tgt_rg_m:.5f}m")
    print(f"  largest joint travel = {max_travel:.4f} rad ({np.degrees(max_travel):.1f} deg)")
    print(f"  -> {n_steps} interpolation steps of <= {args.goto_step_rad} rad, speed={args.speed_percent}")
    print("-" * 78)
    for i in (3, 2, 1):
        print(f"  approaching in {i}s ... (Ctrl-C to abort)")
        time.sleep(1)

    dt = 1.0 / args.action_hz
    try:
        for k in range(1, n_steps + 1):
            f = k / n_steps
            lj = cur_lj + (tgt_lj - cur_lj) * f
            rj = cur_rj + (tgt_rj - cur_rj) * f
            lg = cur_lg_m + (tgt_lg_m - cur_lg_m) * f
            rg = cur_rg_m + (tgt_rg_m - cur_rg_m) * f
            arms[0].send_command(lj, lg, args.speed_percent)
            arms[1].send_command(rj, rg, args.speed_percent)
            if k % 10 == 0 or k == n_steps:
                print(f"  [{k}/{n_steps}] L={np.round(lj,4).tolist()} R={np.round(rj,4).tolist()}")
            logger.log({"t": time.time(), "phase": "goto_start", "k": k, "n": n_steps,
                        "left": lj.tolist(), "right": rj.tolist(),
                        "left_grip_m": lg, "right_grip_m": rg})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[ABORT] approach interrupted by user; holding current pose")
        for a in arms:
            a.hold()
        return False

    if args.goto_settle_s > 0:
        print(f"  settling for {args.goto_settle_s}s ...")
        time.sleep(args.goto_settle_s)

    got_lj, got_lg = arms[0].read_state()
    got_rj, got_rg = arms[1].read_state()
    err_l = float(np.abs(got_lj - tgt_lj).max())
    err_r = float(np.abs(got_rj - tgt_rj).max())
    print(f"  reached: L err={err_l:.4f} rad, R err={err_r:.4f} rad")
    print(f"  L now={np.round(got_lj,4).tolist()} grip={got_lg:.5f}m")
    print(f"  R now={np.round(got_rj,4).tolist()} grip={got_rg:.5f}m")
    print("-" * 78 + "\n")
    return True


# ======================================================================= modes
def run_log_only(args, arms, buf: Optional[ObservationBuffer], client: ServerClient,
                 logger: ActionLogger) -> int:
    """反复推理并打印动作。什么都不发给机械臂。"""
    print("\n" + "=" * 78)
    print("MODE: log-only — NO commands will be sent to the robot.")
    print("=" * 78 + "\n")

    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad,
                        args.max_tracking_error_rad, max_violations=10**9)
    n = 0
    try:
        while True:
            if arms:
                lj, lg_m = arms[0].read_state()
                rj, rg_m = arms[1].read_state()
            else:
                lj = rj = np.zeros(6, dtype=np.float32)
                lg_m = rg_m = 0.0
            state14 = assemble_state14(lj, lg_m, rj, rg_m, args.gripper_scale)

            if buf is not None:
                images, age = buf.snapshot()
                if age > args.obs_timeout_s:
                    print(f"[warn] observation is stale ({age:.3f}s > {args.obs_timeout_s}s)")
            else:
                print("[info] --no-ros2: using synthetic gray images (inference smoke check only)")
                images = {k: np.full((480, 640, 3), 128, dtype=np.uint8) for k in CAMERA_KEYS}

            chunk = client.infer(images, state14, args.instruction)
            n += 1
            print(f"\n--- inference #{n}: chunk {chunk.shape} ---")
            print(f"  current state: L={np.round(lj,4).tolist()} Lg={lg_m:.5f}m | "
                  f"R={np.round(rj,4).tolist()} Rg={rg_m:.5f}m")
            for i in range(min(3, chunk.shape[0])):
                verdict = guard.check(chunk[i], state14)
                tag = "OK" if verdict is None else f"WOULD-REJECT: {verdict}"
                print(f"  [{i}] {describe_action(chunk[i], args.gripper_scale)}  -> {tag}")
            if chunk.shape[0] > 3:
                print(f"  ... ({chunk.shape[0] - 3} more actions in this chunk)")
            logger.log({"t": time.time(), "mode": "log-only", "n": n,
                        "state14": state14.tolist(), "chunk_first3": chunk[:3].tolist()})
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[log-only] stopped by user")
    return 0


def run_replay(args, arms, logger: ActionLogger) -> int:
    """回放数据集 episode 的已录动作 — 无模型, 已知正确的运动。"""
    if not args.episode_parquet:
        print("[error] --mode replay requires --episode-parquet")
        return 2
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[error] pyarrow required for replay")
        return 2

    table = pq.read_table(args.episode_parquet)
    col = "action" if "action" in table.column_names else "actions"
    if col not in table.column_names:
        print(f"[error] no action column in {args.episode_parquet}; columns={table.column_names}")
        return 2
    actions = np.stack(table[col].to_pylist()).astype(np.float32)
    n_frames = min(args.replay_max_frames, actions.shape[0])
    print(f"\nMODE: replay — {n_frames}/{actions.shape[0]} recorded frames from {args.episode_parquet}")
    print("This validates unit conversions and gripper scaling using KNOWN-GOOD data.\n")

    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad,
                        args.max_tracking_error_rad, args.max_violations)
    dt = 1.0 / args.action_hz
    executed = 0
    try:
        for i in range(n_frames):
            lj, lg_m = arms[0].read_state()
            rj, rg_m = arms[1].read_state()
            state14 = assemble_state14(lj, lg_m, rj, rg_m, args.gripper_scale)
            reason = guard.check(actions[i], state14)
            if reason is not None:
                if guard.register_violation(reason):
                    print("[replay] ABORTING due to safety violation")
                    break
                continue
            l_j, l_g, r_j, r_g = split_action14(actions[i], args.gripper_scale)
            arms[0].send_command(l_j, l_g, args.speed_percent)
            arms[1].send_command(r_j, r_g, args.speed_percent)
            executed += 1
            if i % 10 == 0:
                print(f"  [{i}] {describe_action(actions[i], args.gripper_scale)}")
            logger.log({"t": time.time(), "mode": "replay", "i": i,
                        "action14": actions[i].tolist()})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    finally:
        for a in arms:
            a.hold()
    print(f"[replay] executed {executed} frames")
    return 0


def run_motion(args, arms, buf: ObservationBuffer, client: ServerClient,
               logger: ActionLogger) -> int:
    """--mode step (有界) / --mode closed-loop (限时)。"""
    is_step = args.mode == "step"
    budget = args.max_steps if is_step else None
    deadline = None if is_step else time.time() + args.max_duration

    print("\n" + "=" * 78)
    print(f"MODE: {args.mode} — THE ROBOT WILL MOVE.")
    print(f"  speed_percent      = {args.speed_percent} (hard max {SPEED_PERCENT_HARD_MAX})")
    print(f"  joint limit        = ±{args.joint_limit_rad} rad")
    print(f"  max joint delta    = {args.max_joint_delta_rad} rad per action")
    print(f"  gripper scale      = {args.gripper_scale} m per unit fraction")
    if is_step:
        print(f"  will execute       = {budget} action(s) then STOP")
    else:
        print(f"  will run for       = {args.max_duration}s then STOP")
    print("=" * 78 + "\n")
    for i in (3, 2, 1):
        print(f"  starting in {i}s ... (Ctrl-C to abort)")
        time.sleep(1)

    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad,
                        args.max_tracking_error_rad, args.max_violations)
    dt = 1.0 / args.action_hz
    pending: deque = deque()
    executed = 0
    client.reset()

    try:
        while True:
            if budget is not None and executed >= budget:
                print(f"[{args.mode}] reached --max-steps={budget}; stopping")
                break
            if deadline is not None and time.time() >= deadline:
                print(f"[{args.mode}] reached --max-duration; stopping")
                break

            lj, lg_m = arms[0].read_state()
            rj, rg_m = arms[1].read_state()
            state14 = assemble_state14(lj, lg_m, rj, rg_m, args.gripper_scale)

            if not pending:
                images, age = buf.snapshot()
                if age > args.obs_timeout_s:
                    print(f"[ABORT] watchdog: observation stale ({age:.3f}s > {args.obs_timeout_s}s)")
                    break
                chunk = client.infer(images, state14, args.instruction)
                take = min(args.replan_steps, chunk.shape[0])
                if budget is not None:
                    take = min(take, budget - executed)
                for k in range(take):
                    pending.append(chunk[k])
                print(f"[infer] queued {take} action(s) from chunk {chunk.shape}")

            if not pending:
                print("[warn] nothing queued; stopping")
                break

            action = pending.popleft()
            reason = guard.check(action, state14)
            if reason is not None:
                if guard.register_violation(reason):
                    print("[ABORT] safety violation limit reached")
                    break
                pending.clear()  # 不要盲目继续一条可疑的 chunk
                guard.reset_trajectory()  # 下一条命令不再与这条连续
                continue

            l_j, l_g, r_j, r_g = split_action14(action, args.gripper_scale)
            arms[0].send_command(l_j, l_g, args.speed_percent)
            arms[1].send_command(r_j, r_g, args.speed_percent)
            executed += 1
            print(f"  [exec {executed}] {describe_action(action, args.gripper_scale)}")
            logger.log({"t": time.time(), "mode": args.mode, "n": executed,
                        "state14": state14.tolist(), "action14": action.tolist()})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[ABORT] interrupted by user")
    finally:
        print("[safety] holding current pose")
        for a in arms:
            a.hold()
    print(f"[{args.mode}] executed {executed} action(s)")
    print(f"[{args.mode}] worst tracking error observed: {guard.worst_tracking_error:.4f} rad "
          f"(bound {args.max_tracking_error_rad}) — if this is large, the arm is struggling to "
          f"follow {args.action_hz:.0f}Hz commands at speed_percent={args.speed_percent}")
    return 0


def run_motion_rtc_async(args, arms, buf: ObservationBuffer, client: ServerClient,
                         logger: ActionLogger) -> int:
    """Asynchronous RTC closed-loop: infer next chunk while executing committed prefix."""
    deadline = time.time() + args.max_duration
    prefix_steps = int(args.rtc_prefix_steps)
    if prefix_steps < 1 or prefix_steps >= 48:
        print(f"[REFUSED] --rtc-prefix-steps must be in [1,47], got {prefix_steps}")
        return 2
    if prefix_steps > 11:
        print(f"[warn] --rtc-prefix-steps={prefix_steps} exceeds trained delay range 0..11; use with caution")
    if args.replan_steps <= prefix_steps:
        print(f"[REFUSED] --replan-steps must be > --rtc-prefix-steps "
              f"for rtc-async, got replan={args.replan_steps}, prefix={prefix_steps}")
        return 2

    print("\n" + "=" * 78)
    print("MODE: rtc-async — THE ROBOT WILL MOVE.")
    print(f"  speed_percent      = {args.speed_percent}")
    print(f"  action_hz          = {args.action_hz}")
    print(f"  replan_steps       = {args.replan_steps}")
    print(f"  rtc_prefix_steps   = {prefix_steps}")
    print(f"  max joint delta    = {args.max_joint_delta_rad} rad per action")
    print(f"  max duration       = {args.max_duration}s")
    print("=" * 78 + "\n")
    for i in (3, 2, 1):
        print(f"  starting in {i}s ... (Ctrl-C to abort)")
        time.sleep(1)

    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad,
                        args.max_tracking_error_rad, args.max_violations)
    dt = 1.0 / args.action_hz
    pending: deque = deque()
    executed = 0
    client.reset()

    def read_state14():
        lj, lg_m = arms[0].read_state()
        rj, rg_m = arms[1].read_state()
        return lj, lg_m, rj, rg_m, assemble_state14(lj, lg_m, rj, rg_m, args.gripper_scale)

    def snapshot_or_raise():
        images, age = buf.snapshot()
        if age > args.obs_timeout_s:
            raise RuntimeError(f"observation stale ({age:.3f}s > {args.obs_timeout_s}s)")
        return images

    executor = ThreadPoolExecutor(max_workers=1)
    future: Optional[Future] = None
    future_delay = 0
    future_submitted_at = 0.0

    try:
        lj, lg_m, rj, rg_m, state14 = read_state14()
        first_chunk = client.infer(snapshot_or_raise(), state14, args.instruction)
        first_take = min(args.replan_steps, first_chunk.shape[0])
        for k in range(first_take):
            pending.append(first_chunk[k])
        print(f"[infer sync] queued {first_take} action(s) from initial chunk {first_chunk.shape}")

        while True:
            if time.time() >= deadline:
                print(f"[rtc-async] reached --max-duration={args.max_duration}; stopping")
                break

            lj, lg_m, rj, rg_m, state14 = read_state14()

            # Submit exactly when the remaining committed actions equal the RTC delay.
            # If we submit earlier with delay=RTC_PREFIX but execute more than RTC_PREFIX
            # old actions before consuming the result, the new chunk is time-misaligned.
            if future is None and len(pending) == prefix_steps:
                try:
                    images = snapshot_or_raise()
                    prefix = np.stack(list(pending)).astype(np.float32)
                    request_state = state14.copy()
                    future_delay = len(prefix)
                    future_submitted_at = time.time()
                    future = executor.submit(client.infer, images, request_state, args.instruction, prefix, future_delay)
                    print(f"[rtc async] submitted next inference delay={future_delay}, pending={len(pending)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] failed to submit async inference: {exc}")
                    future = None

            if not pending:
                if future is None:
                    print("[warn] no pending action and no async future; falling back to sync inference")
                    chunk = client.infer(snapshot_or_raise(), state14, args.instruction)
                    future_delay = 0
                else:
                    if not future.done():
                        print("[rtc async] waiting for next chunk ...")
                    chunk = future.result()
                    # 原日志把等待承诺队列执行完的时间也叫推理时间，可能误报 2s 推理。
                    # print(f"[rtc async] received chunk {chunk.shape} in {time.time() - future_submitted_at:.3f}s")
                    print(f"[rtc async] received chunk {chunk.shape}; request={client.last_infer_wall_s:.3f}s, "
                          f"consume_after_submit={time.time() - future_submitted_at:.3f}s "
                          "(includes committed action execution)")
                    future = None
                start = min(future_delay, chunk.shape[0])
                take = min(args.replan_steps, chunk.shape[0] - start)
                for k in range(start, start + take):
                    pending.append(chunk[k])
                print(f"[rtc async] queued {take} action(s), skipped prefix={start}")
                if not pending:
                    print("[ABORT] async chunk produced no executable suffix")
                    break
                continue

            action = pending.popleft()
            reason = guard.check(action, state14)
            if reason is not None:
                if guard.register_violation(reason):
                    print("[ABORT] safety violation limit reached")
                    break
                pending.clear()
                guard.reset_trajectory()
                continue

            l_j, l_g, r_j, r_g = split_action14(action, args.gripper_scale)
            arms[0].send_command(l_j, l_g, args.speed_percent)
            arms[1].send_command(r_j, r_g, args.speed_percent)
            executed += 1
            print(f"  [exec {executed}] {describe_action(action, args.gripper_scale)}")
            logger.log({"t": time.time(), "mode": "rtc-async", "n": executed,
                        "state14": state14.tolist(), "action14": action.tolist(),
                        "pending": len(pending), "future_active": future is not None})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[ABORT] interrupted by user")
    except Exception as exc:  # noqa: BLE001
        print(f"[ABORT] rtc-async failed: {exc}")
    finally:
        if future is not None:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print("[safety] holding current pose")
        for a in arms:
            a.hold()
    print(f"[rtc-async] executed {executed} action(s)")
    print(f"[rtc-async] worst tracking error observed: {guard.worst_tracking_error:.4f} rad "
          f"(bound {args.max_tracking_error_rad})")
    return 0


# ======================================================================== main
def main() -> int:
    args = _parse_args()
    motion_modes = {"replay", "step", "closed-loop", "rtc-async"}
    # --goto-start 会真的移动机械臂, 所以即使 log-only 也算运动。
    needs_motion = args.mode in motion_modes or args.goto_start

    # ---- 在碰任何东西之前先做门槛检查
    if needs_motion and not args.confirm_safety:
        what = f"--mode {args.mode}" if args.mode in motion_modes else "--goto-start"
        print(f"[REFUSED] {what} moves the robot but --confirm-safety was not given.\n"
              f"          Verify: e-stop within reach, workspace clear, no one in the envelope.\n"
              f"          Then re-run with --confirm-safety.")
        return 2
    if args.goto_start and not args.start_pose_parquet:
        print(f"[info] --goto-start using the built-in mean starting pose "
              f"(mean first frame of all 471 training episodes).")
    if args.mode in ("closed-loop", "rtc-async"):
        if not args.i_am_watching:
            print("[REFUSED] closed-loop requires --i-am-watching (a human actively supervising).")
            return 2
        if not args.max_duration:
            print("[REFUSED] closed-loop requires an explicit --max-duration (seconds).")
            return 2
    if args.speed_percent < 1 or args.speed_percent > SPEED_PERCENT_HARD_MAX:
        print(f"[REFUSED] --speed-percent must be within 1..{SPEED_PERCENT_HARD_MAX} during bring-up "
              f"(got {args.speed_percent}). Edit SPEED_PERCENT_HARD_MAX deliberately if you truly need more.")
        return 2
    if args.mode not in ("replay", "goto-start") and not args.server:
        print("[REFUSED] --server is required for inference modes.")
        return 2

    # 相机 topic 覆盖
    topic_map = dict(DEFAULT_TOPIC_MAP)
    overrides = {"top": args.topic_top, "left_wrist": args.topic_left, "right_wrist": args.topic_right}
    for key, t in overrides.items():
        if t:
            # 找到映射到该 key 的默认 topic 并替换; 找不到就新增
            replaced = False
            for default_topic, mapped in list(topic_map.items()):
                if mapped == key:
                    topic_map.pop(default_topic)
                    topic_map[t] = key
                    replaced = True
                    break
            if not replaced:
                topic_map[t] = key
    print(f"[info] camera topic map: {json.dumps(topic_map)}")

    logger = ActionLogger(args.log_file)
    _warn_gripper_scale(args.gripper_scale)
    arms = []
    client: Optional[ServerClient] = None
    ros_node = None
    buf: Optional[ObservationBuffer] = None
    executor_thread = None

    try:
        # ---- 机械臂: 运动需要; log-only 读状态也需要
        want_arms = needs_motion or not args.no_ros2
        if want_arms:
            if not HAS_PIPER:
                if needs_motion:
                    print("[REFUSED] piper_sdk not installed but motion was requested.")
                    return 2
                print("[warn] piper_sdk unavailable; log-only will use zero state")
            else:
                left = PiperArmJoint(args.can_left, "left")
                right = PiperArmJoint(args.can_right, "right")
                for a in (left, right):
                    a.connect()
                if needs_motion:
                    for a in (left, right):
                        a.enable(timeout=args.enable_timeout_s)
                else:
                    # log-only: 仍然 enable 以读真实关节状态, 但此模式绝不调用 send_command
                    for a in (left, right):
                        try:
                            a.enable(timeout=args.enable_timeout_s)
                        except Exception as e:  # noqa: BLE001
                            print(f"[warn] {a.name} enable failed ({e}); state will read as zeros")
                arms = [left, right]

        # ---- 相机
        if not args.no_ros2 and args.mode != "replay":
            if not HAS_ROS2:
                print("[REFUSED] ROS2 unavailable; pass --no-ros2 only for offline smoke checks.")
                return 2
            buf = ObservationBuffer()
            rclpy.init()
            ros_node = ROS2ObservationCollector(buf, topic_map)
            executor_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
            executor_thread.start()
            print("[ros2] waiting for all camera topics ...")
            t0 = time.time()
            while time.time() - t0 < 30:
                try:
                    buf.snapshot()
                    print("[ros2] all cameras ready")
                    break
                except RuntimeError:
                    time.sleep(0.2)
            else:
                print("[REFUSED] cameras did not become ready within 30s")
                return 2

        # ---- server
        if args.mode != "replay":
            client = ServerClient(args.server)
            srv_state = client.info.get("state_layout")
            if srv_state:
                print(f"[client] server expects state: {srv_state}")

        # ---- 可选: 把双臂带到数据集的已录起始姿态。
        # 在相机/server 就绪之后做, 这样失败不会把双臂停在接近途中;
        # 在任何推理之前做, 让模型看到训练分布内的状态而不是零位。
        if args.goto_start:
            if not arms:
                print("[REFUSED] --goto-start needs the arms connected")
                return 2
            target = (load_start_pose(args.start_pose_parquet)
                      if args.start_pose_parquet else START_POSE_ACTION14)
            if not goto_start_pose(arms, target, args, logger):
                print("[ABORT] start-pose approach failed; not proceeding to inference")
                return 1

        # ---- 分发
        if args.mode == "log-only":
            return run_log_only(args, arms, buf, client, logger)
        if args.mode == "goto-start":
            if not arms:
                print("[REFUSED] goto-start needs the arms connected")
                return 2
            target = (load_start_pose(args.start_pose_parquet)
                      if args.start_pose_parquet else START_POSE_ACTION14)
            ok = goto_start_pose(arms, target, args, logger)
            print("[goto-start] done" if ok else "[goto-start] FAILED")
            return 0 if ok else 1
        if args.mode == "replay":
            if not arms:
                print("[REFUSED] replay needs the arms connected")
                return 2
            return run_replay(args, arms, logger)
        if args.mode == "rtc-async":
            return run_motion_rtc_async(args, arms, buf, client, logger)
        return run_motion(args, arms, buf, client, logger)

    finally:
        if client:
            client.close()
        for a in arms:
            a.disconnect()
        if ros_node is not None:
            ros_node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
