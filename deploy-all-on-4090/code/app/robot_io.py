from __future__ import annotations

import math
import threading
import time
from typing import Dict, Tuple

import numpy as np

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

try:
    from .layout import CAMERA_KEYS
except ImportError:
    from layout import CAMERA_KEYS  # type: ignore

RAD_TO_DEG_001 = 1000.0 * 180.0 / math.pi
METER_TO_GRIPPER_RAW = 1_000_000.0
GRIPPER_RAW_MIN, GRIPPER_RAW_MAX = 0, 70000

DEFAULT_TOPIC_MAP = {
    "/camera/top/camera/color/image_raw": "top",
    "/camera/left/camera/color/image_raw": "left_wrist",
    "/camera/right/camera/color/image_raw": "right_wrist",
}


class ObservationBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._images: Dict[str, np.ndarray] = {}
        self._stamps: Dict[str, float] = {}

    def update_image(self, cam_key: str, image_rgb_hwc: np.ndarray) -> None:
        with self._lock:
            self._images[cam_key] = image_rgb_hwc
            self._stamps[cam_key] = time.time()

    def snapshot(self) -> Tuple[Dict[str, np.ndarray], float]:
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
        super().__init__("fastwam_local_obs_collector")
        self.buf = buf
        self.bridge = cv_bridge.CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
        for topic, key in topic_map.items():
            self.create_subscription(Image, topic, lambda m, k=key: self._on_image(m, k), qos)
        self.get_logger().info(f"subscribed: {list(topic_map)}")

    def _on_image(self, msg, cam_key: str) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.buf.update_image(cam_key, img)
        except Exception as exc:
            self.get_logger().error(f"image convert failed {cam_key}: {exc}")


class PiperArmJoint:
    def __init__(self, can_name: str, name: str):
        if not HAS_PIPER:
            raise RuntimeError("piper_sdk not installed")
        self.name = name
        self.can_name = can_name
        self.piper = C_PiperInterface_V2(can_name)
        self._enabled = False

    def connect(self) -> None:
        print(f"  [{self.name}] connecting {self.can_name} ...")
        self.piper.ConnectPort()
        time.sleep(0.2)

    def enable(self, timeout: float = 5.0) -> None:
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
        if not self._enabled:
            return np.zeros(6, dtype=np.float32), 0.0
        js = self.piper.GetArmJointMsgs().joint_state
        joints = np.array(
            [
                js.joint_1 / RAD_TO_DEG_001,
                js.joint_2 / RAD_TO_DEG_001,
                js.joint_3 / RAD_TO_DEG_001,
                js.joint_4 / RAD_TO_DEG_001,
                js.joint_5 / RAD_TO_DEG_001,
                js.joint_6 / RAD_TO_DEG_001,
            ],
            dtype=np.float32,
        )
        gripper_m = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / METER_TO_GRIPPER_RAW
        return joints, gripper_m

    def send_command(self, joints_rad: np.ndarray, gripper_m: float, speed_percent: int) -> None:
        if not self._enabled:
            raise RuntimeError(f"[{self.name}] not enabled; refusing to send")
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=int(speed_percent), is_mit_mode=0x00)
        self.piper.JointCtrl(*[round(float(j) * RAD_TO_DEG_001) for j in joints_rad])
        raw = int(round(float(gripper_m) * METER_TO_GRIPPER_RAW))
        raw = max(GRIPPER_RAW_MIN, min(GRIPPER_RAW_MAX, raw))
        self.piper.GripperCtrl(raw, 1000, 0x01, 0x00)

    def hold(self) -> None:
        if not self._enabled:
            return
        try:
            joints, gripper_m = self.read_state()
            self.send_command(joints, gripper_m, speed_percent=5)
        except Exception as exc:
            print(f"  [{self.name}] hold failed: {exc}")

    def disconnect(self) -> None:
        print(f"  [{self.name}] disconnect")

