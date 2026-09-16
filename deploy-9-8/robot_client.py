"""
FastWAM Piper client for the Agilex Cobot Magic dual-arm robot.

Runs on the ROBOT computer (ROS2 + Piper SDK). Needs no torch/CUDA - it talks to
``deploy-9-8/robot_server.py`` over websocket + msgpack.

=============================================================================
                          SAFETY MODEL - READ FIRST
=============================================================================
This script REFUSES to move the robot unless you explicitly opt in, one stage at
a time. There is no single flag that jumps straight to closed-loop autonomy.

    --mode log-only     (DEFAULT) Infer and print actions. NOTHING is sent to
                        the arms. Use this to sanity-check units, ranges and
                        direction before any motion.
    --mode replay       Replay recorded actions from a dataset episode (no model
                        inference). This is the classic way to validate that the
                        unit conversions and gripper scaling are right, because
                        you already know what the motion should look like.
    --mode step         Execute a LIMITED number of model actions
                        (--max-steps, default 1) at low speed, then stop.
    --mode closed-loop  Continuous inference/execution. Requires --i-am-watching
                        and an explicit --max-duration.

Every motion-capable mode additionally requires:
    --confirm-safety    You assert: e-stop within reach, workspace clear,
                        nobody in the robot's envelope.

Guards that are always active in motion modes:
  * joint magnitude limit          (--joint-limit-rad, default 3.14)
  * trajectory smoothness          (--max-joint-delta-rad, default 0.20) — bounds
                                   the step between CONSECUTIVE COMMANDS, i.e.
                                   catches a discontinuous model output
  * tracking error                 (--max-tracking-error-rad, default 0.50) —
                                   bounds COMMAND vs MEASURED, i.e. catches the
                                   arm falling behind. Fix by raising
                                   --speed-percent / lowering --action-hz.
  * gripper clamp to hardware range (0 .. 70000 raw, from the verified client)
  * speed cap                      (--speed-percent, default 10, hard max 30)
  * stale-observation watchdog     (--obs-timeout-s, default 0.5)
  * consecutive-violation abort    (aborts instead of retrying blindly)

Unit conventions (verified previously on THIS robot via lingbot_deploy_client.py):
  * joints: radians -> Piper raw = round(rad * 1000 * 180/pi)
  * gripper: model outputs a 0~1 opening fraction -> meters via
    ``--gripper-scale`` (default 0.105) -> Piper raw = round(m * 1e6),
    then clamped to [0, 70000].
  ⚠️ The 0.105 scale is inherited from a previous deployment on this arm and is
  still marked "to be re-verified" in that code. VERIFY IT IN --mode replay
  BEFORE trusting it in inference modes.

Model output layout (14D, absolute joint angles in radians):
    action[0:6]  left arm joints      action[6]     left gripper  (0~1)
    action[7:13] right arm joints     action[13]    right gripper (0~1)

State sent to the server (14D, TRAINING order - different from action order!):
    state[0:6] left joints, state[6:12] right joints,
    state[12] left gripper (0~1), state[13] right gripper (0~1)

Typical bring-up sequence
-------------------------
 1) python deploy-9-8/robot_client.py --server ws://<gpu-host>:8000 --mode log-only
 2) python deploy-9-8/robot_client.py --mode replay --episode-parquet <...>.parquet \
        --confirm-safety --speed-percent 10
 3) python deploy-9-8/robot_client.py --server ... --mode step --max-steps 1 \
        --confirm-safety --speed-percent 10
 4) python deploy-9-8/robot_client.py --server ... --mode closed-loop \
        --confirm-safety --i-am-watching --max-duration 30 --speed-percent 10

Starting pose
-------------
The model emits ABSOLUTE joint angles and every training episode begins from a
task-appropriate working pose, never from the mechanical zero pose. Running
inference from zero asks the model to extrapolate from a state it never saw, so
add --goto-start to interpolate the arms onto the dataset's starting pose first
(built into this file as the mean first frame of all 471 training episodes; pass
--start-pose-parquet to use a specific episode instead). That approach is itself
motion, so it needs --confirm-safety even when combined with --mode log-only.
"""
from __future__ import annotations

import argparse
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
# Unit conversions - identical to the previously verified joint-space client
# (lingbot_deploy_client.py) on this same robot. Do not "simplify" these.
RAD_TO_DEG_001 = 1000.0 * 180.0 / math.pi   # radians -> Piper 0.001 deg
METER_TO_GRIPPER_RAW = 1_000_000.0          # meters  -> Piper 0.001 mm
GRIPPER_RAW_MIN, GRIPPER_RAW_MAX = 0, 70000  # hardware travel limit - never exceed

ACTION_HZ_DEFAULT = 30.0  # dataset fps == deployment control rate

CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
# ROS2 topic -> server camera key. Topic names taken from the existing verified
# client on this robot; override with --topic-* if your setup differs.
DEFAULT_TOPIC_MAP = {
    "/camera/top/camera/color/image_raw": "top",
    "/camera/left/camera/color/image_raw": "left_wrist",
    "/camera/right/camera/color/image_raw": "right_wrist",
}

SPEED_PERCENT_HARD_MAX = 30  # refuse anything faster during bring-up

# Default starting pose, in the 14D ACTION layout
# ([0:6] left joints, [6] left gripper, [7:13] right joints, [13] right gripper).
#
# Computed as the mean first frame across ALL 471 episodes of
# agilex_empty_the_box_all_542_0711. Every training episode begins from a
# task-appropriate working pose (arms reaching toward the boxes) and never from
# the mechanical zero pose, so inference must start from something inside this
# distribution — otherwise the model is extrapolating from a state it never saw.
#
# Per-joint spread across episodes (std): most joints 0.05–0.19 rad, i.e. the
# starting pose is fairly consistent, so a single representative value is fine.
# Largest travel from zero is 0.851 rad (~49 deg) on the j5 joints.
START_POSE_ACTION14 = np.array([
    -0.2187, +0.1383, -0.4933, -0.2940, +0.8513, +0.2763, +0.5086,   # left:  j1..j6, gripper
    +0.3524, +0.1806, -0.4801, +0.0619, +0.7892, -0.0104, +0.5402,   # right: j1..j6, gripper
], dtype=np.float32)


def _warn_gripper_scale(gripper_scale: float) -> None:
    """Surface the known scale-vs-hardware-limit inconsistency instead of hiding it.

    ``--gripper-scale`` default 0.105 is inherited verbatim from the previous
    deployment on this arm (where it was already flagged "to be re-verified").
    At full opening (model outputs 1.0) it converts to 105000 raw, which exceeds
    the hardware travel limit of 70000. The clamp in ``send_command`` keeps the
    hardware safe, but it also means every model output above ~0.667 collapses
    onto the same fully-open command, so gripper resolution is lost across the
    top third of the range. Kept as-is deliberately (matching the prior
    deployment); verify on the real arm in --mode replay before changing.
    """
    raw_at_full = round(gripper_scale * METER_TO_GRIPPER_RAW)
    if raw_at_full <= GRIPPER_RAW_MAX:
        return
    saturates_at = GRIPPER_RAW_MAX / (gripper_scale * METER_TO_GRIPPER_RAW)
    print("\n" + "!" * 78)
    print("[GRIPPER SCALE WARNING - known unresolved issue, kept as previously deployed]")
    print(f"  --gripper-scale {gripper_scale} maps a full 1.0 opening to {raw_at_full} raw,")
    print(f"  but the hardware travel limit is {GRIPPER_RAW_MAX} raw.")
    print(f"  => Commands are CLAMPED (hardware stays safe), but every model output")
    print(f"     above ~{saturates_at:.3f} becomes the same fully-open command;")
    print(f"     gripper resolution is lost above that point.")
    print(f"  A self-consistent value would be {GRIPPER_RAW_MAX / METER_TO_GRIPPER_RAW:.3f},")
    print(f"  but this is a PHYSICAL parameter - confirm it on the real arm in")
    print(f"  --mode replay (command 0.0 -> 1.0 and measure the actual jaw travel)")
    print(f"  before changing it.")
    print("!" * 78 + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=None, help="ws://host:port of deploy-9-8/robot_server.py (not needed for replay).")
    p.add_argument("--mode", default="log-only",
                   choices=["log-only", "replay", "step", "closed-loop"],
                   help="See SAFETY MODEL in the module docstring. Default log-only sends nothing.")

    # explicit safety opt-ins
    p.add_argument("--confirm-safety", action="store_true",
                   help="Required for any motion. Asserts e-stop reachable and workspace clear.")
    p.add_argument("--i-am-watching", action="store_true",
                   help="Required for closed-loop. Asserts a human is actively supervising.")

    # motion limits
    p.add_argument("--speed-percent", type=int, default=10,
                   help=f"Piper speed rate (1..{SPEED_PERCENT_HARD_MAX}). Default 10.")
    p.add_argument("--joint-limit-rad", type=float, default=3.14,
                   help="Reject any commanded joint whose |angle| exceeds this.")
    p.add_argument("--max-joint-delta-rad", type=float, default=0.2,
                   help="TRAJECTORY SMOOTHNESS bound: reject if a joint moves more than this "
                        "between CONSECUTIVE COMMANDS (i.e. the model emitted a discontinuity). "
                        "Real 30Hz training data has p99.9 = 0.103 rad, so 0.2 leaves headroom "
                        "while still catching wild outputs.")
    p.add_argument("--max-tracking-error-rad", type=float, default=0.5,
                   help="TRACKING ERROR bound: abort if a commanded joint is further than this "
                        "from the MEASURED position, i.e. the arm cannot keep up with the "
                        "commanded trajectory. The fix for this is a higher --speed-percent or "
                        "lower --action-hz, not a looser bound.")
    p.add_argument("--gripper-scale", type=float, default=0.105,
                   help="Model 0~1 gripper fraction -> meters. VERIFY IN REPLAY FIRST.")
    p.add_argument("--max-steps", type=int, default=1, help="For --mode step: how many actions to execute.")
    p.add_argument("--max-duration", type=float, default=None,
                   help="For --mode closed-loop: hard wall-clock stop (seconds). Required.")
    p.add_argument("--replan-steps", type=int, default=8,
                   help="Execute this many actions from each inferred chunk before re-observing.")
    p.add_argument("--action-hz", type=float, default=ACTION_HZ_DEFAULT)
    p.add_argument("--obs-timeout-s", type=float, default=0.5,
                   help="Abort if camera/state data is older than this.")
    p.add_argument("--max-violations", type=int, default=1,
                   help="Abort after this many safety-limit violations (default: abort on the first).")

    # hardware
    # can0 = left arm, can1 = right arm — matches the existing verified clients
    # on this robot (lawam_deploy/lawam_piper_client.py, check_orientation.py etc).
    p.add_argument("--can-left", default="can0")
    p.add_argument("--can-right", default="can1")
    p.add_argument("--no-ros2", action="store_true",
                   help="Skip ROS2 camera capture (log-only/replay debugging without cameras).")

    # replay
    p.add_argument("--episode-parquet", default=None, help="For --mode replay: a dataset episode parquet.")
    p.add_argument("--replay-max-frames", type=int, default=60, help="For --mode replay: cap frames executed.")

    # moving to the dataset's starting pose before inference
    #
    # The model outputs ABSOLUTE joint angles and was trained on episodes that all
    # begin from a task-appropriate working pose (arms reaching toward the boxes),
    # never from the mechanical zero pose. Inferring from zero means asking the
    # model to extrapolate from a state it never saw, so bringing the arms to a
    # recorded starting pose first is part of normal deployment, not a special mode.
    p.add_argument("--goto-start", action="store_true",
                   help="Before inference, interpolate the arms onto the dataset's starting pose "
                        "(built-in mean of all 471 episodes' first frames; override with "
                        "--start-pose-parquet). THIS MOVES THE ROBOT, so --confirm-safety is "
                        "required even in log-only mode.")
    p.add_argument("--start-pose-parquet", default=None,
                   help="Optional: use the FIRST frame of this episode parquet as the starting "
                        "pose instead of the built-in mean.")
    p.add_argument("--goto-step-rad", type=float, default=0.02,
                   help="Max joint change per interpolation step while approaching the start pose. "
                        "Default 0.02 rad is well below the 0.035 rad p95 of real 30Hz motion.")
    p.add_argument("--goto-settle-s", type=float, default=1.0,
                   help="Pause after reaching the start pose, before inference begins.")

    p.add_argument("--instruction", default=None, help="Override task text sent to the server.")
    p.add_argument("--log-file", default=None, help="Also append action logs to this file (JSON lines).")
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
        """Return (images, age_of_oldest_frame_seconds)."""
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
        super().__init__("fastwam_obs_collector")
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
    """Joint-space Piper wrapper. Mirrors the previously verified implementation."""

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
        """-> (6 joint angles in radians, gripper opening in meters)

        NOTE on the API: joints and gripper come from TWO separate SDK calls.
        `GetArmJointMsgs()` has no `gripper_state` attribute — the gripper lives
        on `GetArmGripperMsgs().gripper_state.grippers_angle` (this is the form
        used by the working `lawam_piper_client.py` on this robot).
        Deliberately NOT wrapped in try/except: a silent fallback to zeros would
        feed the model an all-zero proprio vector while looking healthy, which is
        far more dangerous than failing loudly.
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
        """Stop motion by commanding the current measured pose."""
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
    """Two DISTINCT checks for two distinct failure modes.

    Originally these were conflated into one "commanded vs measured" delta check,
    which produced misleading rejections: streaming absolute position targets at
    30Hz while the arm moves at speed_percent=15 makes the arm lag behind, so the
    command-vs-measured gap grows even when the model's trajectory is perfectly
    smooth. (Observed in practice: consecutive commands differed by only ~0.05 rad
    while the command-vs-measured gap had grown to 0.23 rad.)

      1. TRAJECTORY SMOOTHNESS (command vs PREVIOUS COMMAND) — the correct signal
         for "is the model emitting a wild jump?". Kept tight; real 30Hz training
         data has a p99.9 step of 0.103 rad.
      2. TRACKING ERROR (command vs MEASURED) — detects "the arm cannot keep up".
         Needs a looser threshold, and the right remedy is raising --speed-percent
         or lowering --action-hz, not aborting. Reported separately so the two
         causes are never confused again.
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
        """Return an error string if the action must be rejected, else None."""
        a = np.asarray(action14, dtype=np.float32).reshape(-1)
        if a.shape[0] != 14:
            return f"action must be 14D, got {a.shape[0]}D"
        if not np.all(np.isfinite(a)):
            return f"action contains non-finite values: {a.tolist()}"

        left_j, right_j = a[0:6], a[7:13]
        lg, rg = float(a[6]), float(a[13])

        # --- absolute joint limits
        for label, joints in (("left", left_j), ("right", right_j)):
            over = np.abs(joints) > self.joint_limit
            if np.any(over):
                return (f"{label} joint magnitude exceeds {self.joint_limit} rad at "
                        f"indices {np.where(over)[0].tolist()}: {joints.tolist()}")
        for label, g in (("left", lg), ("right", rg)):
            if not (-0.05 <= g <= 1.05):  # small tolerance: dataset had tiny out-of-range noise
                return f"{label} gripper fraction out of 0~1 range: {g}"

        # --- CHECK 1: trajectory smoothness (this command vs the previous command)
        if self._prev_cmd is not None:
            p = self._prev_cmd
            for label, cmd, prev in (("left", left_j, p[0:6]), ("right", right_j, p[7:13])):
                step = np.abs(cmd - prev)
                if np.any(step > self.max_delta):
                    idx = np.where(step > self.max_delta)[0].tolist()
                    return (f"{label} trajectory jump exceeds {self.max_delta} rad at {idx}: "
                            f"step={np.round(step, 4).tolist()} "
                            f"(model emitted a discontinuous command)")

        # --- CHECK 2: tracking error (command vs measured) — advisory, looser bound
        if current_state14 is not None:
            cur = np.asarray(current_state14, dtype=np.float32).reshape(-1)
            # state layout: [0:6] left joints, [6:12] right joints
            worst = 0.0
            for label, cmd, now in (("left", left_j, cur[0:6]), ("right", right_j, cur[6:12])):
                err = np.abs(cmd - now)
                worst = max(worst, float(err.max()))
                if np.any(err > self.max_tracking_error):
                    idx = np.where(err > self.max_tracking_error)[0].tolist()
                    return (f"{label} TRACKING ERROR exceeds {self.max_tracking_error} rad at {idx}: "
                            f"err={np.round(err, 4).tolist()} — the arm is falling behind the "
                            f"commanded trajectory. Raise --speed-percent or lower --action-hz "
                            f"rather than loosening this bound.")
            self.worst_tracking_error = max(self.worst_tracking_error, worst)
            # Warn once, well before the abort threshold, so lag is visible early.
            if not self._tracking_warned and worst > 0.5 * self.max_tracking_error:
                print(f"  [SAFETY] note: tracking error reached {worst:.4f} rad "
                      f"(abort at {self.max_tracking_error}); the arm is lagging behind commands. "
                      f"Consider a higher --speed-percent or lower --action-hz.")
                self._tracking_warned = True

        self._prev_cmd = a.copy()
        return None

    def reset_trajectory(self):
        """Forget the previous command (call when the action queue is discarded)."""
        self._prev_cmd = None

    def register_violation(self, reason: str) -> bool:
        """-> True if we must abort."""
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
        print(f"[client] connecting to {url} ...")
        self._ws = connect(url, open_timeout=30, max_size=None)
        info = self._request({"ping": True})
        print(f"[client] server info: {json.dumps(info.get('info', {}), indent=2, ensure_ascii=False)}")
        self.info = info.get("info", {})

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(self._msgpack.packb(payload, use_bin_type=True))
        resp = self._msgpack.unpackb(self._ws.recv(), raw=False)
        if not resp.get("ok"):
            raise RuntimeError(f"server error: {resp.get('error')}")
        return resp

    def reset(self):
        self._request({"reset": True})

    def infer(self, images: Dict[str, np.ndarray], state14: np.ndarray,
              instruction: Optional[str]) -> np.ndarray:
        payload: Dict[str, Any] = {"obs": images, "state": np.asarray(state14, dtype=np.float32)}
        if instruction:
            payload["instruction"] = instruction
        resp = self._request(payload)
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
    """Build the 14D proprio vector in TRAINING order (joints 12D then grippers 2D)."""
    lg = float(np.clip(left_gripper_m / gripper_scale, 0.0, 1.0))
    rg = float(np.clip(right_gripper_m / gripper_scale, 0.0, 1.0))
    return np.concatenate([
        np.asarray(left_joints, dtype=np.float32).reshape(6),
        np.asarray(right_joints, dtype=np.float32).reshape(6),
        np.array([lg, rg], dtype=np.float32),
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
    """Read the FIRST recorded action of an episode -> 14D action-layout vector."""
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
    """Interpolate from the CURRENT measured pose to `target14`. Returns True on success.

    Moves in small increments (--goto-step-rad) instead of issuing one large jump,
    so the arms travel smoothly. Every intermediate command is bounded by
    construction, and the whole approach is aborted if any joint would exceed
    --joint-limit-rad.
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
    """Infer repeatedly and PRINT actions. Never sends anything to the arms."""
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
    """Replay recorded dataset actions - no model, known-good motion."""
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
    """--mode step (bounded) / --mode closed-loop (time-bounded)."""
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
                pending.clear()  # do not blindly continue a suspect chunk
                guard.reset_trajectory()  # next command won't be continuous with this one
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


# ======================================================================== main
def main() -> int:
    args = _parse_args()
    motion_modes = {"replay", "step", "closed-loop"}
    # --goto-start physically moves the arms, so it counts as motion even in log-only.
    needs_motion = args.mode in motion_modes or args.goto_start

    # ---- gate checks before touching anything
    if needs_motion and not args.confirm_safety:
        what = f"--mode {args.mode}" if args.mode in motion_modes else "--goto-start"
        print(f"[REFUSED] {what} moves the robot but --confirm-safety was not given.\n"
              f"          Verify: e-stop within reach, workspace clear, no one in the envelope.\n"
              f"          Then re-run with --confirm-safety.")
        return 2
    if args.goto_start and not args.start_pose_parquet:
        print(f"[info] --goto-start using the built-in mean starting pose "
              f"(mean first frame of all 471 training episodes).")
    if args.mode == "closed-loop":
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
    if args.mode != "replay" and not args.server:
        print("[REFUSED] --server is required for inference modes.")
        return 2

    logger = ActionLogger(args.log_file)
    _warn_gripper_scale(args.gripper_scale)
    arms = []
    client: Optional[ServerClient] = None
    ros_node = None
    buf: Optional[ObservationBuffer] = None
    executor_thread = None

    try:
        # ---- arms: needed for motion, and for reading state in log-only
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
                        a.enable()
                else:
                    # log-only: still enable so we can read real joint states,
                    # but no send_command is ever called in this mode.
                    for a in (left, right):
                        try:
                            a.enable()
                        except Exception as e:  # noqa: BLE001
                            print(f"[warn] {a.name} enable failed ({e}); state will read as zeros")
                arms = [left, right]

        # ---- cameras
        if not args.no_ros2 and args.mode != "replay":
            if not HAS_ROS2:
                print("[REFUSED] ROS2 unavailable; pass --no-ros2 only for offline smoke checks.")
                return 2
            buf = ObservationBuffer()
            rclpy.init()
            ros_node = ROS2ObservationCollector(buf, DEFAULT_TOPIC_MAP)
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

        # ---- optional: bring the arms to the dataset's recorded starting pose.
        # Done AFTER cameras/server are ready so a failure there doesn't leave the
        # arms parked mid-approach, and BEFORE any inference so the model sees a
        # state drawn from its training distribution rather than the zero pose.
        if args.goto_start:
            if not arms:
                print("[REFUSED] --goto-start needs the arms connected")
                return 2
            target = (load_start_pose(args.start_pose_parquet)
                      if args.start_pose_parquet else START_POSE_ACTION14)
            if not goto_start_pose(arms, target, args, logger):
                print("[ABORT] start-pose approach failed; not proceeding to inference")
                return 1

        # ---- dispatch
        if args.mode == "log-only":
            return run_log_only(args, arms, buf, client, logger)
        if args.mode == "replay":
            if not arms:
                print("[REFUSED] replay needs the arms connected")
                return 2
            return run_replay(args, arms, logger)
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
