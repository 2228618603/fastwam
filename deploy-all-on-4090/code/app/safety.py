from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Optional

import numpy as np

from .robot_io import GRIPPER_RAW_MAX, METER_TO_GRIPPER_RAW

SPEED_PERCENT_HARD_MAX = 30
ACTION_HZ_DEFAULT = 30.0


def warn_gripper_scale(gripper_scale: float) -> None:
    raw_at_full = round(gripper_scale * METER_TO_GRIPPER_RAW)
    if raw_at_full <= GRIPPER_RAW_MAX:
        return
    saturates_at = GRIPPER_RAW_MAX / (gripper_scale * METER_TO_GRIPPER_RAW)
    print("\n" + "!" * 78)
    print("[GRIPPER SCALE WARNING]")
    print(f"  --gripper-scale {gripper_scale} maps 1.0 opening to {raw_at_full} raw,")
    print(f"  but hardware limit is {GRIPPER_RAW_MAX} raw.")
    print(f"  Commands are clamped; outputs above ~{saturates_at:.3f} saturate.")
    print(f"  Verify gripper scale in replay before changing it.")
    print("!" * 78 + "\n")


class SafetyGuard:
    def __init__(self, joint_limit_rad: float, max_joint_delta_rad: float, max_tracking_error_rad: float, max_violations: int):
        self.joint_limit = float(joint_limit_rad)
        self.max_delta = float(max_joint_delta_rad)
        self.max_tracking_error = float(max_tracking_error_rad)
        self.max_violations = int(max_violations)
        self.violations = 0
        self._prev_cmd: Optional[np.ndarray] = None
        self.worst_tracking_error = 0.0
        self._tracking_warned = False

    def check(self, action14: np.ndarray, current_state14: Optional[np.ndarray]) -> Optional[str]:
        a = np.asarray(action14, dtype=np.float32).reshape(-1)
        if a.shape[0] != 14:
            return f"action must be 14D, got {a.shape[0]}D"
        if not np.all(np.isfinite(a)):
            return f"action contains non-finite values: {a.tolist()}"

        left_j, right_j = a[0:6], a[7:13]
        lg, rg = float(a[6]), float(a[13])
        for label, joints in (("left", left_j), ("right", right_j)):
            over = np.abs(joints) > self.joint_limit
            if np.any(over):
                return f"{label} joint magnitude exceeds {self.joint_limit} rad at {np.where(over)[0].tolist()}: {joints.tolist()}"
        for label, g in (("left", lg), ("right", rg)):
            if not (-0.05 <= g <= 1.05):
                return f"{label} gripper fraction out of 0~1 range: {g}"

        if self._prev_cmd is not None:
            prev = self._prev_cmd
            for label, cmd, old in (("left", left_j, prev[0:6]), ("right", right_j, prev[7:13])):
                step = np.abs(cmd - old)
                if np.any(step > self.max_delta):
                    return (
                        f"{label} trajectory jump exceeds {self.max_delta} rad at "
                        f"{np.where(step > self.max_delta)[0].tolist()}: step={np.round(step, 4).tolist()}"
                    )

        if current_state14 is not None:
            cur = np.asarray(current_state14, dtype=np.float32).reshape(-1)
            worst = 0.0
            for label, cmd, now in (("left", left_j, cur[0:6]), ("right", right_j, cur[6:12])):
                err = np.abs(cmd - now)
                worst = max(worst, float(err.max()))
                if np.any(err > self.max_tracking_error):
                    return (
                        f"{label} tracking error exceeds {self.max_tracking_error} rad at "
                        f"{np.where(err > self.max_tracking_error)[0].tolist()}: err={np.round(err, 4).tolist()}"
                    )
            self.worst_tracking_error = max(self.worst_tracking_error, worst)
            if not self._tracking_warned and worst > 0.5 * self.max_tracking_error:
                print(f"  [SAFETY] tracking error reached {worst:.4f} rad; consider lower --action-hz or higher --speed-percent.")
                self._tracking_warned = True

        self._prev_cmd = a.copy()
        return None

    def reset_trajectory(self) -> None:
        self._prev_cmd = None

    def register_violation(self, reason: str) -> bool:
        self.violations += 1
        print(f"  [SAFETY] REJECTED ({self.violations}/{self.max_violations}): {reason}")
        return self.violations >= self.max_violations


def assemble_state14(left_joints: np.ndarray, left_gripper_m: float, right_joints: np.ndarray, right_gripper_m: float, gripper_scale: float) -> np.ndarray:
    lg = float(np.clip(left_gripper_m / gripper_scale, 0.0, 1.0))
    rg = float(np.clip(right_gripper_m / gripper_scale, 0.0, 1.0))
    return np.concatenate(
        [
            np.asarray(left_joints, dtype=np.float32).reshape(6),
            np.asarray(right_joints, dtype=np.float32).reshape(6),
            np.array([lg, rg], dtype=np.float32),
        ]
    )


def split_action14(action14: np.ndarray, gripper_scale: float):
    a = np.asarray(action14, dtype=np.float32).reshape(-1)
    left_j, right_j = a[0:6], a[7:13]
    lg_m = float(np.clip(a[6], 0.0, 1.0)) * gripper_scale
    rg_m = float(np.clip(a[13], 0.0, 1.0)) * gripper_scale
    return left_j, lg_m, right_j, rg_m


def describe_action(action14: np.ndarray, gripper_scale: float) -> str:
    lj, lg, rj, rg = split_action14(action14, gripper_scale)
    return f"L_joints={np.round(lj, 4).tolist()} L_grip={lg:.5f}m | R_joints={np.round(rj, 4).tolist()} R_grip={rg:.5f}m"


class ActionLogger:
    def __init__(self, path: Optional[str]):
        self._f = open(path, "a") if path else None

    def log(self, record: Dict[str, Any]) -> None:
        if self._f:
            self._f.write(json.dumps(record, default=float) + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()

