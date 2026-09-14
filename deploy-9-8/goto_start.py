#!/usr/bin/env python3
"""机械臂归位脚本 —— 把双臂插值移动到训练起始姿态后退出.

与 robot_client.py 共享相同的单位约定和安全参数, 但只做归位, 不推理.

用法（会移动机械臂，必须先确认急停和工作区安全）:
    python goto_start.py --confirm-safety --speed-percent 10  # 内置均值姿态, 低速
    python goto_start.py --confirm-safety --parquet episode_0.parquet  # 用某条 episode 首帧

首次联调优先使用 robot_client.py --goto-start；本脚本是独立的归位工具，
不连接 server，也不做模型推理。不要在第一次真机测试时使用 100% 速度。
"""

import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np

# ------------------------------------------------------------------- 单位换算
RAD_TO_DEG_001 = 1000.0 * 180.0 / math.pi
METER_TO_GRIPPER_RAW = 1_000_000.0
GRIPPER_RAW_MIN, GRIPPER_RAW_MAX = 0, 70000

# ------------------------------------------------------------------- 内置起始姿态 (471 条训练 episode 首帧均值)
START_POSE = np.array([
    -0.2187,  0.1383, -0.4933, -0.2940,  0.8513,  0.2763,  0.5086,  # 左: J1~J6 + 爪
     0.3524,  0.1806, -0.4801,  0.0619,  0.7892, -0.0104,  0.5402,  # 右: J1~J6 + 爪
], dtype=np.float32)


def split_action14(action14: np.ndarray, gripper_scale: float):
    a = np.asarray(action14, dtype=np.float32).reshape(-1)
    return a[0:6], float(np.clip(a[6], 0, 1)) * gripper_scale, \
           a[7:13], float(np.clip(a[13], 0, 1)) * gripper_scale


class Arm:
    """Piper 单臂关节空间控制."""
    def __init__(self, can: str, label: str):
        from piper_sdk import C_PiperInterface_V2
        self.label = label
        self.piper = C_PiperInterface_V2(can)
        self._enabled = False

    def connect(self):
        print(f"  [{self.label}] 连接 {self.piper.can_name} ...")
        self.piper.ConnectPort()
        time.sleep(0.2)

    def enable(self, timeout=5):
        print(f"  [{self.label}] 使能 ...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.piper.EnablePiper():
                self._enabled = True
                print(f"  [{self.label}] 已使能")
                return
            time.sleep(0.01)
        raise RuntimeError(f"[{self.label}] 使能超时")

    def read_state(self) -> Tuple[np.ndarray, float]:
        if not self._enabled:
            return np.zeros(6), 0.0
        js = self.piper.GetArmJointMsgs().joint_state
        joints = np.array([js.joint_1, js.joint_2, js.joint_3,
                           js.joint_4, js.joint_5, js.joint_6]) / RAD_TO_DEG_001
        g = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / METER_TO_GRIPPER_RAW
        return joints.astype(np.float32), float(g)

    def send(self, joints_rad: np.ndarray, gripper_m: float, speed: int):
        if not self._enabled:
            raise RuntimeError(f"[{self.label}] 未使能")
        # 速度档
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=int(speed), is_mit_mode=0x00)
        # 关节
        self.piper.JointCtrl(*[round(float(j) * RAD_TO_DEG_001) for j in joints_rad])
        # 爪
        raw = int(round(float(gripper_m) * METER_TO_GRIPPER_RAW))
        raw = max(GRIPPER_RAW_MIN, min(GRIPPER_RAW_MAX, raw))
        self.piper.GripperCtrl(raw, 1000, 0x01, 0x00)

    def hold(self):
        if not self._enabled:
            return
        try:
            j, g = self.read_state()
            self.send(j, g, speed=5)
        except Exception:
            pass

    def disconnect(self):
        print(f"  [{self.label}] 断开")


def goto_start(left: Arm, right: Arm, target14: np.ndarray,
               gripper_scale: float, speed: int, step_rad: float, action_hz: float,
               joint_limit_rad: float, settle_s: float):
    """从当前姿态插值到 target14."""
    tgt_lj, tgt_lg, tgt_rj, tgt_rg = split_action14(target14, gripper_scale)
    cur_lj, cur_lg = left.read_state()
    cur_rj, cur_rg = right.read_state()

    # 安全检查
    for name, joints in [("左臂目标", tgt_lj), ("右臂目标", tgt_rj)]:
        over = np.abs(joints) > joint_limit_rad
        if np.any(over):
            print(f"[中止] {name} 超出关节限制 (±{joint_limit_rad} rad): {joints}")
            return False

    max_travel = float(max(np.abs(tgt_lj - cur_lj).max(), np.abs(tgt_rj - cur_rj).max()))
    n_steps = max(1, int(np.ceil(max_travel / max(step_rad, 1e-6))))

    print("=" * 60)
    print("归位 — 机械臂即将移动")
    print(f"  当前 左关节: {np.round(cur_lj, 4).tolist()}")
    print(f"  目标 左关节: {np.round(tgt_lj, 4).tolist()}  爪={tgt_lg:.4f}m")
    print(f"  当前 右关节: {np.round(cur_rj, 4).tolist()}")
    print(f"  目标 右关节: {np.round(tgt_rj, 4).tolist()}  爪={tgt_rg:.4f}m")
    print(f"  最大行程: {max_travel:.4f} rad ({math.degrees(max_travel):.1f}°)")
    print(f"  → {n_steps} 步, 每步 ≤{step_rad} rad, 速度 {speed}%")
    print("=" * 60)

    for i in range(3, 0, -1):
        print(f"  {i}s 后开始 ... (Ctrl-C 中止)")
        time.sleep(1)

    dt = 1.0 / action_hz
    try:
        for k in range(1, n_steps + 1):
            f = k / n_steps
            lj = cur_lj + (tgt_lj - cur_lj) * f
            rj = cur_rj + (tgt_rj - cur_rj) * f
            lg = cur_lg + (tgt_lg - cur_lg) * f
            rg = cur_rg + (tgt_rg - cur_rg) * f
            left.send(lj, lg, speed)
            right.send(rj, rg, speed)
            if k % 10 == 0 or k == n_steps:
                print(f"  [{k}/{n_steps}] L={np.round(lj,4).tolist()} R={np.round(rj,4).tolist()}")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[中止] 用户中断, 保持当前姿态")
        left.hold()
        right.hold()
        return False

    if settle_s > 0:
        print(f"  等待 {settle_s}s 稳定 ...")
        time.sleep(settle_s)

    got_lj, got_lg = left.read_state()
    got_rj, got_rg = right.read_state()
    err_l = float(np.abs(got_lj - tgt_lj).max())
    err_r = float(np.abs(got_rj - tgt_rj).max())
    print(f"  到达: 左误差={err_l:.4f} rad, 右误差={err_r:.4f} rad")
    print(f"  左: {np.round(got_lj,4).tolist()}  爪={got_lg:.4f}m")
    print(f"  右: {np.round(got_rj,4).tolist()}  爪={got_rg:.4f}m")
    print("=" * 60)
    return True


def load_start_pose(parquet_path: str) -> np.ndarray:
    """从 parquet 读首帧作为目标姿态."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path, columns=["actions", "episode_index"])
    first = np.asarray(t.column("actions").to_pylist()[0], dtype=np.float32)
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--can-left", default="can0")
    p.add_argument("--can-right", default="can1")
    p.add_argument("--speed-percent", type=int, default=30)
    p.add_argument("--gripper-scale", type=float, default=0.105,
                   help="0~1 → 米 (默认 0.105, 待真机验证)")
    p.add_argument("--joint-limit-rad", type=float, default=3.14)
    p.add_argument("--step-rad", type=float, default=0.02,
                   help="每步最大关节变化 (默认 0.02 rad)")
    p.add_argument("--action-hz", type=float, default=30)
    p.add_argument("--settle-s", type=float, default=1.0,
                   help="到达后稳定时间")
    p.add_argument("--parquet", default=None,
                   help="用指定 episode parquet 首帧作为目标 (默认内置均值)")
    p.add_argument("--confirm-safety", action="store_true",
                   help="确认急停在手边、工作区清空")
    args = p.parse_args()

    if not args.confirm_safety:
        print("[拒绝] 归位会移动机械臂, 需要 --confirm-safety")
        return 2

    target = START_POSE if args.parquet is None else load_start_pose(args.parquet)
    print(f"[目标姿态来源] {'内置 471 episode 均值' if args.parquet is None else args.parquet}")

    left = Arm(args.can_left, "左臂")
    right = Arm(args.can_right, "右臂")
    try:
        left.connect()
        right.connect()
        left.enable()
        right.enable()

        ok = goto_start(left, right, target,
                        gripper_scale=args.gripper_scale,
                        speed=args.speed_percent,
                        step_rad=args.step_rad,
                        action_hz=args.action_hz,
                        joint_limit_rad=args.joint_limit_rad,
                        settle_s=args.settle_s)
        return 0 if ok else 1
    finally:
        left.disconnect()
        right.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
