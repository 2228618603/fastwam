from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fastwam_local_policy import FastWAMLocalPolicy
from app.layout import CAMERA_KEYS
from app.robot_io import DEFAULT_TOPIC_MAP, HAS_PIPER, HAS_ROS2, ObservationBuffer, PiperArmJoint, ROS2ObservationCollector
from app.safety import (
    ACTION_HZ_DEFAULT,
    SPEED_PERCENT_HARD_MAX,
    ActionLogger,
    SafetyGuard,
    assemble_state14,
    describe_action,
    split_action14,
    warn_gripper_scale,
)
from app.start_pose import START_POSE_ACTION14, goto_start_pose, load_start_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fastwam_local_runner")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="log-only", choices=["self-test", "log-only", "replay", "step", "closed-loop"])
    p.add_argument("--no-ros2", action="store_true", help="Use synthetic gray images; only for smoke/log-only debugging.")

    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset-stats", default=None)
    p.add_argument("--fixed-context", default=None)
    p.add_argument("--action-dit", default=None)
    p.add_argument("--model-cache", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--action-horizon", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cuda-memory-fraction", type=float, default=None)
    p.add_argument("--compile-action-infer", action="store_true")
    p.add_argument("--print-paths", action="store_true")

    p.add_argument("--confirm-safety", action="store_true")
    p.add_argument("--i-am-watching", action="store_true")
    p.add_argument("--speed-percent", type=int, default=10)
    p.add_argument("--joint-limit-rad", type=float, default=3.14)
    p.add_argument("--max-joint-delta-rad", type=float, default=0.2)
    p.add_argument("--max-tracking-error-rad", type=float, default=0.5)
    p.add_argument("--gripper-scale", type=float, default=0.105)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--max-duration", type=float, default=None)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--action-hz", type=float, default=ACTION_HZ_DEFAULT)
    p.add_argument("--obs-timeout-s", type=float, default=0.5)
    p.add_argument("--max-violations", type=int, default=1)
    p.add_argument("--can-left", default="can0")
    p.add_argument("--can-right", default="can1")

    p.add_argument("--episode-parquet", default=None)
    p.add_argument("--replay-max-frames", type=int, default=60)
    p.add_argument("--goto-start", action="store_true")
    p.add_argument("--start-pose-parquet", default=None)
    p.add_argument("--goto-step-rad", type=float, default=0.02)
    p.add_argument("--goto-settle-s", type=float, default=1.0)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


def make_policy(args) -> FastWAMLocalPolicy:
    return FastWAMLocalPolicy(
        ckpt=args.ckpt,
        dataset_stats=args.dataset_stats,
        fixed_context=args.fixed_context,
        action_dit=args.action_dit,
        model_cache=args.model_cache,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        cuda_memory_fraction=args.cuda_memory_fraction,
        compile_action_infer=args.compile_action_infer,
    )


def synthetic_images(value: int = 128):
    return {k: np.full((480, 640, 3), value, dtype=np.uint8) for k in CAMERA_KEYS}


def run_self_test(args) -> int:
    policy = make_policy(args)
    rng = np.random.default_rng(0)
    images = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in CAMERA_KEYS}
    state = np.zeros(14, dtype=np.float32)
    t0 = time.perf_counter()
    chunk = policy.infer(images, state)
    infer_s = time.perf_counter() - t0
    print(f"SELF TEST PASSED: action chunk {chunk.shape}, dtype={chunk.dtype}, infer_s={infer_s:.3f}")
    print(f"first action row: {np.round(chunk[0], 4).tolist()}")
    return 0


def run_log_only(args, arms, buf: Optional[ObservationBuffer], policy: FastWAMLocalPolicy, action_logger: ActionLogger) -> int:
    print("\nMODE: log-only - NO commands will be sent to the robot.\n")
    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad, args.max_tracking_error_rad, max_violations=10**9)
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
                print("[info] --no-ros2: using synthetic gray images")
                images = synthetic_images()
            chunk = policy.infer(images, state14)
            n += 1
            print(f"\n--- inference #{n}: chunk {chunk.shape} ---")
            for i in range(min(3, chunk.shape[0])):
                verdict = guard.check(chunk[i], state14)
                tag = "OK" if verdict is None else f"WOULD-REJECT: {verdict}"
                print(f"  [{i}] {describe_action(chunk[i], args.gripper_scale)} -> {tag}")
            action_logger.log({"t": time.time(), "mode": "log-only", "n": n, "state14": state14.tolist(), "chunk_first3": chunk[:3].tolist()})
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[log-only] stopped")
    return 0


def run_replay(args, arms, action_logger: ActionLogger) -> int:
    if not args.episode_parquet:
        print("[error] --mode replay requires --episode-parquet")
        return 2
    import pyarrow.parquet as pq

    table = pq.read_table(args.episode_parquet)
    col = "action" if "action" in table.column_names else "actions"
    actions = np.stack(table[col].to_pylist()).astype(np.float32)
    n_frames = min(args.replay_max_frames, actions.shape[0])
    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad, args.max_tracking_error_rad, args.max_violations)
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
                    break
                continue
            l_j, l_g, r_j, r_g = split_action14(actions[i], args.gripper_scale)
            arms[0].send_command(l_j, l_g, args.speed_percent)
            arms[1].send_command(r_j, r_g, args.speed_percent)
            executed += 1
            action_logger.log({"t": time.time(), "mode": "replay", "i": i, "action14": actions[i].tolist()})
            time.sleep(dt)
    finally:
        for arm in arms:
            arm.hold()
    print(f"[replay] executed {executed} frames")
    return 0


def run_motion(args, arms, buf: ObservationBuffer, policy: FastWAMLocalPolicy, action_logger: ActionLogger) -> int:
    is_step = args.mode == "step"
    budget = args.max_steps if is_step else None
    deadline = None if is_step else time.time() + args.max_duration
    print(f"\nMODE: {args.mode} - THE ROBOT WILL MOVE. speed={args.speed_percent}\n")
    for i in (3, 2, 1):
        print(f"  starting in {i}s ... (Ctrl-C to abort)")
        time.sleep(1)
    guard = SafetyGuard(args.joint_limit_rad, args.max_joint_delta_rad, args.max_tracking_error_rad, args.max_violations)
    pending: deque = deque()
    executed = 0
    dt = 1.0 / args.action_hz
    policy.reset()
    try:
        while True:
            if budget is not None and executed >= budget:
                break
            if deadline is not None and time.time() >= deadline:
                break
            lj, lg_m = arms[0].read_state()
            rj, rg_m = arms[1].read_state()
            state14 = assemble_state14(lj, lg_m, rj, rg_m, args.gripper_scale)
            if not pending:
                images, age = buf.snapshot()
                if age > args.obs_timeout_s:
                    print(f"[ABORT] observation stale ({age:.3f}s > {args.obs_timeout_s}s)")
                    break
                chunk = policy.infer(images, state14)
                take = min(args.replan_steps, chunk.shape[0])
                if budget is not None:
                    take = min(take, budget - executed)
                for k in range(take):
                    pending.append(chunk[k])
                print(f"[infer] queued {take} action(s) from chunk {chunk.shape}")
            action = pending.popleft()
            reason = guard.check(action, state14)
            if reason is not None:
                if guard.register_violation(reason):
                    break
                pending.clear()
                guard.reset_trajectory()
                continue
            l_j, l_g, r_j, r_g = split_action14(action, args.gripper_scale)
            arms[0].send_command(l_j, l_g, args.speed_percent)
            arms[1].send_command(r_j, r_g, args.speed_percent)
            executed += 1
            print(f"  [exec {executed}] {describe_action(action, args.gripper_scale)}")
            action_logger.log({"t": time.time(), "mode": args.mode, "n": executed, "state14": state14.tolist(), "action14": action.tolist()})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[ABORT] interrupted")
    finally:
        for arm in arms:
            arm.hold()
    print(f"[{args.mode}] executed {executed} action(s); worst tracking error {guard.worst_tracking_error:.4f} rad")
    return 0


def main() -> int:
    args = parse_args()
    if args.print_paths:
        for name in ("ckpt", "dataset_stats", "fixed_context", "action_dit", "model_cache"):
            print(f"{name}: {getattr(args, name)}")
        return 0
    if args.mode == "self-test":
        return run_self_test(args)

    motion_modes = {"replay", "step", "closed-loop"}
    needs_motion = args.mode in motion_modes or args.goto_start
    if needs_motion and not args.confirm_safety:
        print("[REFUSED] motion requested without --confirm-safety")
        return 2
    if args.mode == "closed-loop":
        if not args.i_am_watching:
            print("[REFUSED] closed-loop requires --i-am-watching")
            return 2
        if not args.max_duration:
            print("[REFUSED] closed-loop requires --max-duration")
            return 2
    if args.speed_percent < 1 or args.speed_percent > SPEED_PERCENT_HARD_MAX:
        print(f"[REFUSED] --speed-percent must be within 1..{SPEED_PERCENT_HARD_MAX}")
        return 2

    warn_gripper_scale(args.gripper_scale)
    action_logger = ActionLogger(args.log_file)
    arms = []
    ros_node = None
    buf = None
    try:
        want_arms = needs_motion or not args.no_ros2
        if want_arms:
            if not HAS_PIPER:
                if needs_motion:
                    print("[REFUSED] piper_sdk not installed but motion was requested.")
                    return 2
                print("[warn] piper_sdk unavailable; using zero state")
            else:
                arms = [PiperArmJoint(args.can_left, "left"), PiperArmJoint(args.can_right, "right")]
                for arm in arms:
                    arm.connect()
                    arm.enable()
        if not args.no_ros2 and args.mode != "replay":
            if not HAS_ROS2:
                print("[REFUSED] ROS2 unavailable; pass --no-ros2 for offline smoke checks.")
                return 2
            import rclpy

            buf = ObservationBuffer()
            rclpy.init()
            ros_node = ROS2ObservationCollector(buf, DEFAULT_TOPIC_MAP)
            threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()
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

        policy = None if args.mode == "replay" else make_policy(args)
        if args.goto_start:
            target = load_start_pose(args.start_pose_parquet) if args.start_pose_parquet else START_POSE_ACTION14
            if not goto_start_pose(arms, target, args, action_logger):
                return 1
        if args.mode == "log-only":
            return run_log_only(args, arms, buf, policy, action_logger)
        if args.mode == "replay":
            return run_replay(args, arms, action_logger)
        return run_motion(args, arms, buf, policy, action_logger)
    finally:
        for arm in arms:
            arm.disconnect()
        if ros_node is not None:
            ros_node.destroy_node()
            try:
                import rclpy

                rclpy.shutdown()
            except Exception:
                pass
        action_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
