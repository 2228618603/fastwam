from __future__ import annotations

import time

import numpy as np

from .safety import split_action14

START_POSE_ACTION14 = np.array(
    [
        -0.2187,
        +0.1383,
        -0.4933,
        -0.2940,
        +0.8513,
        +0.2763,
        +0.5086,
        +0.3524,
        +0.1806,
        -0.4801,
        +0.0619,
        +0.7892,
        -0.0104,
        +0.5402,
    ],
    dtype=np.float32,
)


def load_start_pose(parquet_path: str) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    col = "action" if "action" in table.column_names else "actions"
    if col not in table.column_names:
        raise ValueError(f"no action column in {parquet_path}; columns={table.column_names}")
    first = np.asarray(table[col].to_pylist()[0], dtype=np.float32).reshape(-1)
    if first.shape[0] != 14:
        raise ValueError(f"expected 14D action, got {first.shape[0]}D")
    return first


def goto_start_pose(arms, target14: np.ndarray, args, logger) -> bool:
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
    print("APPROACHING DATASET START POSE - THE ROBOT WILL MOVE")
    print(f"  largest joint travel = {max_travel:.4f} rad ({np.degrees(max_travel):.1f} deg)")
    print(f"  -> {n_steps} interpolation steps, speed={args.speed_percent}")
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
            logger.log({"t": time.time(), "phase": "goto_start", "k": k, "n": n_steps, "left": lj.tolist(), "right": rj.tolist()})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[ABORT] approach interrupted; holding current pose")
        for arm in arms:
            arm.hold()
        return False

    if args.goto_settle_s > 0:
        time.sleep(args.goto_settle_s)
    return True

