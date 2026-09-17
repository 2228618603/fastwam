#!/usr/bin/env python
"""假 RobotArmService —— 无硬件跑通 deploy_real.py 的 run 路径。

协议与 openpi 的 `RobotArmService` 完全一致(ping / reset / step / stop,外加只读 obs)，
并模拟一阶跟踪动力学:手臂以有限速度趋近被下发的目标位置。

图像来自 /tmp/fake_obs.npz(真实数据集帧)，所以模型看到的是分布内输入。
起始位姿可选:
  --init dataset   数据集里那个 episode 的真实起始位姿(默认，分布内)
  --init openpi    openpi service 的 _DEFAULT_INIT_JOINTS_*(**分布外**，用来看 preflight 会不会报)

用法:
    source env.sh
    # 1) 先从数据集抽一帧(约 1 分钟,只做一次)
    python scripts/deploy_fake_service.py --extract
    # 2) 起假服务(记下 PID —— 别用 pkill -f 收尾,-f 会匹配到调用方自己的命令行)
    python scripts/deploy_fake_service.py --port 19912 & FAKE_PID=$!
    # 3) 跑 deploy_real.py 的 run 路径
    python scripts/deploy_real.py run --config configs/deploy/agilex_real.yaml \
        --endpoint tcp://127.0.0.1:19912 --max-steps 40 --record-dir /tmp/deploy_rec
    # 4) 收尾
    kill $FAKE_PID

已用它验证过:preflight 全项、sync/async 两种控制环、安全层钳位、落盘、OOD 起始位姿告警。
"""
import argparse
import io
import socket
import struct
import sys
import time

from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_robot_client import Packer, unpackb  # 复用同一套 msgpack-numpy 编解码

GRIPPER_TRAVEL_M = 0.07
# openpi RobotArmService 的默认初始位姿(robot_arm_service.py:103-110)
OPENPI_INIT_L = np.array([-0.47566299, 0.12493393, -0.49851463, 0.09187755, 0.84711553, -0.11246147], np.float32)
OPENPI_INIT_R = np.array([0.05784430, 0.45746890, -0.47620376, 0.25457774, 0.83889940, -0.19685554], np.float32)
OPENPI_INIT_GRIP_M = np.array([0.000, 0.060], np.float32)

# 一步(33.3ms)最多走多少 —— 取训练数据 p99.9,模拟真实伺服跟踪能力
TRACK_RATE_JOINT = 0.10
TRACK_RATE_GRIP_M = 0.20 * GRIPPER_TRAVEL_M


class FakeArm:
    def __init__(self, init_joint: np.ndarray, init_grip_m: np.ndarray):
        self.init_joint = init_joint.copy()
        self.init_grip_m = init_grip_m.copy()
        self.joint = init_joint.copy()
        self.grip_m = init_grip_m.copy()

    def reset(self):
        self.joint = self.init_joint.copy()
        self.grip_m = self.init_grip_m.copy()

    def step(self, action14: np.ndarray):
        """action 排布: [左6, 左爪(米), 右6, 右爪(米)]，一阶限速趋近。"""
        tgt_j = np.concatenate([action14[0:6], action14[7:13]])
        tgt_g = np.array([action14[6], action14[13]], np.float32)
        self.joint += np.clip(tgt_j - self.joint, -TRACK_RATE_JOINT, TRACK_RATE_JOINT)
        self.grip_m += np.clip(tgt_g - self.grip_m, -TRACK_RATE_GRIP_M, TRACK_RATE_GRIP_M)
        self.grip_m = np.clip(self.grip_m, 0.0, GRIPPER_TRAVEL_M)


def jpeg(rgb: np.ndarray) -> np.ndarray:
    """等价于 service 的 cv2.imencode(".jpg", rgb[:,:,::-1]) —— 落盘颜色正确。"""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def extract(out_npz: str, dataset_dir: str, task: str, norm_stats: str, idx: int) -> None:
    """从训练数据集抽一帧(3 相机 + 真实 joint/gripper)存成 npz,给本假 service 用。

    喂**分布内**的图像和起始位姿,模型才会输出合理动作,控制环/安全层才算在真实工况下被测到。
    用随机噪声图会让首步跳变巨大、直接触发 abort,测不到东西。
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    repo = Path(__file__).resolve().parent.parent
    misc.register_work_dir(str(repo / ".cache" / "deploy_fake"))
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.3"):
        hcfg = compose(config_name="train", overrides=[f"task={task}"])
    ds = instantiate(hcfg.data.val, dataset_dirs=[dataset_dir], pretrained_norm_stats=norm_stats)
    base = ds.lerobot_dataset
    metas = base.processor.shape_meta["images"]
    saved, base.processor = base.processor, None      # 摘掉 processor 拿原始 sample
    try:
        raw = base[idx]
    finally:
        base.processor = saved
    out = {m["key"]: raw["images"][m["key"]][0].permute(1, 2, 0).numpy().astype(np.uint8)
           for m in metas}
    out["joint"] = raw["state"]["joint"][0].numpy().astype(np.float32)
    out["gripper_frac"] = raw["state"]["gripper_position"][0].numpy().astype(np.float32)
    np.savez_compressed(out_npz, **out)
    np.set_printoptions(precision=4, suppress=True)
    print(f"已存 {out_npz}")
    print("  joint        =", out["joint"])
    print("  gripper_frac =", out["gripper_frac"], "-> 米", out["gripper_frac"] * GRIPPER_TRAVEL_M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true",
                    help="不起服务,只从数据集抽一帧存到 --obs-npz(第一次用要先跑这个)")
    ap.add_argument("--dataset-dir", default="./data/agilex_empty_the_box_fastwam")
    ap.add_argument("--task", default="agilex_final_3cam_384_1e-4")
    ap.add_argument("--norm-stats", default="./runs/_shared/dataset_stats.json")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--port", type=int, default=19912)
    ap.add_argument("--init", choices=["dataset", "openpi"], default="dataset")
    ap.add_argument("--obs-npz", default="/tmp/fake_obs.npz")
    args = ap.parse_args()

    if args.extract:
        extract(args.obs_npz, args.dataset_dir, args.task, args.norm_stats, args.sample_idx)
        return 0

    d = np.load(args.obs_npz)
    imgs = {k: jpeg(d[k]) for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    # service 端的相机名是 cam_top / cam_left_wrist / cam_right_wrist
    imgs = {"cam_top": imgs["cam_high"],
            "cam_left_wrist": imgs["cam_left_wrist"],
            "cam_right_wrist": imgs["cam_right_wrist"]}

    if args.init == "dataset":
        arm = FakeArm(d["joint"], d["gripper_frac"] * GRIPPER_TRAVEL_M)
    else:
        arm = FakeArm(np.concatenate([OPENPI_INIT_L, OPENPI_INIT_R]), OPENPI_INIT_GRIP_M)
    np.set_printoptions(precision=4, suppress=True)
    print(f"[fake] init={args.init}  joint={arm.init_joint}  grip_m={arm.init_grip_m}")

    def obs():
        return {"state": arm.joint.astype(np.float32),
                "gripper_position": arm.grip_m.astype(np.float32),
                "images": imgs,
                "prompt": "fake service"}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[fake] listening on 127.0.0.1:{args.port}")
    packer = Packer()
    n_step = 0
    while True:
        conn, _ = srv.accept()
        print("[fake] client connected")
        try:
            while True:
                hdr = conn.recv(4)
                if not hdr:
                    break
                (n,) = struct.unpack(">I", hdr)
                buf = b""
                while len(buf) < n:
                    buf += conn.recv(n - len(buf))
                msg = unpackb(buf)
                cmd = msg.get("cmd")
                if cmd == "ping":
                    rep = {"pong": True}
                elif cmd == "obs":
                    rep = {"obs": obs()}
                elif cmd == "reset":
                    arm.reset()
                    n_step = 0
                    time.sleep(0.3)          # 真机是 5s，这里缩短
                    rep = {"obs": obs()}
                elif cmd == "step":
                    arm.step(np.asarray(msg["action"], np.float32))
                    n_step += 1
                    rep = {"obs": obs(), "done": False, "info": {"step": n_step}}
                elif cmd == "stop":
                    print("[fake] stop (go_zero)")
                    rep = {"status": "ok"}
                else:
                    rep = {"error": f"unknown cmd: {cmd!r}"}
                p = packer.pack(rep)
                conn.sendall(struct.pack(">I", len(p)) + p)
        except (ConnectionError, EOFError, struct.error):
            pass
        finally:
            conn.close()
            print(f"[fake] client disconnected after {n_step} steps")


if __name__ == "__main__":
    raise SystemExit(main())
