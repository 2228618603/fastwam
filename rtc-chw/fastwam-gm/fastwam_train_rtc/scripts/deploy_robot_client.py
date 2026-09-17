#!/usr/bin/env python
"""RobotArmService 的最小 TCP 客户端 —— FastWAM 真机部署专用。

对端是 openpi 仓库的 `scripts/deploy/start_robot_arm_service.py`
（实现在 `openpi/src/openpi/serving/robot_arm_service.py`），跑在机器人 PC 上，
里面封装了 ROS2 观测采集 + Piper SDK 双臂控制。

为什么不直接 import openpi 的 `AgilexEnv`
------------------------------------------
1. **numpy 大版本冲突**：fastwam conda env 是 numpy 2.2.6，openpi 钉死 `numpy>=1.22,<2.0`
2. **cv2 没装**：fastwam env 里没有 opencv，而 `AgilexEnv._decode_images` 依赖 `cv2.imdecode`
3. openpi 的 Env 还要拖进 `geekrl.core.env`，为了 4 个命令不值得

线协议本身只有 4 个命令、40 行编解码，所以这里**自带实现**，
`msgpack_numpy` 那 40 行是从 `openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py`
**逐字搬过来的**（必须逐字，否则 ndarray 的 `__ndarray__`/`__npgeneric__` 键名一变就通信不上）。
fastwam env 有 `msgpack 1.2.1`，够用。

线协议
------
帧格式：4 字节大端无符号长度前缀 + msgpack-numpy 载荷。

    client -> server                        server -> client
    {"cmd": "ping"}                         {"pong": True}
    {"cmd": "reset"}                        {"obs": obs}
    {"cmd": "step", "action": (14,) f32}    {"obs": obs, "done": bool, "info": dict}
    {"cmd": "stop"}                         {"status": "ok"}
    任何异常                                 {"error": str}

obs 的字段与单位（**注意与训练数据的单位不同，换算在 deploy_real.py 里做**）：

    state:            (12,) float32  弧度，[left_j0..j5, right_j0..j5]
    gripper_position: (2,)  float32  **米**，0~0.07（训练数据里是 0~1 分数）
    images:           dict  cam_top / cam_left_wrist / cam_right_wrist
                            每个是 1-D uint8 numpy 数组 = JPEG 字节流（q95，已中心裁到 480x640）
    prompt:           str   service 启动时 --prompt 传的，本部署**不使用**（见 deploy_real.py）

step 的 action 语义（`robot_arm_service.py:467-468`）：

    action[0:6]  -> 左臂 6 关节，弧度
    action[6]    -> 左夹爪，**米**（内部 round(m*1e6) 后 clip 到 [0, 70000]）
    action[7:13] -> 右臂 6 关节，弧度
    action[13]   -> 右夹爪，**米**

⚠️ JPEG 颜色通道：**用 PIL 解码时不要再翻通道**
--------------------------------------------
service 那边做的是 `cv2.imencode(".jpg", rgb[:, :, ::-1])`：cv2 把入参当 BGR 处理，
而入参正好是真实图像的 BGR，所以**落盘的 JPEG 颜色是正确的**。
于是：
  - `cv2.imdecode(...)` 返回 BGR，所以 openpi 的 `AgilexEnv` 要 `[:, :, ::-1]` 翻回 RGB；
  - `PIL.Image.open(...)` 直接返回 **RGB**，**不能再翻**。
翻错了模型看到的是蓝红互换的世界，而且**不会报任何错**，只是动作全乱。
"""

from __future__ import annotations

import functools
import io
import socket as _socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import msgpack
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# msgpack-numpy：逐字搬自 openpi_client/msgpack_numpy.py。
# 不要"优化"这里的键名或字段顺序 —— 它是与机器人 PC 之间的线协议契约。
# ──────────────────────────────────────────────────────────────────────────────


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


# ──────────────────────────────────────────────────────────────────────────────
# 帧收发
# ──────────────────────────────────────────────────────────────────────────────


def _send(sock, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed by RobotArmService")
        buf += chunk
    return bytes(buf)


def _recv(sock) -> bytes:
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    return _recv_exact(sock, length)


# ──────────────────────────────────────────────────────────────────────────────
# 客户端
# ──────────────────────────────────────────────────────────────────────────────


class RobotArmClient:
    """RobotArmService 的同步客户端。

    一条长连接服务整个部署进程；多个 episode 靠 `reset()` 切换。

    Args:
        endpoint:       "tcp://host:port"，本机就是 "tcp://127.0.0.1:9900"
        recv_timeout_s: socket 收超时（秒）。`reset` 服务端会 sleep 5s 走初始位姿，
                        所以别设得比 10s 还小
        decode_images:  False 时 obs["images"] 保持 JPEG 字节数组不解码（benchmark 用）
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:9900",
        recv_timeout_s: float = 30.0,
        decode_images: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self._recv_timeout_s = float(recv_timeout_s)
        self._decode_images = bool(decode_images)
        self._packer = Packer()
        self._sock: Optional[_socket.socket] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        # 统计：TCP 往返 / JPEG 解码耗时，控制环用来定位卡在哪
        self.stat_rtt_s = 0.0
        self.stat_decode_s = 0.0
        self.stat_calls = 0

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def connect(self) -> "RobotArmClient":
        host, port = self.endpoint.replace("tcp://", "").rsplit(":", 1)
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(self._recv_timeout_s)
        sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        try:
            sock.connect((host, int(port)))
        except OSError as exc:
            sock.close()
            raise ConnectionError(
                f"连不上 RobotArmService {self.endpoint}: {exc}\n"
                "  机器人 PC 上先起: python scripts/deploy/start_robot_arm_service.py --port 9900"
            ) from exc
        self._sock = sock
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="jpegdec")

        reply = self._request({"cmd": "ping"})
        if not reply.get("pong"):
            self.close(send_stop=False)
            raise ConnectionError(f"ping 回复异常: {reply}")
        return self

    def close(self, send_stop: bool = False) -> None:
        """断开连接。

        send_stop=True 会让服务端 `go_zero()` 把双臂打到全零位 —— 从弯曲姿态出发
        可能扫过整个工作区撞到箱子，所以**默认不发**，只断开（手臂保持当前位姿通电驻留）。
        """
        if self._sock is not None and send_stop:
            try:
                self._request({"cmd": "stop"})
            except Exception as exc:
                print(f"[robot] stop 失败（忽略）: {exc}")
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __enter__(self) -> "RobotArmClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ── 命令 ─────────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """服务端走到初始位姿（内部 sleep 5s），返回第一帧观测。**这是真实运动。**"""
        return self._decode_obs(self._request({"cmd": "reset"})["obs"])

    def try_obs(self) -> Optional[Dict[str, Any]]:
        """只读一帧观测，**不动手臂**。服务端不支持则返回 None。

        当前上游 `RobotArmService._handle` 只有 ping / reset / step / stop 四个命令，
        **没有纯读接口** —— 也就是说"连上去看一眼观测但不碰手臂"这件事做不到：
        `reset` 会以 20% 速度走初始位姿，`step` 必然下发 JointCtrl。

        想要的话在机器人 PC 的 `openpi/src/openpi/serving/robot_arm_service.py`
        的 `_handle()` 里加两行（纯读，零风险）：

            if cmd == "obs":
                return {"obs": self._get_obs()}

        加了之后本方法就能用；没加则返回 None，由调用方决定是否退到 `reset()`。
        """
        try:
            reply = self._request({"cmd": "obs"})
        except RuntimeError as exc:
            if "unknown cmd" in str(exc):
                return None
            raise
        return self._decode_obs(reply["obs"])

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """下发一条 14 维绝对指令（关节弧度 + 夹爪**米**），返回新观测。

        ⚠️ 返回的 obs 是 `send_command()` **之后立刻**采的，手臂还没走到目标位置，
        即它是"当前实测状态"。这与训练时 `observation.state` 的语义一致
        （实测状态 -> 未来 32 步指令），所以直接拿来当 proprio 是对的。
        但如果中间停顿过（比如同步模式的推理阻塞），这个 obs 就**过期**了 ——
        必须重发一次保持指令再取一次新 obs，见 deploy_real.py 的 settle_steps。
        """
        act = np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(-1))
        if act.shape != (14,):
            raise ValueError(f"action 必须是 14 维，得到 {act.shape}")
        reply = self._request({"cmd": "step", "action": act})
        return (
            self._decode_obs(reply["obs"]),
            bool(reply.get("done", False)),
            dict(reply.get("info", {})),
        )

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("客户端未连接，先调用 connect()")
        t0 = time.perf_counter()
        _send(self._sock, self._packer.pack(msg))
        raw = _recv(self._sock)
        self.stat_rtt_s += time.perf_counter() - t0
        self.stat_calls += 1
        reply = unpackb(raw)
        if "error" in reply:
            raise RuntimeError(f"RobotArmService 报错: {reply['error']}")
        return reply

    def _decode_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # ⚠️ msgpack 解出来的 ndarray 是用 `np.ndarray(buffer=<bytes>)` 构造的，
        # 而 bytes 是不可变的 -> **数组只读**。torch.as_tensor 对只读数组会告警，
        # 且下游任何原地写都是未定义行为。所以这里一律 copy 成可写数组（就 14 个 float，不心疼）。
        out = dict(obs)
        out["state"] = np.array(obs["state"], dtype=np.float32).reshape(-1)
        out["gripper_position"] = np.array(obs["gripper_position"], dtype=np.float32).reshape(-1)
        if self._decode_images:
            t0 = time.perf_counter()
            out["images"] = self._decode_jpegs(obs["images"])
            self.stat_decode_s += time.perf_counter() - t0
        return out

    def _decode_jpegs(self, images: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """JPEG -> HWC uint8 **RGB**。三路并发解码，串行约 12ms，并发约 6ms。

        ⚠️ 不翻通道 —— 见模块 docstring 顶部的说明。
        """

        def one(item):
            name, jpeg = item
            if jpeg is None:
                raise RuntimeError(
                    f"相机 '{name}' 没有数据（service 端 ROS2 topic 没上来?）"
                )
            buf = jpeg.tobytes() if isinstance(jpeg, np.ndarray) else bytes(jpeg)
            with Image.open(io.BytesIO(buf)) as im:
                # 用 np.array(copy) 而非 np.asarray:PIL 给的 buffer 是只读的，
                # torch.from_numpy 对只读数组会告警（且写入是未定义行为）。
                arr = np.array(im.convert("RGB"), dtype=np.uint8)
            return name, arr

        items = list(images.items())
        assert self._pool is not None
        return dict(self._pool.map(one, items))


# ──────────────────────────────────────────────────────────────────────────────
# 独立自检：连服务、取一帧观测、打印形状与单位。
# 优先走只读的 {"cmd":"obs"}（不动手臂）；服务端没这个命令时需要 --allow-reset。
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="RobotArmService 连通性自检",
        epilog="默认只读。服务端若没加 {'cmd':'obs'} 命令，取观测就只能靠 reset，"
               "那会让手臂走到初始位姿 —— 需显式加 --allow-reset。",
    )
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:9900")
    ap.add_argument("--allow-reset", action="store_true",
                    help="允许在服务端不支持只读 obs 时退到 reset（**手臂会运动**）")
    args = ap.parse_args()

    with RobotArmClient(args.endpoint) as cli:
        print(f"[ok] ping 通过 {args.endpoint}")
        obs = cli.try_obs()
        if obs is None:
            print("[i] 服务端不支持只读 {'cmd':'obs'}（见 RobotArmClient.try_obs 的说明）")
            if not args.allow_reset:
                print("[i] 未加 --allow-reset，就此结束（不碰手臂）。")
                raise SystemExit(0)
            print("[!] 退到 reset —— 手臂将以 20% 速度走到初始位姿，约 5 s")
            obs = cli.reset()
        else:
            print("[ok] 走只读路径，手臂未动")

        print(f"state            {obs['state'].shape} rad   {np.round(obs['state'], 4)}")
        g = obs["gripper_position"]
        print(f"gripper_position {g.shape} m     {np.round(g, 5)}  -> 分数(/0.07) {np.round(g / 0.07, 4)}")
        for k, v in obs["images"].items():
            print(f"images[{k:16s}] {v.shape} {v.dtype}  均值 {v.mean():.1f}")
        print(f"prompt(service)  {obs.get('prompt')!r}   <- 本部署忽略它")
