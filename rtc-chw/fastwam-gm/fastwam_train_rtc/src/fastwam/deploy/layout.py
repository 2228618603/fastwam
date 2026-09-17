"""14 维排布与单位换算 —— **纯 numpy，不 import torch**。

这个模块是部署链路上唯一的「契约真值来源」。抽出来的原因很实在：
`scripts/deploy_real.py`(单进程)、`scripts/fastwam_server.py`(GPU 侧)、
`scripts/fastwam_client.py`(控制环侧) 三处都要用同一套排布和换算，
而这些常量错了**大多不报错，只是动作全乱**（见 deploy_real.py 顶部的六条契约）。
复制粘贴三份的话，迟早漂移。

不 import torch 是硬要求：`fastwam_client.py` 要跑在没有 torch 的机器人 PC 环境里
（或至少不该为了几个下标去加载 2 GB 的 torch）。
"""

from __future__ import annotations

import numpy as np

# ── 两套 14 维排布(deploy_real.py 契约 1)──────────────────────────────────────
#   proprio = [12 关节(左 j0-5, 右 j0-5), 2 夹爪(左, 右)]      <- 夹爪在末尾
#   action  = [左 j0-5, 左夹爪, 右 j0-5, 右夹爪]               <- 夹爪插在 6 / 13
# openpi RobotArmService 的 step 用的正是 action 排布(robot_arm_service.py:467-468)。
ACT_JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
ACT_GRIP_IDX = np.array([6, 13], dtype=np.int64)
# proprio 里的排布(ConcatLeftAlign 按 shape_meta 顺序拼:joint(12) 然后 gripper_position(2))
PROP_JOINT_SLICE = slice(0, 12)
PROP_GRIP_SLICE = slice(12, 14)

# 夹爪满开度(米)。训练数据是 0~1 归一化,openpi service 两侧都用米。已确认 = 0.07。
# ⚠️ 别抄 GWP 的 0.105：那台机器的硬件行程上限是 GRIPPER_RAW_MAX=70000 raw = 0.07 m,
#    用 0.105 会让 0.667 以上的模型输出全部塌缩成"全开"(GWP 自己也标着"待重新验证")。
#    0.07 * 1e6 = 70000 正好对齐硬件上限。
GRIPPER_TRAVEL_M = 0.07

# 数据集 meta/tasks.jsonl 里那条唯一的任务描述。必须**逐字一致**,
# 否则 T5 embedding 缓存的 sha256 对不上(缓存目录里只有这一条)。
DEFAULT_INSTRUCTION = (
    "Alternately use the left arm and the right arm to pick up the goods from the nearby box "
    "and place them in the distant box, until the nearby box is empty."
)

STATE_LAYOUT_DOC = (
    "state[0:12]=joints(rad, 左 j0-5 然后 右 j0-5), state[12]=左夹爪(0~1), "
    "state[13]=右夹爪(0~1)  [训练序 = shape_meta.state 的 joint(12)+gripper_position(2)]"
)
ACTION_LAYOUT_DOC = (
    "action[0:6]=左臂关节(rad), action[6]=左夹爪(0~1), "
    "action[7:13]=右臂关节(rad), action[13]=右夹爪(0~1)  [6+1+6+1]"
)


# ══════════════════════════════════════════════════════════════════ 单位换算
def grip_m_to_frac(m, travel_m: float = GRIPPER_TRAVEL_M) -> np.ndarray:
    """夹爪:米 -> 训练用的 0~1 分数。"""
    return np.asarray(m, dtype=np.float32) / float(travel_m)


def grip_frac_to_m(frac, travel_m: float = GRIPPER_TRAVEL_M) -> np.ndarray:
    """夹爪:0~1 分数 -> 米。"""
    return np.asarray(frac, dtype=np.float32) * float(travel_m)


# ══════════════════════════════════════════════════════════════════ 排布转换
def obs_to_physical(obs: dict, travel_m: float = GRIPPER_TRAVEL_M) -> tuple[np.ndarray, np.ndarray]:
    """openpi service 的 obs -> (12 维关节弧度, 2 维夹爪分数),即训练数据的物理量。"""
    joint = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if joint.shape != (12,):
        raise ValueError(f"obs['state'] 应为 12 维,得到 {joint.shape}")
    grip = grip_m_to_frac(np.asarray(obs["gripper_position"]).reshape(-1), travel_m)
    if grip.shape != (2,):
        raise ValueError(f"obs['gripper_position'] 应为 2 维,得到 {grip.shape}")
    return joint, grip


def to_action_layout(joint12: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """(12 关节, 2 夹爪) -> action 排布的 14 维(夹爪插在 6 / 13)。"""
    out = np.zeros(14, dtype=np.float32)
    out[ACT_JOINT_IDX] = np.asarray(joint12, dtype=np.float32).reshape(-1)
    out[ACT_GRIP_IDX] = np.asarray(grip2, dtype=np.float32).reshape(-1)
    return out


def to_state_layout(joint12: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """(12 关节, 2 夹爪) -> proprio/state 排布的 14 维(夹爪在末尾)。

    这是发给 server 的 `state` 的线上格式,见 STATE_LAYOUT_DOC。
    """
    return np.concatenate([
        np.asarray(joint12, dtype=np.float32).reshape(12),
        np.asarray(grip2, dtype=np.float32).reshape(2),
    ]).astype(np.float32)


def split_state14(state14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """state 排布的 14 维 -> (12 关节, 2 夹爪分数)。`to_state_layout` 的逆。"""
    s = np.asarray(state14, dtype=np.float32).reshape(-1)
    if s.shape != (14,):
        raise ValueError(f"state 必须是 14 维({STATE_LAYOUT_DOC}),得到 {s.shape}")
    return s[PROP_JOINT_SLICE].copy(), s[PROP_GRIP_SLICE].copy()


def to_service_action(a14: np.ndarray, travel_m: float = GRIPPER_TRAVEL_M) -> np.ndarray:
    """action 排布的物理量 -> openpi service 的单位(夹爪分数换成米,并夹到物理可达范围)。"""
    out = np.asarray(a14, dtype=np.float32).reshape(-1).copy()
    if out.shape != (14,):
        raise ValueError(f"action 必须是 14 维,得到 {out.shape}")
    out[ACT_GRIP_IDX] = np.clip(out[ACT_GRIP_IDX], 0.0, 1.0) * float(travel_m)
    return out


def describe_action(a14: np.ndarray) -> str:
    """人眼可读的一行摘要(日志用)。"""
    a = np.asarray(a14, dtype=np.float32).reshape(-1)
    return (f"L_j={np.round(a[0:6], 4).tolist()} L_g={a[6]:.4f} | "
            f"R_j={np.round(a[7:13], 4).tolist()} R_g={a[13]:.4f}")
