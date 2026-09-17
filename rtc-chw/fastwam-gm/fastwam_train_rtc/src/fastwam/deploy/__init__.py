"""FastWAM 真机部署的公共层。

三个使用方共用这里的实现，避免 deploy_real.py 顶部那六条「错了不报错、只是动作全乱」
的契约漂移成三份：

    scripts/deploy_real.py      单进程(原有，sync 路径已真机验证)
    scripts/fastwam_server.py   GPU 侧推理服务
    scripts/fastwam_client.py   控制环 + 安全层(无 torch)

分层规则:`layout` 与 `control` **不 import torch**，`policy` 才需要。
client 只碰前两个。
"""

from .layout import (
    ACT_GRIP_IDX,
    ACT_JOINT_IDX,
    ACTION_LAYOUT_DOC,
    DEFAULT_INSTRUCTION,
    GRIPPER_TRAVEL_M,
    PROP_GRIP_SLICE,
    PROP_JOINT_SLICE,
    STATE_LAYOUT_DOC,
    describe_action,
    grip_frac_to_m,
    grip_m_to_frac,
    obs_to_physical,
    split_state14,
    to_action_layout,
    to_service_action,
    to_state_layout,
)

__all__ = [
    "ACT_GRIP_IDX",
    "ACT_JOINT_IDX",
    "ACTION_LAYOUT_DOC",
    "DEFAULT_INSTRUCTION",
    "GRIPPER_TRAVEL_M",
    "PROP_GRIP_SLICE",
    "PROP_JOINT_SLICE",
    "STATE_LAYOUT_DOC",
    "describe_action",
    "grip_frac_to_m",
    "grip_m_to_frac",
    "obs_to_physical",
    "split_state14",
    "to_action_layout",
    "to_service_action",
    "to_state_layout",
]
