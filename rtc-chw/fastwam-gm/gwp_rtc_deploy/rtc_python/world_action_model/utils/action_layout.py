"""Utilities for the 14D dual-arm action/state layout.

Canonical 14D layout used by actions and by newly constructed state:

    [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]

Some LeRobot exports store state split as:

    observation.state.joint      = [left_joints(6), right_joints(6)]
    observation.gripper_position = [left_gripper, right_gripper]

Do not concatenate these as [joint12, grip2]; that was the legacy bug and it
misaligns state with action on delta dimensions.
"""

from __future__ import annotations

import numpy as np


def make_state14_from_joint_gripper(joint, gripper) -> np.ndarray:
    """Build 14D state in action-compatible order from split joint/gripper arrays.

    Supports both a single frame `(12,) + (2,)` and batched arrays
    `(..., 12) + (..., 2)`.
    """
    joint = np.asarray(joint)
    gripper = np.asarray(gripper)
    if joint.shape[-1] != 12:
        raise ValueError(f"expected joint last dim 12, got shape {joint.shape}")
    if gripper.shape[-1] != 2:
        raise ValueError(f"expected gripper last dim 2, got shape {gripper.shape}")
    # Legacy wrong version:
    #   np.concatenate([joint, gripper], axis=-1)
    # would produce [L6 joints, R6 joints, L gripper, R gripper].
    # Correct action-compatible order is [L6, L gripper, R6, R gripper].
    return np.concatenate(
        [
            joint[..., :6],
            gripper[..., :1],
            joint[..., 6:12],
            gripper[..., 1:2],
        ],
        axis=-1,
    )
