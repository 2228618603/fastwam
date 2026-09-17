"""LeRobot adapter for the Agilex Magic dual-arm dataset.

The dataset uses ``observation.image.*``/``actions`` while the generic
FastWAM reader expects ``observation.images.*``/``action``. This adapter
keeps the source columns intact and exposes the logical FastWAM features.
"""

from __future__ import annotations

import torch

from .base_lerobot_dataset import BaseLerobotDataset
from .robot_video_dataset import RobotVideoDataset


class AgilexDataset(BaseLerobotDataset):
    """Map Agilex video, state and action columns to FastWAM conventions."""

    _IMAGE_KEYS = {
        "top": "observation.image.top",
        "left_wrist": "observation.image.left_wrist",
        "right_wrist": "observation.image.right_wrist",
    }

    def _get_source_keys(self, kind: str, key: str, default_key: str) -> list[str]:
        if kind == "image":
            try:
                return [self._IMAGE_KEYS[key]]
            except KeyError as exc:
                raise KeyError(f"Unsupported Agilex camera key: {key}") from exc
        if kind == "action":
            if key != "default":
                raise KeyError(f"Agilex action must use logical key `default`, got `{key}`")
            return ["actions"]
        if kind == "state":
            if key != "default":
                raise KeyError(f"Agilex state must use logical key `default`, got `{key}`")
            return ["observation.state.joint", "observation.gripper_position"]
        return [default_key]

    @staticmethod
    def _as_sequence(value: torch.Tensor, source_key: str) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(f"Agilex source `{source_key}` must be [T, D], got {tuple(value.shape)}")
        return value

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        source_key = meta["source_keys"][0]
        action = self._as_sequence(lerobot_sample[source_key], source_key)
        if action.shape[-1] != meta["raw_shape"]:
            raise ValueError(
                f"Agilex action `{source_key}` has dim {action.shape[-1]}, expected {meta['raw_shape']}"
            )
        return action.float()

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        source_keys = meta["source_keys"]
        parts = [self._as_sequence(lerobot_sample[key], key) for key in source_keys]
        if len({part.shape[0] for part in parts}) != 1:
            raise ValueError(f"Agilex state sources have inconsistent lengths: {[tuple(x.shape) for x in parts]}")
        state = torch.cat(parts, dim=-1)
        if state.shape[-1] != meta["raw_shape"]:
            raise ValueError(
                f"Agilex state `{source_keys}` has dim {state.shape[-1]}, expected {meta['raw_shape']}"
            )
        return state.float()

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        source_key = meta["source_keys"][0]
        image = lerobot_sample[source_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Agilex image `{source_key}` must be [T, C, H, W], got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"Agilex image `{source_key}` must have 3 channels, got {image.shape[1]}")
        # LeRobot's torch transform returns float RGB in [0, 1].
        if image.is_floating_point():
            image = (image * 255).clamp(0, 255)
        return image.to(torch.uint8)


class AgilexRobotVideoDataset(RobotVideoDataset):
    """FastWAM video dataset wrapper using :class:`AgilexDataset`."""

    base_dataset_cls = AgilexDataset
