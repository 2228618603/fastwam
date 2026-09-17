from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PACKAGE_ROOT / "app"
ASSET_DIR = PACKAGE_ROOT / "assets"
CONFIGS_DIR = PACKAGE_ROOT / "configs"

DEFAULT_EXTERNAL_MODEL_ROOT = Path(
    os.environ.get("FASTWAM_MODEL_ROOT", "/media/geekplus/PortableSSD/chw/fastwam-load-giga")
)
WEIGHTS_DIR = DEFAULT_EXTERNAL_MODEL_ROOT / "weights"
MODEL_CACHE_DIR = DEFAULT_EXTERNAL_MODEL_ROOT / "model_cache"
PACKAGE_WEIGHTS_DIR = PACKAGE_ROOT / "weights"
PACKAGE_MODEL_CACHE_DIR = PACKAGE_ROOT / "model_cache"

REPO_FALLBACK_ROOT = Path("/home/chw/code/packages/FastWAM")
SRC_FALLBACK_ROOT = REPO_FALLBACK_ROOT / "src"

FALLBACK_CKPT = Path(
    "/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/"
    "checkpoints/weights/step_035000.pt"
)
FALLBACK_ACTION_DIT = Path(
    "/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
)
FALLBACK_MODEL_CACHE = Path("/mnt/data/chw/fastwam/checkpoints")
FALLBACK_DATASET_DIR = Path(
    "/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711"
)

DEFAULT_CKPT = WEIGHTS_DIR / "step_035000.pt"
DEFAULT_ACTION_DIT = WEIGHTS_DIR / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
DEFAULT_DATASET_STATS = ASSET_DIR / "train_stats.json"
DEFAULT_CONTEXT = ASSET_DIR / "fixed_task_context.pt"
DEFAULT_TASK_TEXT = ASSET_DIR / "task.txt"

TASK_NAME = "agilex_empty_box_uncond_3cam384"
CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
TOP_SIZE_WH = (320, 256)
WRIST_SIZE_WH = (160, 128)

STATE_LAYOUT_DOC = (
    "state[0:6]=left arm joints (rad), state[6:12]=right arm joints (rad), "
    "state[12]=left gripper (0~1), state[13]=right gripper (0~1)"
)
ACTION_LAYOUT_DOC = (
    "action[0:6]=left arm joints (rad), action[6]=left gripper (0~1), "
    "action[7:13]=right arm joints (rad), action[13]=right gripper (0~1)"
)

DEFAULT_TASK = (
    "Alternately use the left arm and the right arm to pick up the goods from "
    "the nearby box and place them in the distant box, until the nearby box is empty."
)


def prefer_packaged(path: Path, fallback: Path | None = None) -> Path:
    if path.exists():
        return path
    if fallback is not None and fallback.exists():
        return fallback
    return path
