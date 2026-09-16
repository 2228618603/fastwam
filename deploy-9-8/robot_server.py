"""
FastWAM inference server for real-robot deployment (Agilex Cobot Magic / Piper SDK).

Runs on the GPU server. Loads a trained FastWAM checkpoint and serves action
chunks over websocket + msgpack, so the robot computer needs no torch/CUDA.

Design notes
------------
- Inference logic mirrors the official FastWAM real-robot reference path
  (same ``model.infer_action`` call, same normalize/denormalize path via
  ``FastWAMProcessor`` + ``dataset_stats.json``, same 3-camera composite).
- The camera composite MUST match training exactly (see
  ``configs/data/agilex_empty_box.yaml`` +
  ``RobotVideoDataset._get``, ``concat_multi_camera="robotwin"``):
      top    -> resized to 320x256 (WxH)
      lwrist -> resized to 160x128, rwrist -> resized to 160x128
      bottom = [left | right]  (concat on width -> 320x128)
      image  = [top ; bottom]  (concat on height -> 320x384, i.e. HxW = 384x320)
  then scaled to [-1, 1].
- State/action layout for this dataset (14D, absolute joint angles in radians):
      action[0:6]   left arm joints
      action[6]     left gripper   (0~1 normalized opening fraction)
      action[7:13]  right arm joints
      action[13]    right gripper  (0~1 normalized opening fraction)
  Proprio state fed to the model is also 14D and must be assembled in the SAME
  order the training config declared: shape_meta.state = [joint(12D), gripper(2D)],
  i.e. [left_j0..j5, right_j0..j5, left_gripper, right_gripper].
  NOTE this differs from the action layout - the client must send `state` in the
  training order, not the action order. See ``STATE_LAYOUT_DOC``.

Protocol (msgpack over websocket, one request -> one response)
-------------------------------------------------------------
Request:
    {"reset": True}                                  -> {"ok": True}
    {"ping": True}                                   -> {"ok": True, "info": {...}}
    {"obs": {"top": <HWC uint8 RGB>,
             "left_wrist": <HWC uint8 RGB>,
             "right_wrist": <HWC uint8 RGB>},
     "state": <14 floats, training order>,
     "instruction": "<optional, defaults to the trained task text>"}
        -> {"ok": True, "action": <[T,14] float32 absolute joint radians + 0~1 grippers>,
            "action_horizon": T, "infer_s": float}

Images are sent at native resolution (uint8 HWC RGB); the server does all
resizing/compositing so the client cannot silently break preprocessing.

Usage
-----
    python deploy-9-8/robot_server.py \
        --ckpt /mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_015000.pt \
        --dataset-stats /mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json \
        --dataset-dir /mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711 \
        --task agilex_empty_box_uncond_3cam384 \
        --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fastwam_server")

# Order in which the 14D proprio vector must be assembled, matching
# configs/data/agilex_empty_box.yaml -> shape_meta.state
# ([joint 12D, gripper 2D]) and ConcatLeftAlign's concat order.
STATE_LAYOUT_DOC = (
    "state[0:6]=left arm joints (rad), state[6:12]=right arm joints (rad), "
    "state[12]=left gripper (0~1), state[13]=right gripper (0~1)"
)
ACTION_LAYOUT_DOC = (
    "action[0:6]=left arm joints (rad), action[6]=left gripper (0~1), "
    "action[7:13]=right arm joints (rad), action[13]=right gripper (0~1)"
)

CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
# (width, height) targets, matching the training-time robotwin composite
TOP_SIZE_WH = (320, 256)
WRIST_SIZE_WH = (160, 128)
DEFAULT_CKPT = Path(
    "/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/"
    "checkpoints/weights/step_015000.pt"
)
DEFAULT_DATASET_STATS = Path("/mnt/data/chw/fastwam/dataset_stats/agilex_empty_box/train_stats.json")
DEFAULT_DATASET_DIR = Path(
    "/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711"
)
DEFAULT_ACTION_DIT = Path("/mnt/data/chw/fastwam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Path to trained weights .pt.")
    p.add_argument("--dataset-stats", default=str(DEFAULT_DATASET_STATS), help="Path to dataset_stats.json from the training run.")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                   help="Training dataset directory used for task text lookup and config consistency.")
    p.add_argument("--action-dit", default=str(DEFAULT_ACTION_DIT),
                   help="ActionDiT backbone path used when constructing the model.")
    p.add_argument("--task", default="agilex_empty_box_uncond_3cam384", help="Hydra task config name.")
    p.add_argument("--configs-dir", default=str(PROJECT_ROOT / "configs"), help="FastWAM configs dir.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--action-horizon", type=int, default=None,
                   help="Defaults to num_frames-1 from the data config (32 for this dataset).")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cuda-memory-fraction", type=float, default=None,
                   help="Optional torch.cuda per-process memory fraction, useful for low-impact smoke tests.")
    p.add_argument("--instruction", default=None,
                   help="Default task text; if omitted, read from the dataset's meta/tasks.jsonl if available.")
    p.add_argument("--self-test", action="store_true",
                   help="Load the model, run one inference on synthetic input, print shapes, and exit. "
                        "No network serving, no robot involved.")
    return p.parse_args()


class FastWAMActionServer:
    """Loads FastWAM and turns (3 images + 14D state) into a [T,14] action chunk."""

    def __init__(
        self,
        ckpt: str,
        dataset_stats: str,
        dataset_dir: str,
        action_dit: str,
        task: str,
        configs_dir: str,
        device: str,
        mixed_precision: str,
        action_horizon: Optional[int],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        cuda_memory_fraction: Optional[float],
        instruction: Optional[str],
    ):
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self._torch = torch
        self._DEFAULT_PROMPT = DEFAULT_PROMPT

        ckpt_path = Path(ckpt).expanduser().resolve()
        stats_path = Path(dataset_stats).expanduser().resolve()
        dataset_path = Path(dataset_dir).expanduser().resolve()
        action_dit_path = Path(action_dit).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset_stats.json not found: {stats_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset dir not found: {dataset_path}")
        if not action_dit_path.exists():
            raise FileNotFoundError(f"ActionDiT backbone not found: {action_dit_path}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(Path(configs_dir).resolve())):
            cfg = compose(config_name="train", overrides=[f"task={task}"])
        cfg.data.train.dataset_dirs = [str(dataset_path)]
        cfg.data.val.dataset_dirs = [str(dataset_path)]
        cfg.data.train.pretrained_norm_stats = str(stats_path)
        cfg.data.val.pretrained_norm_stats = str(stats_path)
        self.cfg = cfg

        # The checkpoint's dataset_stats.json is the ground truth of the
        # training-time state layout (this dataset: [joint 12D, gripper 2D]).
        # The yaml config in the repo may declare a different split (e.g. a
        # single "default" 14D state), which makes the normalizer fail with
        # KeyError. Rebuild the state shape_meta from the stats so per-component
        # normalization matches training exactly.
        import json as _json
        from omegaconf import OmegaConf

        _stats_raw = _json.loads(stats_path.read_text())
        _state_meta = []
        for _key, _comp in _stats_raw["state"].items():
            _dim = len(_comp["global_mean"])
            _state_meta.append({"key": _key, "raw_shape": _dim, "shape": _dim})
        cfg.data.train.shape_meta.state = OmegaConf.create(_state_meta)
        logger.info("State shape_meta rebuilt from %s: %s", stats_path.name, _state_meta)

        dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed_precision]
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU (this will be very slow).")
            device = "cpu"
        self.device = device
        if device.startswith("cuda") and cuda_memory_fraction is not None:
            memory_device = torch.device(device if ":" in device else "cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(cuda_memory_fraction), device=memory_device)
            logger.info("Set CUDA per-process memory fraction to %.3f on %s", float(cuda_memory_fraction), memory_device)

        # Real-robot inference needs the text encoder (we encode the instruction
        # online rather than relying on the precomputed training cache), matching
        # the official RoboTwin deploy policy which also forces this to True.
        # Convert OmegaConf's struct-mode DictConfig to a regular dict so the
        # compatibility cleanup below can remove an unsupported optional key.
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if not isinstance(model_cfg, dict):
            raise TypeError(f"Expected cfg.model to convert to dict, got {type(model_cfg)}")
        model_cfg["load_text_encoder"] = True
        model_cfg["action_dit_pretrained_path"] = str(action_dit_path)

        # The final_A non-RTC checkpoint was trained with a newer config that
        # contains `rtc: {enabled: false}`. Older local runtime factories do not
        # accept that keyword even though disabled RTC has no effect on the
        # model path. Drop it only for disabled RTC and fail loudly for an RTC
        # checkpoint instead of silently loading an incompatible model.
        if "rtc" in model_cfg:
            rtc_cfg = model_cfg["rtc"]
            rtc_enabled = bool(rtc_cfg.get("enabled", False))
            target_path = str(model_cfg.get("_target_", ""))
            try:
                module_name, factory_name = target_path.rsplit(".", 1)
                factory = getattr(importlib.import_module(module_name), factory_name)
                supports_rtc = "rtc" in inspect.signature(factory).parameters
            except (ImportError, AttributeError, ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Could not inspect model factory {target_path!r} while handling RTC config"
                ) from exc

            if not supports_rtc:
                if rtc_enabled:
                    raise RuntimeError(
                        "This deployment runtime does not support RTC, but the selected "
                        "model config has rtc.enabled=true. Use the matching newer "
                        "FastWAM source tree for this checkpoint."
                    )
                del model_cfg["rtc"]
                logger.warning(
                    "Ignoring disabled `model.rtc` config because the active runtime "
                    "factory %s has no rtc parameter.",
                    target_path,
                )

        logger.info("Loading model (this pulls the Wan2.2 base weights from checkpoints/)...")
        self.model = instantiate(model_cfg, model_dtype=dtype, device=device)
        self.model.load_checkpoint(str(ckpt_path))
        self.model = self.model.to(device).eval()
        logger.info("Loaded trained checkpoint: %s", ckpt_path)

        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        logger.info("Loaded normalization stats: %s", stats_path)

        num_frames = int(cfg.data.train.num_frames)
        self.action_horizon = int(action_horizon) if action_horizon else num_frames - 1
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed

        self.instruction = instruction or self._read_task_text()
        logger.info("Default instruction: %r", self.instruction)
        logger.info("action_horizon=%d num_inference_steps=%d", self.action_horizon, self.num_inference_steps)
        logger.info("Expected state layout: %s", STATE_LAYOUT_DOC)
        logger.info("Returned action layout: %s", ACTION_LAYOUT_DOC)

        self._infer_count = 0

    def _read_task_text(self) -> str:
        """Read the single task string from the training dataset's tasks.jsonl."""
        import json
        for ds_dir in self.cfg.data.train.dataset_dirs:
            p = (PROJECT_ROOT / str(ds_dir)).resolve() / "meta" / "tasks.jsonl"
            if p.exists():
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            return json.loads(line)["task"]
        raise ValueError(
            "Could not read task text from tasks.jsonl; pass --instruction explicitly."
        )

    @staticmethod
    def _resize_rgb(image: np.ndarray, size_wh) -> np.ndarray:
        from PIL import Image
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
        return np.asarray(
            Image.fromarray(image.astype(np.uint8), mode="RGB").resize(size_wh, resample=Image.BILINEAR),
            dtype=np.uint8,
        )

    def _build_image_tensor(self, images: Dict[str, np.ndarray]):
        """Compose the 3 cameras exactly as training did -> [1,3,384,320] in [-1,1]."""
        missing = [k for k in CAMERA_KEYS if k not in images]
        if missing:
            raise ValueError(f"Missing camera(s): {missing}; required: {list(CAMERA_KEYS)}")
        top = self._resize_rgb(images["top"], TOP_SIZE_WH)               # 256x320
        left = self._resize_rgb(images["left_wrist"], WRIST_SIZE_WH)     # 128x160
        right = self._resize_rgb(images["right_wrist"], WRIST_SIZE_WH)   # 128x160
        bottom = np.concatenate([left, right], axis=1)                   # 128x320
        composite = np.concatenate([top, bottom], axis=0)                # 384x320
        t = self._torch.from_numpy(composite).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        return t * (2.0 / 255.0) - 1.0

    def _normalize_state(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14:
            raise ValueError(f"state must be 14D ({STATE_LAYOUT_DOC}), got {state.shape[0]}D")
        meta = self.processor.shape_meta["state"]

        # `shape_meta['state']` may list SEVERAL named components (this dataset:
        # [joint 12D, gripper 2D]) that get concatenated by `action_state_merger`
        # only *after* per-component normalization - mirroring the exact order
        # used in FastWAMProcessor.preprocess(): transform -> normalize -> merge.
        # (The official RoboTwin/LIBERO reference implementation only ever had a
        # single state component, so it never needed this split/merge step.)
        state_dict: Dict[str, Any] = {}
        idx = 0
        for m in meta:
            k, raw_shape = m["key"], int(m["raw_shape"])
            state_dict[k] = self._torch.as_tensor(state[idx: idx + raw_shape]).unsqueeze(0)
            idx += raw_shape
        if idx != state.shape[0]:
            raise ValueError(
                f"shape_meta['state'] components sum to {idx}D but got a {state.shape[0]}D input"
            )

        batch = {"state": state_dict}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return batch["state"]

    def _denormalize_action(self, action) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        meta = self.processor.shape_meta["action"]
        if len(meta) != 1:
            raise ValueError(
                f"Expected one merged action key, got {[m['key'] for m in meta]}"
            )
        normalizer = self.processor.normalizer.normalizers["action"][meta[0]["key"]]
        out = normalizer.backward(action.to(dtype=self._torch.float32, device="cpu"))
        return out.numpy()[0]  # [T, 14]

    def infer(self, images: Dict[str, np.ndarray], state: np.ndarray,
              instruction: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        image_tensor = self._build_image_tensor(images)
        proprio = self._normalize_state(state)
        prompt = self._DEFAULT_PROMPT.format(task=instruction or self.instruction)

        with self._torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=image_tensor,
                action_horizon=self.action_horizon,
                proprio=proprio,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
            )
        chunk = self._denormalize_action(pred["action"]).astype(np.float32)
        if chunk.shape[1] != 14:
            raise RuntimeError(f"Model returned {chunk.shape[1]}D actions, expected 14D")
        dt = time.perf_counter() - t0
        self._infer_count += 1
        logger.info("infer #%d -> chunk %s in %.3fs", self._infer_count, chunk.shape, dt)
        return {"action": chunk, "action_horizon": int(chunk.shape[0]), "infer_s": dt}

    def info(self) -> Dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "state_layout": STATE_LAYOUT_DOC,
            "action_layout": ACTION_LAYOUT_DOC,
            "camera_keys": list(CAMERA_KEYS),
            "instruction": self.instruction,
            "control_hz_from_training_fps": 30,
            "infer_count": self._infer_count,
        }


def _self_test(server: FastWAMActionServer) -> int:
    """Run one inference on synthetic data to prove the whole path works."""
    logger.info("=== SELF TEST (synthetic input, no robot, no network) ===")
    rng = np.random.default_rng(0)
    images = {
        "top": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "left_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "right_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    # A plausible resting state: small joint angles, grippers closed.
    state = np.zeros(14, dtype=np.float32)
    out = server.infer(images, state)
    chunk = out["action"]
    logger.info("action chunk shape: %s dtype=%s", chunk.shape, chunk.dtype)
    logger.info("first action row: %s", np.round(chunk[0], 4).tolist())
    logger.info("left  joints min/max: %.4f / %.4f", chunk[:, 0:6].min(), chunk[:, 0:6].max())
    logger.info("right joints min/max: %.4f / %.4f", chunk[:, 7:13].min(), chunk[:, 7:13].max())
    logger.info("left  gripper min/max: %.4f / %.4f", chunk[:, 6].min(), chunk[:, 6].max())
    logger.info("right gripper min/max: %.4f / %.4f", chunk[:, 13].min(), chunk[:, 13].max())
    logger.info("infer time: %.3fs", out["infer_s"])
    logger.info("=== SELF TEST PASSED ===")
    return 0


async def _serve(server: FastWAMActionServer, host: str, port: int):
    import msgpack
    import msgpack_numpy
    import websockets

    msgpack_numpy.patch()

    async def handler(websocket):
        peer = getattr(websocket, "remote_address", "?")
        logger.info("client connected: %s", peer)
        try:
            async for raw in websocket:
                try:
                    req = msgpack.unpackb(raw, raw=False)
                    if req.get("ping"):
                        resp = {"ok": True, "info": server.info()}
                    elif req.get("reset"):
                        logger.info("reset requested")
                        resp = {"ok": True}
                    elif "obs" in req:
                        out = server.infer(
                            images=req["obs"],
                            # msgpack-numpy may decode state as a read-only view;
                            # make an owned array before handing it to torch.
                            state=np.array(req["state"], dtype=np.float32, copy=True),
                            instruction=req.get("instruction"),
                        )
                        resp = {"ok": True, **out}
                    else:
                        resp = {"ok": False, "error": f"unknown request keys: {list(req)}"}
                except Exception as exc:  # keep the connection alive on bad requests
                    logger.error("request failed: %s\n%s", exc, traceback.format_exc())
                    resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                await websocket.send(msgpack.packb(resp, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    logger.info("serving on ws://%s:%d", host, port)
    # ping_interval=None: a single inference can take longer than the default
    # websocket ping timeout, which would otherwise drop the connection.
    async with websockets.serve(handler, host, port, ping_interval=None, max_size=None):
        await asyncio.Future()


def main() -> int:
    args = _parse_args()
    server = FastWAMActionServer(
        ckpt=args.ckpt,
        dataset_stats=args.dataset_stats,
        dataset_dir=args.dataset_dir,
        action_dit=args.action_dit,
        task=args.task,
        configs_dir=args.configs_dir,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        cuda_memory_fraction=args.cuda_memory_fraction,
        instruction=args.instruction,
    )
    if args.self_test:
        return _self_test(server)
    asyncio.run(_serve(server, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
