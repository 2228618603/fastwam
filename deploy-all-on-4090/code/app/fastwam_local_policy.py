from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from .layout import (
        ACTION_LAYOUT_DOC,
        CAMERA_KEYS,
        CONFIGS_DIR,
        DEFAULT_ACTION_DIT,
        DEFAULT_CKPT,
        DEFAULT_CONTEXT,
        DEFAULT_DATASET_STATS,
        DEFAULT_TASK,
        DEFAULT_TASK_TEXT,
        FALLBACK_ACTION_DIT,
        FALLBACK_CKPT,
        FALLBACK_DATASET_DIR,
        FALLBACK_MODEL_CACHE,
        MODEL_CACHE_DIR,
        PACKAGE_ROOT,
        PACKAGE_MODEL_CACHE_DIR,
        PACKAGE_WEIGHTS_DIR,
        REPO_FALLBACK_ROOT,
        SRC_FALLBACK_ROOT,
        STATE_LAYOUT_DOC,
        TASK_NAME,
        TOP_SIZE_WH,
        WRIST_SIZE_WH,
        prefer_packaged,
    )
except ImportError:  # direct script execution during local debugging
    from layout import (  # type: ignore
        ACTION_LAYOUT_DOC,
        CAMERA_KEYS,
        CONFIGS_DIR,
        DEFAULT_ACTION_DIT,
        DEFAULT_CKPT,
        DEFAULT_CONTEXT,
        DEFAULT_DATASET_STATS,
        DEFAULT_TASK,
        DEFAULT_TASK_TEXT,
        FALLBACK_ACTION_DIT,
        FALLBACK_CKPT,
        FALLBACK_DATASET_DIR,
        FALLBACK_MODEL_CACHE,
        MODEL_CACHE_DIR,
        PACKAGE_ROOT,
        REPO_FALLBACK_ROOT,
        SRC_FALLBACK_ROOT,
        STATE_LAYOUT_DOC,
        TASK_NAME,
        TOP_SIZE_WH,
        WRIST_SIZE_WH,
        prefer_packaged,
    )

logger = logging.getLogger("fastwam_local_policy")


def _install_import_paths() -> None:
    candidates = [
        PACKAGE_ROOT / "src",
        PACKAGE_ROOT,
        SRC_FALLBACK_ROOT,
        REPO_FALLBACK_ROOT,
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class FastWAMLocalPolicy:
    """In-process FastWAM action policy for local robot deployment."""

    def __init__(
        self,
        ckpt: str | Path | None = None,
        dataset_stats: str | Path | None = None,
        fixed_context: str | Path | None = None,
        action_dit: str | Path | None = None,
        configs_dir: str | Path | None = None,
        model_cache: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        task: str = TASK_NAME,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: Optional[int] = None,
        num_inference_steps: int = 4,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        cuda_memory_fraction: Optional[float] = None,
        compile_action_infer: bool = False,
    ):
        _install_import_paths()

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers

        self._torch = torch
        self._OmegaConf = OmegaConf

        self.ckpt_path = (
            Path(ckpt).expanduser()
            if ckpt
            else prefer_packaged(DEFAULT_CKPT, prefer_packaged(PACKAGE_WEIGHTS_DIR / "step_015000.pt", FALLBACK_CKPT))
        )
        self.stats_path = Path(dataset_stats).expanduser() if dataset_stats else DEFAULT_DATASET_STATS
        self.context_path = Path(fixed_context).expanduser() if fixed_context else DEFAULT_CONTEXT
        self.action_dit_path = (
            Path(action_dit).expanduser()
            if action_dit
            else prefer_packaged(
                DEFAULT_ACTION_DIT,
                prefer_packaged(PACKAGE_WEIGHTS_DIR / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt", FALLBACK_ACTION_DIT),
            )
        )
        self.configs_dir = Path(configs_dir).expanduser() if configs_dir else CONFIGS_DIR
        self.dataset_dir = Path(dataset_dir).expanduser() if dataset_dir else FALLBACK_DATASET_DIR
        if model_cache:
            self.model_cache = Path(model_cache).expanduser()
        else:
            external_cache_ready = (
                MODEL_CACHE_DIR / "Wan-AI" / "Wan2.2-TI2V-5B"
            ).exists() and (
                MODEL_CACHE_DIR
                / "DiffSynth-Studio"
                / "Wan-Series-Converted-Safetensors"
                / "Wan2.2_VAE.safetensors"
            ).exists()
            packaged_cache_ready = (
                PACKAGE_MODEL_CACHE_DIR / "Wan-AI" / "Wan2.2-TI2V-5B"
            ).exists() and (
                PACKAGE_MODEL_CACHE_DIR
                / "DiffSynth-Studio"
                / "Wan-Series-Converted-Safetensors"
                / "Wan2.2_VAE.safetensors"
            ).exists()
            if external_cache_ready:
                self.model_cache = MODEL_CACHE_DIR
            elif packaged_cache_ready:
                self.model_cache = PACKAGE_MODEL_CACHE_DIR
            else:
                self.model_cache = FALLBACK_MODEL_CACHE

        for label, path in (
            ("checkpoint", self.ckpt_path),
            ("dataset_stats", self.stats_path),
            ("fixed_context", self.context_path),
            ("ActionDiT", self.action_dit_path),
            ("configs_dir", self.configs_dir),
            ("model_cache", self.model_cache),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(self.model_cache.resolve())
        register_default_resolvers()

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU.")
            device = "cpu"
        self.device = device
        if device.startswith("cuda") and cuda_memory_fraction is not None:
            memory_device = torch.device(device if ":" in device else "cuda:0")
            torch.cuda.set_per_process_memory_fraction(float(cuda_memory_fraction), device=memory_device)
            logger.info("Set CUDA per-process memory fraction %.3f on %s", float(cuda_memory_fraction), memory_device)

        if mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("--mixed-precision must be one of: no, fp16, bf16")
        dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed_precision]

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(self.configs_dir.resolve())):
            cfg = compose(config_name="train", overrides=[f"task={task}"])

        if self.dataset_dir.exists():
            cfg.data.train.dataset_dirs = [str(self.dataset_dir)]
            cfg.data.val.dataset_dirs = [str(self.dataset_dir)]
        cfg.data.train.pretrained_norm_stats = str(self.stats_path)
        cfg.data.val.pretrained_norm_stats = str(self.stats_path)
        self._rebuild_state_meta_from_stats(cfg)
        self.cfg = cfg

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if not isinstance(model_cfg, dict):
            raise TypeError(f"Expected cfg.model as dict, got {type(model_cfg)}")
        model_cfg["load_text_encoder"] = False
        model_cfg["action_dit_pretrained_path"] = str(self.action_dit_path)
        model_cfg["compile_training_denoise"] = False
        model_cfg["mot_checkpoint_mixed_attn"] = False
        self._drop_unsupported_disabled_rtc(model_cfg)

        logger.info("Loading FastWAM without text encoder from %s", self.ckpt_path)
        t0 = time.perf_counter()
        self.model = instantiate(model_cfg, model_dtype=dtype, device=device)
        self.model.load_checkpoint(str(self.ckpt_path))
        self.model = self.model.to(device).eval()
        logger.info("Loaded model in %.2fs", time.perf_counter() - t0)

        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.stats_path)))

        context_payload = torch.load(str(self.context_path), map_location="cpu")
        if isinstance(context_payload, dict):
            context = context_payload.get("context")
            mask = context_payload.get("mask", context_payload.get("context_mask"))
        else:
            raise ValueError(f"Unsupported fixed context format in {self.context_path}")
        if context is None or mask is None:
            raise ValueError(f"Fixed context must contain context and mask: {self.context_path}")
        self.context = context
        self.context_mask = mask

        num_frames = int(cfg.data.train.num_frames)
        self.action_horizon = int(action_horizon) if action_horizon else num_frames - 1
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.compile_action_infer = bool(compile_action_infer)
        self.instruction = self._read_task_text()
        self._infer_count = 0

        logger.info("Context: %s %s, mask %s", tuple(self.context.shape), self.context.dtype, tuple(self.context_mask.shape))
        logger.info("action_horizon=%d num_inference_steps=%d", self.action_horizon, self.num_inference_steps)
        logger.info("Expected state layout: %s", STATE_LAYOUT_DOC)
        logger.info("Returned action layout: %s", ACTION_LAYOUT_DOC)

    def _rebuild_state_meta_from_stats(self, cfg: Any) -> None:
        stats_raw = json.loads(self.stats_path.read_text())
        state_meta = []
        for key, comp in stats_raw["state"].items():
            dim = len(comp["global_mean"])
            state_meta.append({"key": key, "raw_shape": dim, "shape": dim})
        cfg.data.train.shape_meta.state = self._OmegaConf.create(state_meta)
        cfg.data.val.shape_meta.state = cfg.data.train.shape_meta.state

    @staticmethod
    def _drop_unsupported_disabled_rtc(model_cfg: Dict[str, Any]) -> None:
        if "rtc" not in model_cfg:
            return
        rtc_cfg = model_cfg["rtc"]
        rtc_enabled = bool(rtc_cfg.get("enabled", False))
        target_path = str(model_cfg.get("_target_", ""))
        module_name, factory_name = target_path.rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        supports_rtc = "rtc" in inspect.signature(factory).parameters
        if supports_rtc:
            return
        if rtc_enabled:
            raise RuntimeError("Selected runtime does not support RTC but model.rtc.enabled=true.")
        del model_cfg["rtc"]

    def _read_task_text(self) -> str:
        if DEFAULT_TASK_TEXT.exists():
            return DEFAULT_TASK_TEXT.read_text().strip()
        task_file = self.dataset_dir / "meta" / "tasks.jsonl"
        if task_file.exists():
            with task_file.open() as f:
                for line in f:
                    if line.strip():
                        return json.loads(line)["task"]
        return DEFAULT_TASK

    @staticmethod
    def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        from PIL import Image

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
        return np.asarray(
            Image.fromarray(image.astype(np.uint8), mode="RGB").resize(size_wh, resample=Image.BILINEAR),
            dtype=np.uint8,
        )

    def _build_image_tensor(self, images: Dict[str, np.ndarray]):
        missing = [k for k in CAMERA_KEYS if k not in images]
        if missing:
            raise ValueError(f"Missing camera(s): {missing}; required: {list(CAMERA_KEYS)}")
        top = self._resize_rgb(images["top"], TOP_SIZE_WH)
        left = self._resize_rgb(images["left_wrist"], WRIST_SIZE_WH)
        right = self._resize_rgb(images["right_wrist"], WRIST_SIZE_WH)
        bottom = np.concatenate([left, right], axis=1)
        composite = np.concatenate([top, bottom], axis=0)
        t = self._torch.from_numpy(composite).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        return t * (2.0 / 255.0) - 1.0

    def _normalize_state(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14:
            raise ValueError(f"state must be 14D ({STATE_LAYOUT_DOC}), got {state.shape[0]}D")
        meta = self.processor.shape_meta["state"]
        state_dict: Dict[str, Any] = {}
        idx = 0
        for m in meta:
            key, raw_shape = m["key"], int(m["raw_shape"])
            state_dict[key] = self._torch.as_tensor(state[idx: idx + raw_shape]).unsqueeze(0)
            idx += raw_shape
        if idx != state.shape[0]:
            raise ValueError(f"state shape_meta sums to {idx}D but input is {state.shape[0]}D")
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
            raise ValueError(f"Expected one action key, got {[m['key'] for m in meta]}")
        normalizer = self.processor.normalizer.normalizers["action"][meta[0]["key"]]
        out = normalizer.backward(action.to(dtype=self._torch.float32, device="cpu"))
        return out.numpy()[0]

    def infer(self, images: Dict[str, np.ndarray], state14: np.ndarray, instruction: Optional[str] = None) -> np.ndarray:
        del instruction
        t0 = time.perf_counter()
        image_tensor = self._build_image_tensor(images)
        proprio = self._normalize_state(state14)
        with self._torch.no_grad():
            pred = self.model.infer_action(
                prompt=None,
                context=self.context,
                context_mask=self.context_mask,
                input_image=image_tensor,
                action_horizon=self.action_horizon,
                proprio=proprio,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                compile_action_infer=self.compile_action_infer,
            )
        chunk = self._denormalize_action(pred["action"]).astype(np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise RuntimeError(f"Model returned {chunk.shape}, expected [T,14]")
        self._infer_count += 1
        logger.info("infer #%d -> chunk %s in %.3fs", self._infer_count, chunk.shape, time.perf_counter() - t0)
        return chunk

    def reset(self) -> None:
        self._infer_count = 0

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "state_layout": STATE_LAYOUT_DOC,
            "action_layout": ACTION_LAYOUT_DOC,
            "camera_keys": list(CAMERA_KEYS),
            "instruction": self.instruction,
            "load_text_encoder": False,
            "model_cache": str(self.model_cache),
            "infer_count": self._infer_count,
        }
