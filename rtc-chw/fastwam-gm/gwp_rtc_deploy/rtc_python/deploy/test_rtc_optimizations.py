#!/usr/bin/env python3
"""离线回归测试；默认 CPU/loopback，不导入或连接机械臂。

RTC_TEST_CUDA=1 额外跑微型随机 MoT 的 CUDA Graph 数值回归，不加载真实权重。
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("GIGA_MODELS_LIGHT_IMPORT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
import robot_server as server
import robot_client as client
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rtc_test_pipeline", ROOT / "scripts/inference_openloop.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
from world_action_model.models import CasualWorldActionTransformer_MoT, rtc

torch.set_num_threads(2)


def tiny_model(device="cpu"):
    torch.manual_seed(1)
    model = CasualWorldActionTransformer_MoT(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16,
        in_channels=16, out_channels=16, text_dim=32, freq_dim=32, ffn_dim=64,
        num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        action_expert_dim=32, action_ffn_dim=64, in_action_channels=16,
        out_action_channels=16, num_embodiments=2).eval().to(device)
    model._enable_action_only_prefix_cache = True
    inputs = dict(ref_latents=torch.randn(1, 16, 1, 4, 4, device=device),
                  noisy_latents=torch.zeros(1, 16, 0, 4, 4, device=device),
                  timestep=torch.ones(1, 53, device=device) * 500,
                  encoder_hidden_states=torch.randn(1, 4, 32, device=device),
                  state=torch.zeros(1, 1, 16, device=device),
                  action=torch.randn(1, 48, 16, device=device))
    return model, inputs


class OptimizationTests(unittest.TestCase):
    def test_wire_preprocess_pixel_equivalence(self):
        c = client.ServerClient.__new__(client.ServerClient)
        c.wire_targets = server.WIRE_TARGET_SIZES
        rng = np.random.default_rng(7)
        for h, w in [(480, 640), (721, 1279), (640, 480), (192, 320)]:
            obs = {k: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for k in server.CAMERA_KEYS}
            shrunk = c._shrink(obs)
            np.testing.assert_array_equal(server.compose_tshape(obs), server.compose_tshape(shrunk))
            again = c._shrink(shrunk)
            for k in obs:
                np.testing.assert_array_equal(shrunk[k], again[k])
        self.assertEqual(sum(a.nbytes for a in shrunk.values()), 320 * 384 * 3)

    def test_compile_scope_and_mark_step(self):
        for scope in ("action-blocks", "action-stack"):
            model, _ = tiny_model()
            policy = SimpleNamespace(transformer=model, transformer_2=None)
            with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn) as compile_mock:
                names = pipeline.compile_policy_action_blocks(policy, scope=scope, compile_prefix=True)
            self.assertEqual(compile_mock.call_count, 4 if scope == "action-blocks" else 3)
            self.assertTrue(policy._torch_compile_mark_step)
            self.assertIn("transformer.blocks.forward_prefix_cache[2]", names)

    def test_cached_tensors_survive_output_buffer_mutation(self):
        model, inputs = tiny_model()
        raw_caches = []
        for block in model.blocks:
            original = block.forward_prefix_cache

            def capture(*args, _fn=original, **kwargs):
                result = _fn(*args, **kwargs)
                raw_caches.append(result[2])
                return result

            block.forward_prefix_cache = capture
        with torch.no_grad():
            expected = model(action_only=True, return_dict=False, **inputs).clone()
            cache = model._action_only_prefix_cache
            for raw, saved in zip(raw_caches, cache["blocks"]):
                for key, tensor in saved.items():
                    reference = tensor.clone()
                    self.assertNotEqual(tensor.data_ptr(), raw[key].data_ptr())
                    raw[key].add_(100)  # 模拟 CUDA Graph 下一次 replay 覆写原输出。
                    torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
            actual = model(action_only=True, return_dict=False, **inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_warmup_both_paths_and_seed(self):
        calls = []
        def infer(images, state, action_prefix, delay, seed):
            calls.append((delay, seed))
            self.assertAlmostEqual(float(state[6]), 0.5086, places=5)
            return np.tile(state, (48, 1))
        fake = SimpleNamespace(infer=infer, last_prefix_error=0.0)
        server.PythonBackend.warmup(fake, n=2, delay=10, repeat=3)
        self.assertEqual(calls, [(0, 12345)] * 5 + [(10, 12345)] * 5)

    def test_real_websocket_rtc_and_worker_affinity(self):
        async def run():
            seen = []
            class Backend:
                def info(self):
                    return {"backend": "test", "wire_target_sizes": server.WIRE_TARGET_SIZES}
                def infer(self, images, state, action_prefix=None, delay=0):
                    seen.append(threading.get_ident())
                    self_test.assertEqual(delay, 8)
                    self_test.assertEqual(images["top"].shape, (192, 320, 3))
                    np.testing.assert_array_equal(action_prefix, prefix)
                    return np.tile(state, (48, 1))
            self_test = self
            prefix = np.ones((8, 14), dtype=np.float32)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            ready = asyncio.Event()
            import websockets
            real_serve = websockets.serve
            @contextlib.asynccontextmanager
            async def serve_ready(*args, **kwargs):
                async with real_serve(*args, **kwargs) as ws:
                    ready.set()
                    yield ws
            with ThreadPoolExecutor(max_workers=1) as pool:
                warmup_thread = pool.submit(threading.get_ident).result()
                with patch.object(websockets, "serve", serve_ready):
                    task = asyncio.create_task(server._serve(server.ActionServer(Backend(), "test"),
                                                             "127.0.0.1", port, pool))
                    try:
                        await asyncio.wait_for(ready.wait(), 5)
                        def request():
                            c = client.ServerClient(f"ws://127.0.0.1:{port}")
                            try:
                                obs = {k: np.zeros((480, 640, 3), dtype=np.uint8) for k in server.CAMERA_KEYS}
                                state = np.arange(14, dtype=np.float32) / 10
                                for _ in range(2):
                                    out = c.infer(obs, state, None, prefix, 8)
                                    np.testing.assert_array_equal(out, np.tile(state, (48, 1)))
                                self.assertLess(c.last_timing["payload_mib"], 0.36)
                            finally:
                                c.close()
                        await asyncio.wait_for(asyncio.to_thread(request), 10)
                    finally:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                self.assertEqual(seen, [warmup_thread, warmup_thread])
        asyncio.run(run())

    @unittest.skipUnless(os.environ.get("RTC_TEST_CUDA") == "1", "optional CUDA smoke")
    def test_cuda_graphs_multiple_observations_and_rtc(self):
        model, inputs = tiny_model("cuda")
        # 部署权重/计算均为 BF16；这项测试也覆盖同一精度下的图缓冲区行为。
        model.to(dtype=torch.bfloat16)
        inputs = {k: v.to(dtype=torch.bfloat16) for k, v in inputs.items()}
        def infer(delay):
            model.reset_action_only_prefix_cache()
            torch.compiler.cudagraph_mark_step_begin()
            action = inputs["action"].clone()
            mask = rtc.build_prefix_mask(torch.tensor([delay], device="cuda"), 48)
            prefix = torch.full_like(action, 0.125)
            for step in range(3):
                action = rtc.clamp_prefix(action, prefix, mask)
                ts = inputs["timestep"].clone()
                ts[:, -48:] = 500 - step * 100
                ts[:, 5:5 + delay] = 0
                torch.compiler.cudagraph_mark_step_begin()
                prediction = model(**{**inputs, "action": action, "timestep": ts},
                                   action_only=True, return_dict=False)
                action = action - prediction * 0.01
            action = rtc.clamp_prefix(action, prefix, mask)
            self.assertTrue(torch.equal(action[:, :delay], prefix[:, :delay]))
            return action.cpu().clone()
        with torch.no_grad():
            baseline = {d: infer(d) for d in (0, 8, 10)}
            pipeline.compile_policy_action_blocks(SimpleNamespace(transformer=model), compile_prefix=True)
            for d in (0, 0, 8, 8, 10, 0, 8):
                torch.testing.assert_close(infer(d), baseline[d], atol=0.005, rtol=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
