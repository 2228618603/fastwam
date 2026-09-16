from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_FILES = [
    "app/local_robot_runner.py",
    "app/fastwam_local_policy.py",
    "assets/train_stats.json",
    "assets/fixed_task_context.pt",
    "configs/train.yaml",
    "configs/task/agilex_empty_box_uncond_3cam384.yaml",
    "configs/model/fastwam.yaml",
    "weights/step_015000.pt",
    "weights/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
    "model_cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
]

REQUIRED_DIRS = [
    "model_cache/Wan-AI/Wan2.2-TI2V-5B",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--write-manifest", default="MANIFEST.generated.json")
    p.add_argument("--allow-missing-large", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    missing = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.exists():
            missing.append(rel)
    for rel in REQUIRED_DIRS:
        path = root / rel
        if not path.is_dir() or not any(path.iterdir()):
            missing.append(rel)

    large = [m for m in missing if m.startswith("weights/") or m.startswith("model_cache/")]
    if missing and not (args.allow_missing_large and len(large) == len(missing)):
        print("Missing required bundle paths:")
        for rel in missing:
            print(f"  - {rel}")
        return 2

    manifest = {"root": str(root), "files": {}}
    for rel in REQUIRED_FILES:
        path = root / rel
        if path.exists():
            manifest["files"][rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (root / args.write_manifest).write_text(json.dumps(manifest, indent=2))
    if missing:
        print("Bundle skeleton verified; large model assets are not copied yet.")
    else:
        print(f"Bundle verified. Manifest written to {root / args.write_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

