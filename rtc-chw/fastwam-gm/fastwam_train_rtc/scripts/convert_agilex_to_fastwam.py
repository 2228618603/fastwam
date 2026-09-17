#!/usr/bin/env python
"""把 Agilex/GR00T 风格的 LeRobot v2.1 数据集改名成 FastWAM 期望的字段命名。

FastWAM 从 shape_meta 的 key 拼出 LeRobot 字段名
(src/fastwam/datasets/lerobot/base_lerobot_dataset.py:79-96)：

    images -> observation.images.{key}
    state  -> observation.state.{key}   /  "observation.state"  (key == "default")
    action -> action.{key}              /  "action"             (key == "default")

拼出的名字用于三处，所以必须与磁盘上的实际命名严格一致：
  1. delta_timestamps 的 key -> 查询 parquet 列
  2. info.json 的 features 键 -> 判定是否 video 字段
  3. video_path 模板的 {video_key} -> 视频子目录名

因此需要的改名：
    observation.image.top           -> observation.images.cam_high
    observation.image.left_wrist    -> observation.images.cam_left_wrist
    observation.image.right_wrist   -> observation.images.cam_right_wrist
    observation.gripper_position    -> observation.state.gripper_position
    actions                         -> action

observation.state.joint / observation.state.end 天然合规，不动。

视频不复制、不转码，只做目录软链接（28GB -> 0 额外磁盘）；
parquet 逐个重写列名（169MB，几分钟）。源目录全程只读。

用法:
    python scripts/convert_agilex_to_fastwam.py \
      --src ./data/agilex_empty_the_box_all_470 \
      --dst ./data/agilex_empty_the_box_fastwam [--limit 5]
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

# ---- 改名表 ----------------------------------------------------------------

VIDEO_RENAME = {
    "observation.image.top": "observation.images.cam_high",
    "observation.image.left_wrist": "observation.images.cam_left_wrist",
    "observation.image.right_wrist": "observation.images.cam_right_wrist",
}

COLUMN_RENAME = {
    "observation.gripper_position": "observation.state.gripper_position",
    "actions": "action",
}

ALL_RENAME = {**VIDEO_RENAME, **COLUMN_RENAME}


def convert_info(src_meta: Path, dst_meta: Path) -> dict:
    """改写 info.json 的 features 键（保持插入顺序）。"""
    info = json.loads((src_meta / "info.json").read_text())

    new_features = {}
    for key, value in info["features"].items():
        new_features[ALL_RENAME.get(key, key)] = value
    info["features"] = new_features

    (dst_meta / "info.json").write_text(json.dumps(info, indent=4))
    return info


def convert_episodes_stats(src_meta: Path, dst_meta: Path) -> int:
    """改写 episodes_stats.jsonl 的 stats 键。

    技术上可选（aggregate_stats 只对 key 求并集、不校验 features；FastWAM 的
    归一化用它自己生成的 dataset_stats.json），但保持一致更省心。
    """
    src = src_meta / "episodes_stats.jsonl"
    if not src.exists():
        print("[skip] episodes_stats.jsonl 不存在")
        return 0

    n = 0
    with src.open() as fin, (dst_meta / "episodes_stats.jsonl").open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["stats"] = {
                ALL_RENAME.get(k, k): v for k, v in record["stats"].items()
            }
            fout.write(json.dumps(record) + "\n")
            n += 1
    return n


def convert_parquet(src_root: Path, dst_root: Path, limit: int | None) -> int:
    """逐个重写 parquet 的列名。"""
    files = sorted((src_root / "data").rglob("*.parquet"))
    if limit is not None:
        files = files[:limit]

    for src_file in tqdm(files, desc="rewriting parquet"):
        table = pq.read_table(src_file)
        new_names = [COLUMN_RENAME.get(n, n) for n in table.column_names]
        table = table.rename_columns(new_names)

        dst_file = dst_root / src_file.relative_to(src_root)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_file)

    return len(files)


def link_videos(src_root: Path, dst_root: Path) -> int:
    """把每个相机目录软链接过去（不复制、不转码）。"""
    n = 0
    for chunk_dir in sorted((src_root / "videos").glob("chunk-*")):
        dst_chunk = dst_root / "videos" / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)

        for cam_dir in sorted(chunk_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            new_name = VIDEO_RENAME.get(cam_dir.name)
            if new_name is None:
                print(f"[warn] 未知相机目录，跳过：{cam_dir.name}")
                continue
            target = dst_chunk / new_name
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(cam_dir.resolve(), target_is_directory=True)
            n_mp4 = len(list(cam_dir.glob("*.mp4")))
            print(f"[link] {new_name} -> {cam_dir}  ({n_mp4} mp4)")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只转前 N 个 episode 的 parquet，用于快速验证",
    )
    args = ap.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    if src_root == dst_root:
        raise SystemExit("--src 与 --dst 不能相同（源目录必须保持只读）")

    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # 1) meta
    info = convert_info(src_meta, dst_meta)
    print("[ok] info.json features:")
    for k in info["features"]:
        print(f"       {k}")

    for name in ["tasks.jsonl", "episodes.jsonl"]:
        shutil.copy2(src_meta / name, dst_meta / name)
    print("[ok] 复制 tasks.jsonl / episodes.jsonl")

    n_stats = convert_episodes_stats(src_meta, dst_meta)
    print(f"[ok] episodes_stats.jsonl（{n_stats} 行）")

    # 2) parquet
    n = convert_parquet(src_root, dst_root, args.limit)
    print(f"[ok] 重写 {n} 个 parquet")

    # 3) videos（软链接）
    n_link = link_videos(src_root, dst_root)
    print(f"[ok] 建立 {n_link} 个相机目录软链接")

    print(f"\n完成。数据集根目录：{dst_root}")
    print("下一步：确认 configs/data/agilex_3cam.yaml 的 dataset_dirs 指向它。")


if __name__ == "__main__":
    main()
