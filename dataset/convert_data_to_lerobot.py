"""Convert plate_task_0505 raw demos to a LeRobot v2 dataset for DreamZero/YAM training.

Source layout (per demo):
  demo{N}/
    instruction.npy             (T,)    object strings, per-frame task description
    left_joint_positions.npy    (T, 7)  float64  observed left arm joints
    left_gripper_position.npy   (T, 1)  float64  observed left gripper
    left_control.npy            (T, 7)  float64  commanded left arm joints (action)
    left_joint_velocities.npy   (T, 7)  unused
    stage_id.npy                (T,)    unused
    metadata.json               episode info (hz=30, task, segments)
    top_camera_rgb.mp4          h264, 640x480, 30 fps, T frames
    left_camera_rgb.mp4
    right_camera_rgb.mp4

Target layout (LeRobot v2, matches what convert_lerobot_to_gear.py reads):
  plate_task_0505_lerobot/
    data/chunk-000/episode_000000.parquet ...
    videos/chunk-000/observation.images.top_camera-images-rgb/episode_000000.mp4 ...
    videos/chunk-000/observation.images.left_camera-images-rgb/episode_000000.mp4 ...
    videos/chunk-000/observation.images.right_camera-images-rgb/episode_000000.mp4 ...
    meta/info.json, episodes.jsonl, tasks.jsonl

Bimanual padding: the YAM modality config expects 14-dim state/action
(left_joint_pos[0:7], left_gripper_pos[7:8], right_joint_pos[8:15], right_gripper_pos[15:16]).
Our YAM station is single-arm (left only), so right_* slots are zero-filled.

Camera key names (top/left/right) match modality_config_yam in
groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml.

Usage:
  python convert_data_to_lerobot.py \
      --src /iris/u/kewalk/dataset/plate_task_0505 \
      --dst /iris/u/kewalk/dataset/plate_task_0505_lerobot

After this, run dreamzero's GEAR converter:
  python scripts/data/convert_lerobot_to_gear.py \
      --dataset-path /iris/u/kewalk/dataset/plate_task_0505_lerobot \
      --embodiment-tag yam \
      --state-keys '{"left_joint_pos": [0, 7], "left_gripper_pos": [7, 8], "right_joint_pos": [8, 15], "right_gripper_pos": [15, 16]}' \
      --action-keys '{"left_joint_pos": [0, 7], "left_gripper_pos": [7, 8], "right_joint_pos": [8, 15], "right_gripper_pos": [15, 16]}' \
      --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
      --task-key annotation.task
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEFT_JOINT_DIM = 7
GRIPPER_DIM = 1
STATE_DIM = 2 * (LEFT_JOINT_DIM + GRIPPER_DIM)
ACTION_DIM = STATE_DIM

CAM_MAP = {
    "top_camera_rgb.mp4": "top_camera-images-rgb",
    "left_camera_rgb.mp4": "left_camera-images-rgb",
    "right_camera_rgb.mp4": "right_camera-images-rgb",
}

CHUNKS_SIZE = 1000


def ffprobe_video(path: Path) -> dict:
    """Return width/height/nb_frames for a video via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "nb_frames": int(info["nb_frames"]),
        "r_frame_rate": info["r_frame_rate"],
    }


def demo_sort_key(p: Path) -> int:
    m = re.match(r"demo(\d+)$", p.name)
    if not m:
        raise ValueError(f"Unexpected folder name: {p}")
    return int(m.group(1))


def load_demo_arrays(demo_dir: Path) -> dict:
    """Load and validate one demo's per-frame arrays. Returns dict of T-length numpy arrays."""
    left_joint = np.load(demo_dir / "left_joint_positions.npy")
    left_gripper = np.load(demo_dir / "left_gripper_position.npy")
    left_control = np.load(demo_dir / "left_control.npy")
    instruction = np.load(demo_dir / "instruction.npy", allow_pickle=True)

    if left_gripper.ndim == 1:
        left_gripper = left_gripper.reshape(-1, 1)

    T = left_joint.shape[0]
    assert left_joint.shape == (T, LEFT_JOINT_DIM), f"{demo_dir}: left_joint {left_joint.shape}"
    assert left_gripper.shape == (T, GRIPPER_DIM), f"{demo_dir}: left_gripper {left_gripper.shape}"
    assert left_control.shape == (T, LEFT_JOINT_DIM), f"{demo_dir}: left_control {left_control.shape}"
    assert instruction.shape[0] == T, f"{demo_dir}: instruction len {instruction.shape[0]} vs T={T}"

    # State: [left_joint(7), left_gripper(1), right_joint=0(7), right_gripper=0(1)] -> 16
    zeros_joint = np.zeros((T, LEFT_JOINT_DIM), dtype=np.float32)
    zeros_gripper = np.zeros((T, GRIPPER_DIM), dtype=np.float32)
    state = np.concatenate(
        [left_joint.astype(np.float32), left_gripper.astype(np.float32),
         zeros_joint, zeros_gripper],
        axis=1,
    )

    # Action: [left_control(7), left_gripper(1) as cmd, right=0(8)]
    # left_gripper command isn't logged separately; use observed gripper as action target.
    action = np.concatenate(
        [left_control.astype(np.float32), left_gripper.astype(np.float32),
         zeros_joint, zeros_gripper],
        axis=1,
    )

    assert state.shape == (T, STATE_DIM)
    assert action.shape == (T, ACTION_DIM)

    instructions = [str(x) for x in instruction]
    return {"state": state, "action": action, "instructions": instructions, "T": T}


def write_episode_parquet(
    ep_idx: int,
    arrays: dict,
    task_to_index: dict[str, int],
    episode_index_global_offset: int,
    out_path: Path,
    fps: int,
) -> None:
    T = arrays["T"]
    timestamps = np.arange(T, dtype=np.float64) / fps
    frame_index = np.arange(T, dtype=np.int64)
    episode_index = np.full(T, ep_idx, dtype=np.int64)
    index = np.arange(episode_index_global_offset, episode_index_global_offset + T, dtype=np.int64)
    task_index = np.array(
        [task_to_index[t] for t in arrays["instructions"]], dtype=np.int64
    )

    df = pd.DataFrame({
        "observation.state": list(arrays["state"]),
        "action": list(arrays["action"]),
        "annotation.task": task_index,
        "timestamp": timestamps,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": index,
        "task_index": task_index,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def copy_videos(demo_dir: Path, ep_idx: int, dst_root: Path, chunk_idx: int) -> dict:
    """Copy the 3 camera mp4s into LeRobot's video layout. Returns dict of probed video info."""
    chunk = f"chunk-{chunk_idx:03d}"
    probes = {}
    for src_name, cam_key in CAM_MAP.items():
        src = demo_dir / src_name
        dst_dir = dst_root / "videos" / chunk / f"observation.images.{cam_key}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"episode_{ep_idx:06d}.mp4"
        shutil.copyfile(src, dst)
        probes[cam_key] = ffprobe_video(dst)
    return probes


def build_info_json(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    video_probe: dict,
) -> dict:
    num_chunks = (total_episodes // CHUNKS_SIZE) + (1 if total_episodes % CHUNKS_SIZE else 0)
    features = {}
    for cam_key, probe in video_probe.items():
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": [probe["height"], probe["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    features.update({
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [
                *[f"left_joint_{i}" for i in range(LEFT_JOINT_DIM)],
                "left_gripper",
                *[f"right_joint_{i}" for i in range(LEFT_JOINT_DIM)],
                "right_gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [
                *[f"left_joint_{i}" for i in range(LEFT_JOINT_DIM)],
                "left_gripper",
                *[f"right_joint_{i}" for i in range(LEFT_JOINT_DIM)],
                "right_gripper",
            ],
        },
        "annotation.task": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    })
    return {
        "codebase_version": "v2.0",
        "robot_type": "yam",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAM_MAP),
        "total_chunks": num_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src", type=str, default="/iris/u/kewalk/dataset/plate_task_0505",
        help="Combined raw demo dir (output of combine_parts.py)",
    )
    parser.add_argument(
        "--dst", type=str, default="/iris/u/kewalk/dataset/plate_task_0505_lerobot",
        help="LeRobot v2 dataset output dir",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS (YAM=30)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing --dst")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        print(f"ERROR: source {src} does not exist", file=sys.stderr)
        sys.exit(1)

    if dst.exists():
        if not args.force:
            print(f"ERROR: {dst} exists. Pass --force to overwrite.", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    demos = sorted(
        (p for p in src.iterdir() if p.is_dir() and p.name.startswith("demo")),
        key=demo_sort_key,
    )
    if not demos:
        print(f"ERROR: no demo* folders under {src}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(demos)} demos in {src}")

    # First pass: load arrays, collect unique tasks, compute totals.
    all_arrays = []
    task_set: dict[str, int] = {}
    for demo in demos:
        arrs = load_demo_arrays(demo)
        for inst in arrs["instructions"]:
            if inst not in task_set:
                task_set[inst] = len(task_set)
        all_arrays.append(arrs)

    tasks_sorted = sorted(task_set.items(), key=lambda kv: kv[1])
    tasks_jsonl = [{"task_index": idx, "task": text} for text, idx in tasks_sorted]
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks_jsonl:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote tasks.jsonl ({len(tasks_jsonl)} unique instructions)")

    # Second pass: write parquet + copy videos.
    total_frames = 0
    episodes_jsonl = []
    video_probe = {}
    for ep_idx, (demo, arrs) in enumerate(zip(demos, all_arrays)):
        chunk_idx = ep_idx // CHUNKS_SIZE
        parquet_path = dst / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        write_episode_parquet(
            ep_idx=ep_idx,
            arrays=arrs,
            task_to_index=task_set,
            episode_index_global_offset=total_frames,
            out_path=parquet_path,
            fps=args.fps,
        )
        probes = copy_videos(demo, ep_idx, dst, chunk_idx)
        # Sanity: video frame count vs array frame count.
        for cam_key, probe in probes.items():
            if probe["nb_frames"] != arrs["T"]:
                print(
                    f"  WARNING: {demo.name} {cam_key} has {probe['nb_frames']} frames "
                    f"but state/action has {arrs['T']}"
                )
        if not video_probe:
            video_probe = probes

        ep_tasks = sorted(set(arrs["instructions"]))
        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": ep_tasks,
            "length": arrs["T"],
        })
        total_frames += arrs["T"]
        print(f"  [{ep_idx + 1:3d}/{len(demos)}] {demo.name}  T={arrs['T']}  -> episode_{ep_idx:06d}")

    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep) + "\n")

    info = build_info_json(
        total_episodes=len(demos),
        total_frames=total_frames,
        total_tasks=len(tasks_jsonl),
        fps=args.fps,
        video_probe=video_probe,
    )
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print()
    print("=" * 60)
    print(f"Done. {len(demos)} episodes, {total_frames} frames, {len(tasks_jsonl)} tasks.")
    print(f"Output: {dst}")
    print("=" * 60)
    print()
    print("Next: run dreamzero's GEAR converter to add meta/modality.json + stats:")
    print(f"  cd /iris/u/kewalk/dreamzero")
    print(f"  python scripts/data/convert_lerobot_to_gear.py \\")
    print(f"      --dataset-path {dst} \\")
    print(f"      --embodiment-tag yam \\")
    print(f"      --state-keys '{{\"left_joint_pos\": [0, 7], \"left_gripper_pos\": [7, 8], \"right_joint_pos\": [8, 15], \"right_gripper_pos\": [15, 16]}}' \\")
    print(f"      --action-keys '{{\"left_joint_pos\": [0, 7], \"left_gripper_pos\": [7, 8], \"right_joint_pos\": [8, 15], \"right_gripper_pos\": [15, 16]}}' \\")
    print(f"      --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \\")
    print(f"      --task-key annotation.task")


if __name__ == "__main__":
    main()
