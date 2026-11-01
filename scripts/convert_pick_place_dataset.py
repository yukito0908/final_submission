#!/usr/bin/env python3
"""Convert physically verified Pick and Place bags to LeRobotDataset v2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


JOINT_NAMES = [
    "crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint4",
    "crane_plus_joint3", "crane_plus_joint_hand",
]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
}


def successful_results(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for path in sorted((root / "attempts").glob("*/result.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("success"):
                payload["result_path"] = str(path)
                rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"no successful demonstrations under {roots}")
    return rows


def ordered(message) -> list[float]:
    mapping = dict(zip(message.name, message.position))
    missing = [name for name in JOINT_NAMES if name not in mapping]
    if missing:
        raise ValueError(f"joint message missing {missing}")
    return [mapping[name] for name in JOINT_NAMES]


def read_bag(path: Path):
    streams = {"state": [], "action": [], "stage": []}
    topic_keys = {
        "/joint_states": "state",
        "/pick_place/action": "action",
        "/pick_place/stage": "stage",
    }
    with Reader(path) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topic_keys]
        for connection, timestamp, data in reader.messages(connections=connections):
            message = deserialize_cdr(data, connection.msgtype)
            key = topic_keys[connection.topic]
            if key == "stage":
                streams[key].append((timestamp, message.data))
            else:
                streams[key].append((timestamp, ordered(message)))
    if not streams["state"] or not streams["action"] or not streams["stage"]:
        raise ValueError(f"missing required streams in {path}")
    return streams


def sample_hold(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, targets, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return values[indices]


def resample(streams: dict, fps: int):
    state_t = np.asarray([row[0] for row in streams["state"]], dtype=np.int64)
    state_v = np.asarray([row[1] for row in streams["state"]], dtype=np.float32)
    action_t = np.asarray([row[0] for row in streams["action"]], dtype=np.int64)
    action_v = np.asarray([row[1] for row in streams["action"]], dtype=np.float32)
    stage_times = {stage: timestamp for timestamp, stage in streams["stage"]}
    if "approach" not in stage_times or "retreat" not in stage_times:
        raise ValueError("episode is missing approach/retreat crop markers")
    # Learn only the task itself. Reset and post-task retreat both revisit the
    # vertical posture and would make a state-only policy temporally ambiguous.
    start = max(state_t[0], action_t[0], stage_times["approach"])
    end = min(state_t[-1], action_t[-1], stage_times["retreat"])
    targets = np.arange(start, end + 1, round(1e9 / fps), dtype=np.int64)
    states = sample_hold(state_t, state_v, targets)
    actions = sample_hold(action_t, action_v, targets)
    if len(targets) < 2 or states.shape[1:] != (5,) or actions.shape[1:] != (5,):
        raise ValueError(f"bad synchronized shapes: {states.shape}, {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite values in episode")
    return states, actions, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}")
        shutil.rmtree(args.output)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output,
        robot_type="crane_plus",
        features=FEATURES,
        use_videos=False,
    )
    manifest = []
    rejected = []
    for result in successful_results(args.bags):
        streams = read_bag(Path(result["bag_path"]))
        stages_present = {stage for _timestamp, stage in streams["stage"]}
        if not {"approach", "retreat"}.issubset(stages_present) or len(streams["stage"]) != 8:
            rejected.append({"source": result["bag_path"], "reason": "incomplete stage events"})
            continue
        states, actions, targets = resample(streams, args.fps)
        for state, action in zip(states, actions):
            dataset.add_frame({
                "observation.state": torch.from_numpy(state),
                "observation.environment_state": torch.from_numpy(state.copy()),
                "action": torch.from_numpy(action),
                "task": "pick up the fixed block and place it at the goal",
            })
        dataset.save_episode()
        stage_zero = targets[0]
        stages = [
            {"time_s": (timestamp - stage_zero) / 1e9, "stage": stage}
            for timestamp, stage in streams["stage"]
        ]
        manifest.append({
            "episode_index": len(manifest),
            "source": result["bag_path"],
            "frames": len(states),
            "stages": stages,
            "pose_metrics": result["pose_metrics"],
            "state_min": states.min(axis=0).tolist(),
            "state_max": states.max(axis=0).tolist(),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
        })
        print(f"converted {len(manifest)}: {len(states)} frames", flush=True)

    report = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "episodes": len(manifest),
        "frames": sum(row["frames"] for row in manifest),
        "joint_order": JOINT_NAMES,
        "source_bags": [str(path) for path in args.bags],
        "output": str(args.output),
        "rejected": rejected,
        "manifest": manifest,
    }
    if len(manifest) != args.expected_episodes:
        raise RuntimeError(
            f"expected {args.expected_episodes} complete episodes, found {len(manifest)}; "
            f"rejected {len(rejected)}")
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["repo_id", "episodes", "frames"]}, indent=2))


if __name__ == "__main__":
    main()
