#!/usr/bin/env python3
"""Collect verified Pick and Place demonstrations with rosbag2 and Gazebo poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


TOPICS = [
    "/joint_states",
    "/pick_place/action",
    "/pick_place/stage",
]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stop_process(process: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def stop_process_group(process: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def reset_cube(cfg: dict) -> None:
    x, y, z, qx, qy, qz, qw = cfg["cube_initial_pose"]
    request = (
        f'id: {int(cfg["cube_entity_id"])} position {{ x: {x} y: {y} z: {z} }} '
        f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
    )
    completed = subprocess.run(
        [
            "ign", "service", "-s", "/world/default/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "5000", "--req", request,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if "true" not in completed.stdout.lower():
        raise RuntimeError(f"Gazebo rejected cube reset: {completed.stdout}")
    time.sleep(float(cfg["reset_settle_s"]))


def pose_metrics(path: Path, cfg: dict) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 10:
        raise RuntimeError(f"too few Gazebo pose samples: {len(rows)}")
    positions = [(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])) for row in rows]
    first = positions[0]
    tail = positions[-min(10, len(positions)):]
    final = tuple(sum(point[i] for point in tail) / len(tail) for i in range(3))
    maximum_z = max(point[2] for point in positions)
    lift = maximum_z - first[2]
    transport_xy = math.hypot(final[0] - first[0], final[1] - first[1])
    goal_x_min, goal_x_max = map(float, cfg["goal_x_range_m"])
    checks = {
        "lift": lift >= float(cfg["minimum_lift_m"]),
        "transport": transport_xy >= float(cfg["minimum_transport_xy_m"]),
        "goal_x": goal_x_min <= final[0] <= goal_x_max,
        "goal_y": abs(final[1]) <= float(cfg["goal_y_abs_max_m"]),
    }
    return {
        "samples": len(positions),
        "initial_xyz_m": first,
        "final_xyz_m": final,
        "maximum_z_m": maximum_z,
        "lift_m": lift,
        "transport_xy_m": transport_xy,
        "checks": checks,
        "success": all(checks.values()),
    }


def collect_attempt(index: int, attempt: int, cfg: dict, output: Path) -> dict:
    name = f"pick_place_{index:03d}"
    attempt_root = output / "attempts" / f"{name}_attempt_{attempt:02d}"
    bag_path = attempt_root / "bag"
    pose_path = attempt_root / "cube_pose.csv"
    attempt_root.mkdir(parents=True, exist_ok=True)

    reset_cube(cfg)
    pose_process = subprocess.Popen([
        "python3", "/workspace/scripts/capture_gazebo_entity_pose.py",
        "--entity", str(cfg["cube_entity_name"]), "--output", str(pose_path),
        "--max-hz", str(cfg["fps"]),
    ])
    bag_log = (attempt_root / "rosbag.log").open("w", encoding="utf-8")
    teacher_log = (attempt_root / "teacher.log").open("w", encoding="utf-8")
    bag_process = subprocess.Popen(
        ["ros2", "bag", "record", "--storage", "sqlite3", "-o", str(bag_path), *TOPICS],
        stdout=bag_log, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(1.5)
    started = time.time()
    teacher_returncode = -1
    teacher_process = None
    try:
        teacher_process = subprocess.Popen(
            [
                "ros2", "launch", "crane_plus_pick_place_teacher", "teacher.launch.py",
                "use_sim_time:=true",
            ],
            stdout=teacher_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        teacher_returncode = teacher_process.wait(timeout=float(cfg["teacher_timeout_s"]))
        time.sleep(3.0)
    except subprocess.TimeoutExpired:
        teacher_returncode = 124
        if teacher_process is not None:
            stop_process_group(teacher_process)
    finally:
        stop_process(bag_process)
        stop_process(pose_process)
        bag_log.close()
        teacher_log.close()

    metrics = pose_metrics(pose_path, cfg)
    result = {
        "episode": name,
        "attempt": attempt,
        "teacher_returncode": teacher_returncode,
        "duration_wall_s": time.time() - started,
        "bag_path": str(bag_path),
        "pose_path": str(pose_path),
        "pose_metrics": metrics,
        "success": teacher_returncode == 0 and metrics["success"],
    }
    atomic_json(attempt_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/workspace/configs/pick_place.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/artifacts/pick_place/data/rosbags"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-attempts", type=int, default=30)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target_count = int(args.episodes if args.episodes is not None else cfg["episodes"])
    args.output.mkdir(parents=True, exist_ok=True)

    successes = []
    attempts = []
    attempt_number = 0
    while len(successes) < target_count and attempt_number < args.max_attempts:
        attempt_number += 1
        result = collect_attempt(len(successes), attempt_number, cfg, args.output)
        attempts.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["success"]:
            successes.append(result)
    summary = {
        "requested_episodes": target_count,
        "successful_episodes": len(successes),
        "attempt_count": len(attempts),
        "all_successful": len(successes) == target_count,
        "successes": successes,
        "attempts": attempts,
    }
    atomic_json(args.output / "collection_summary.json", summary)
    if not summary["all_successful"]:
        raise SystemExit("did not collect the requested number of successful episodes")


if __name__ == "__main__":
    main()
