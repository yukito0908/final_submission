#!/usr/bin/env python3
"""Run the trained ACT policy in Gazebo and score physical Pick and Place stages."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
import yaml

from collect_pick_place import atomic_json, pose_metrics, reset_cube, stop_process
from evaluate_policy import EvaluationNode, PolicyWorker, clipped_action


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("/workspace/configs/pick_place.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--run-duration", type=float, default=36.0)
    parser.add_argument("--max-step", type=float, default=0.08)
    parser.add_argument("--worker-python", default="/home/ubuntu/pai_training_stage/.pixi/envs/default/bin/python")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frequency = float(cfg["fps"])
    initial = np.zeros(5, dtype=np.float64)

    worker = PolicyWorker(
        args.worker_python, Path(__file__).with_name("policy_worker.py"), args.model)
    rclpy.init()
    node = EvaluationNode(frequency)
    summaries = []
    all_traces = []
    try:
        node.wait_ready()
        for trial in range(args.trials):
            assert node.latest is not None
            node.segment(node.latest.copy(), initial, 4.0)
            node.hold(initial, 1.0)
            reset_cube(cfg)
            worker.reset()

            trial_root = args.output / f"trial_{trial:02d}"
            trial_root.mkdir(parents=True, exist_ok=True)
            pose_path = trial_root / "cube_pose.csv"
            pose_process = subprocess.Popen([
                "python3", "/workspace/scripts/capture_gazebo_entity_pose.py",
                "--entity", str(cfg["cube_entity_name"]),
                "--output", str(pose_path), "--max-hz", str(cfg["fps"]),
            ])
            time.sleep(0.5)
            start = time.monotonic()
            trial_traces = []
            try:
                while time.monotonic() - start < args.run_duration:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    measured = node.latest.copy()
                    raw = worker.action(measured)
                    command = clipped_action(raw, measured, args.max_step)
                    elapsed = time.monotonic() - start
                    row = {"trial": trial, "time_s": elapsed}
                    for index in range(5):
                        row[f"state_{index}"] = float(measured[index])
                        row[f"action_{index}"] = float(command[index])
                        row[f"raw_action_{index}"] = float(raw[index])
                    trial_traces.append(row)
                    all_traces.append(row)
                    node.tick(command)
                node.hold(node.latest.copy(), 1.0)
            finally:
                stop_process(pose_process)

            metrics = pose_metrics(pose_path, cfg)
            states = np.asarray([
                [row[f"state_{index}"] for index in range(5)] for row in trial_traces])
            actions = np.asarray([
                [row[f"action_{index}"] for index in range(5)] for row in trial_traces])
            arm_excursion = float(np.max(np.linalg.norm(states[:, :4] - initial[:4], axis=1)))
            opened = float(np.min(actions[:, 4])) <= -0.30
            closed = float(np.max(actions[:, 4])) >= 0.05
            stages = {
                "approach": arm_excursion >= 1.0 and opened,
                "grasp": closed and bool(metrics["checks"]["lift"]),
                "lift": bool(metrics["checks"]["lift"]),
                "transport": bool(metrics["checks"]["transport"]),
                "place": bool(metrics["checks"]["goal_x"] and metrics["checks"]["goal_y"] and opened),
            }
            summary = {
                "trial": trial,
                "run_duration_s": args.run_duration,
                "arm_excursion_rad": arm_excursion,
                "gripper_min_rad": float(np.min(actions[:, 4])),
                "gripper_max_rad": float(np.max(actions[:, 4])),
                "pose_metrics": metrics,
                "stages": stages,
                "success": all(stages.values()),
            }
            summaries.append(summary)
            atomic_json(trial_root / "result.json", summary)
            write_csv(trial_root / "trace.csv", trial_traces)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        worker.close()

    report = {
        "model": str(args.model),
        "trials": len(summaries),
        "successes": sum(row["success"] for row in summaries),
        "success_rate": sum(row["success"] for row in summaries) / len(summaries),
        "stage_success_counts": {
            stage: sum(row["stages"][stage] for row in summaries)
            for stage in ["approach", "grasp", "lift", "transport", "place"]
        },
        "results": summaries,
    }
    atomic_json(args.output / "evaluation_summary.json", report)
    write_csv(args.output / "all_traces.csv", all_traces)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
