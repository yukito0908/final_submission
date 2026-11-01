#!/usr/bin/env python3
"""Generate smooth CRANE+ V2 demonstrations and record one rosbag per episode."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml


ARM_JOINT_NAMES = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint4",
    "crane_plus_joint3",
]
GRIPPER_JOINT_NAME = "crane_plus_joint_hand"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]

# Limits from crane_plus.urdf.xacro, reordered to match the controller command order.
ARM_LOWER = np.array([-2.61799, -1.82471, -1.84027, -2.55656], dtype=np.float64)
ARM_UPPER = np.array([2.61799, 1.78466, 1.76844, 2.45428], dtype=np.float64)
GRIPPER_LIMITS = (-0.70930, 0.66794)

ARM_TOPIC = "/crane_plus_arm_controller/joint_trajectory"
GRIPPER_TOPIC = "/crane_plus_gripper_controller/joint_trajectory"
JOINT_STATE_TOPIC = "/joint_states"


def smootherstep(x: float) -> float:
    """Quintic interpolation with zero velocity and acceleration at both ends."""
    x = min(1.0, max(0.0, float(x)))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def interpolate(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    return q0 + smootherstep(alpha) * (q1 - q0)


def validate_arm(q: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(q), dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"arm command must have shape (4,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("arm command contains a non-finite value")
    if np.any(value < ARM_LOWER) or np.any(value > ARM_UPPER):
        raise ValueError(f"arm command outside joint limits: {value.tolist()}")
    return value


def ordered_positions(msg: JointState) -> np.ndarray:
    positions = dict(zip(msg.name, msg.position))
    missing = [name for name in ALL_JOINT_NAMES if name not in positions]
    if missing:
        raise KeyError(f"joint state is missing {missing}")
    return np.asarray([positions[name] for name in ALL_JOINT_NAMES], dtype=np.float64)


class TeacherNode(Node):
    def __init__(self, frequency_hz: float, gripper: float):
        super().__init__("physical_ai_teacher")
        self.frequency_hz = float(frequency_hz)
        self.period_s = 1.0 / self.frequency_hz
        self.gripper = float(gripper)
        if not GRIPPER_LIMITS[0] <= self.gripper <= GRIPPER_LIMITS[1]:
            raise ValueError("gripper command outside joint limits")
        self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, 20)
        self.latest_state: np.ndarray | None = None

    def _on_joint_state(self, msg: JointState) -> None:
        try:
            self.latest_state = ordered_positions(msg)
        except KeyError:
            return

    def spin_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(self.period_s, max(0.0, deadline - time.monotonic())))

    def wait_ready(self, timeout_s: float = 20.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.latest_state is not None
                and self.arm_pub.get_subscription_count() > 0
                and self.gripper_pub.get_subscription_count() > 0
            ):
                return
        raise TimeoutError("joint states/controllers were not ready before timeout")

    def publish(self, arm: Iterable[float], time_from_start_s: float | None = None) -> None:
        arm_value = validate_arm(arm)
        horizon = self.period_s * 1.5 if time_from_start_s is None else time_from_start_s

        arm_msg = JointTrajectory()
        arm_msg.joint_names = ARM_JOINT_NAMES
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_value.tolist()
        arm_point.time_from_start.sec = int(horizon)
        arm_point.time_from_start.nanosec = int((horizon - int(horizon)) * 1e9)
        arm_msg.points.append(arm_point)
        self.arm_pub.publish(arm_msg)

        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = [GRIPPER_JOINT_NAME]
        gripper_point = JointTrajectoryPoint()
        gripper_point.positions = [self.gripper]
        gripper_point.time_from_start.sec = int(horizon)
        gripper_point.time_from_start.nanosec = int((horizon - int(horizon)) * 1e9)
        gripper_msg.points.append(gripper_point)
        self.gripper_pub.publish(gripper_msg)

    def execute_segment(self, q0: np.ndarray, q1: np.ndarray, duration_s: float) -> None:
        steps = max(2, int(round(duration_s * self.frequency_hz)))
        next_tick = time.monotonic()
        for index in range(steps + 1):
            self.publish(interpolate(q0, q1, index / steps))
            rclpy.spin_once(self, timeout_sec=0.0)
            next_tick += self.period_s
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def hold(self, q: np.ndarray, duration_s: float) -> None:
        steps = max(1, int(math.ceil(duration_s * self.frequency_hz)))
        next_tick = time.monotonic()
        for _ in range(steps):
            self.publish(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            next_tick += self.period_s
            time.sleep(max(0.0, next_tick - time.monotonic()))


class BagRecorder:
    def __init__(self, output_dir: Path, log_path: Path):
        self.output_dir = output_dir
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def __enter__(self) -> "BagRecorder":
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "ros2", "bag", "record", "--storage", "sqlite3", "-o", str(self.output_dir),
                JOINT_STATE_TOPIC, ARM_TOPIC, GRIPPER_TOPIC,
            ],
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1.0)
        if self.process.poll() is not None:
            raise RuntimeError(f"ros2 bag record exited early; see {self.log_path}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_episode(
    node: TeacherNode,
    cfg: dict,
    output_root: Path,
    episode_index: int,
    kind: str,
    perturbation_rad: float,
) -> dict:
    initial = validate_arm(cfg["initial_arm"])
    target = validate_arm(cfg["target_arm"])
    joint_index = int(cfg["perturb_joint_index"])
    reset_duration = float(cfg["reset_duration_s"])
    motion_duration = float(cfg["motion_duration_s"])
    perturb_time = float(cfg["perturb_time_s"])
    perturb_hold = float(cfg["perturb_hold_s"])
    settle_duration = float(cfg["settle_duration_s"])
    threshold = float(cfg["success_threshold_rad"])

    episode_name = f"{kind}_{episode_index:03d}"
    bag_path = output_root / kind / episode_name
    metadata_path = output_root / kind / f"{episode_name}.json"
    log_path = output_root / "logs" / f"{episode_name}_rosbag.log"

    node.execute_segment(node.latest_state[:4].copy(), initial, reset_duration)
    node.hold(initial, settle_duration)
    initial_measured = node.latest_state.copy()

    started_at = time.time()
    with BagRecorder(bag_path, log_path):
        if kind == "normal":
            node.execute_segment(initial, target, motion_duration)
        else:
            pre_fraction = perturb_time / motion_duration
            pre_perturb = interpolate(initial, target, pre_fraction)
            node.execute_segment(initial, pre_perturb, perturb_time)
            perturbed = pre_perturb.copy()
            perturbed[joint_index] += perturbation_rad
            validate_arm(perturbed)
            node.hold(perturbed, perturb_hold)
            # The recovery trajectory starts from the measured perturbed posture.
            recovery_start = node.latest_state[:4].copy()
            node.execute_segment(recovery_start, target, motion_duration - perturb_time)
        node.hold(target, settle_duration)

    final_state = node.latest_state.copy()
    final_error = float(np.linalg.norm(final_state[:4] - target))
    result = {
        "episode": episode_name,
        "kind": kind,
        "seed": int(cfg["seed"]),
        "joint_order": ALL_JOINT_NAMES,
        "initial_command": initial.tolist(),
        "initial_measured": initial_measured.tolist(),
        "target": target.tolist(),
        "gripper": float(cfg["gripper"]),
        "perturb_joint_index": joint_index if kind == "recovery" else None,
        "perturbation_rad": float(perturbation_rad),
        "perturb_time_s": perturb_time if kind == "recovery" else None,
        "final_measured": final_state.tolist(),
        "final_error_l2_rad": final_error,
        "success_threshold_rad": threshold,
        "success": bool(final_error <= threshold),
        "duration_wall_s": float(time.time() - started_at),
        "bag_path": str(bag_path),
        "started_unix_s": started_at,
    }
    write_json(metadata_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-episodes", type=int, default=None)
    parser.add_argument("--recovery-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    normal_count = int(cfg["normal_episodes"] if args.normal_episodes is None else args.normal_episodes)
    recovery_count = int(cfg["recovery_episodes"] if args.recovery_episodes is None else args.recovery_episodes)
    perturbations = [float(value) for value in cfg["training_perturbations_rad"]]

    rclpy.init()
    node = TeacherNode(float(cfg["frequency_hz"]), float(cfg["gripper"]))
    results: list[dict] = []
    try:
        node.wait_ready()
        print("ROS 2 joint state and trajectory controllers are ready", flush=True)
        for index in range(normal_count):
            results.append(run_episode(node, cfg, args.output, index, "normal", 0.0))
        for index in range(recovery_count):
            perturbation = perturbations[index % len(perturbations)]
            results.append(run_episode(node, cfg, args.output, index, "recovery", perturbation))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    summary = {
        "episodes": len(results),
        "normal_episodes": normal_count,
        "recovery_episodes": recovery_count,
        "successes": sum(int(row["success"]) for row in results),
        "mean_final_error_l2_rad": float(np.mean([row["final_error_l2_rad"] for row in results])),
        "results": results,
    }
    write_json(args.output / "collection_summary.json", summary)


if __name__ == "__main__":
    main()
