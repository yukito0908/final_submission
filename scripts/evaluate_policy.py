#!/usr/bin/env python3
"""Deploy a trained ACT policy through ROS 2 and evaluate it in Gazebo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml

from teacher_collect import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    ARM_LOWER,
    ARM_UPPER,
    GRIPPER_JOINT_NAME,
    GRIPPER_LIMITS,
    ARM_TOPIC,
    GRIPPER_TOPIC,
    JOINT_STATE_TOPIC,
    interpolate,
    ordered_positions,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PolicyWorker:
    def __init__(self, python: str, script: Path, model: Path):
        self.process = subprocess.Popen(
            [python, str(script), "--model", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        reply = self._read()
        if reply.get("status") != "ready":
            raise RuntimeError(f"policy worker did not become ready: {reply}")

    def _read(self) -> dict:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"policy worker exited unexpectedly: {stderr}")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                # Some dependency versions print informational startup messages to stdout.
                continue
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply

    def request(self, payload: dict) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read()

    def reset(self) -> None:
        self.request({"op": "reset"})

    def action(self, state: np.ndarray) -> np.ndarray:
        reply = self.request({"state": state.tolist()})
        value = np.asarray(reply["action"], dtype=np.float64)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError(f"invalid policy action: {value}")
        return value

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class EvaluationNode(Node):
    def __init__(self, frequency_hz: float):
        super().__init__("physical_ai_policy_evaluation")
        self.period = 1.0 / frequency_hz
        self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_state, 20)
        self.latest: np.ndarray | None = None

    def _on_state(self, message: JointState) -> None:
        try:
            self.latest = ordered_positions(message)
        except KeyError:
            pass

    def wait_ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest is not None and self.arm_pub.get_subscription_count() and self.gripper_pub.get_subscription_count():
                return
        raise TimeoutError("Gazebo joint state/controllers were not ready")

    def publish(self, command: np.ndarray) -> None:
        horizon = self.period * 1.5
        arm = JointTrajectory()
        arm.joint_names = ARM_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = command[:4].tolist()
        point.time_from_start.nanosec = int(horizon * 1e9)
        arm.points.append(point)
        self.arm_pub.publish(arm)

        gripper = JointTrajectory()
        gripper.joint_names = [GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [float(command[4])]
        point.time_from_start.nanosec = int(horizon * 1e9)
        gripper.points.append(point)
        self.gripper_pub.publish(gripper)

    def tick(self, command: np.ndarray) -> None:
        self.publish(command)
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(self.period)

    def segment(self, start: np.ndarray, end: np.ndarray, duration: float) -> None:
        steps = max(2, round(duration / self.period))
        for index in range(steps + 1):
            self.tick(interpolate(start, end, index / steps))

    def hold(self, command: np.ndarray, duration: float) -> None:
        for _ in range(max(1, math.ceil(duration / self.period))):
            self.tick(command)


def clipped_action(raw: np.ndarray, measured: np.ndarray, max_step: float) -> np.ndarray:
    limits_low = np.r_[ARM_LOWER, GRIPPER_LIMITS[0]]
    limits_high = np.r_[ARM_UPPER, GRIPPER_LIMITS[1]]
    safe = np.clip(raw, limits_low, limits_high)
    return np.clip(safe, measured - max_step, measured + max_step)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-python", default="/home/ubuntu/pai_training_stage/.pixi/envs/default/bin/python")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--run-duration", type=float, default=7.0)
    parser.add_argument("--max-step", type=float, default=0.08)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frequency = float(cfg["frequency_hz"])
    initial = np.r_[np.asarray(cfg["initial_arm"], dtype=float), float(cfg["gripper"])]
    target = np.r_[np.asarray(cfg["target_arm"], dtype=float), float(cfg["gripper"])]
    threshold = float(cfg["success_threshold_rad"])
    perturb_index = int(cfg["perturb_joint_index"])
    perturb_time = float(cfg["perturb_time_s"])
    perturb_hold = float(cfg["perturb_hold_s"])
    conditions = [("none", 0.0), ("seen_0p10", 0.10), ("larger_0p15", 0.15)]

    worker = PolicyWorker(args.worker_python, Path(__file__).with_name("policy_worker.py"), args.model)
    rclpy.init()
    node = EvaluationNode(frequency)
    summaries: list[dict] = []
    traces: list[dict] = []
    try:
        node.wait_ready()
        for condition, perturbation in conditions:
            for trial in range(args.trials):
                assert node.latest is not None
                node.segment(node.latest.copy(), initial, float(cfg["reset_duration_s"]))
                node.hold(initial, 0.5)
                worker.reset()
                start = time.monotonic()
                perturbed = False
                recovery_time = None
                raw_peak_step = 0.0
                while time.monotonic() - start < args.run_duration:
                    elapsed = time.monotonic() - start
                    rclpy.spin_once(node, timeout_sec=0.0)
                    measured = node.latest.copy()
                    if perturbation and not perturbed and elapsed >= perturb_time:
                        injected = measured.copy()
                        injected[perturb_index] = np.clip(
                            injected[perturb_index] + perturbation,
                            ARM_LOWER[perturb_index], ARM_UPPER[perturb_index],
                        )
                        node.hold(injected, perturb_hold)
                        perturbed = True
                        continue
                    raw = worker.action(measured)
                    command = clipped_action(raw, measured, args.max_step)
                    raw_peak_step = max(raw_peak_step, float(np.max(np.abs(command - measured))))
                    error = float(np.linalg.norm(measured[:4] - target[:4]))
                    if perturbed and recovery_time is None and error <= threshold:
                        recovery_time = elapsed - perturb_time
                    row = {
                        "model": args.model_name, "condition": condition, "trial": trial,
                        "time_s": elapsed, "error_l2_rad": error,
                    }
                    for index, name in enumerate(ALL_JOINT_NAMES):
                        row[f"state_{name}"] = float(measured[index])
                        row[f"action_{name}"] = float(command[index])
                        row[f"raw_action_{name}"] = float(raw[index])
                    traces.append(row)
                    node.tick(command)
                node.hold(node.latest.copy(), 0.25)
                final = node.latest.copy()
                final_error = float(np.linalg.norm(final[:4] - target[:4]))
                summary = {
                    "model": args.model_name,
                    "condition": condition,
                    "trial": trial,
                    "initial_state": initial.tolist(),
                    "perturbation_rad": perturbation,
                    "final_state": final.tolist(),
                    "final_error_l2_rad": final_error,
                    "success_threshold_rad": threshold,
                    "success": final_error <= threshold,
                    "recovery_time_s": recovery_time,
                    "max_command_step_rad": raw_peak_step,
                    "run_duration_s": args.run_duration,
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        worker.close()

    write_csv(args.output.with_suffix(".csv"), summaries)
    write_csv(args.output.with_name(args.output.name + "_traces").with_suffix(".csv"), traces)
    atomic_json(args.output.with_suffix(".json"), summaries)


if __name__ == "__main__":
    main()
