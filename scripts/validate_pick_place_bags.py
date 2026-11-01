#!/usr/bin/env python3
"""Validate successful Pick and Place bags and their physical-pose evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


REQUIRED = {
    "/joint_states": "sensor_msgs/msg/JointState",
    "/pick_place/action": "sensor_msgs/msg/JointState",
    "/pick_place/stage": "std_msgs/msg/String",
}


def inspect(result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    bag_path = Path(result["bag_path"])
    counts = {topic: 0 for topic in REQUIRED}
    types = {}
    stages = []
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            types[connection.topic] = connection.msgtype
        connections = [connection for connection in reader.connections if connection.topic in REQUIRED]
        for connection, _timestamp, data in reader.messages(connections=connections):
            counts[connection.topic] += 1
            if connection.topic == "/pick_place/stage":
                stages.append(deserialize_cdr(data, connection.msgtype).data)
    errors = []
    for topic, expected_type in REQUIRED.items():
        if counts[topic] == 0:
            errors.append(f"missing messages on {topic}")
        if types.get(topic) != expected_type:
            errors.append(f"wrong type on {topic}: {types.get(topic)}")
    expected_stages = 8
    if counts["/pick_place/stage"] != expected_stages:
        errors.append(f"expected {expected_stages} stage events, got {counts['/pick_place/stage']}")
    if not result.get("success") or not result.get("pose_metrics", {}).get("success"):
        errors.append("episode lacks verified physical success")
    return {
        "episode": result["episode"],
        "result_path": str(result_path),
        "bag_path": str(bag_path),
        "counts": counts,
        "stages": stages,
        "pose_metrics": result.get("pose_metrics"),
        "valid": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()
    rows = []
    for root in args.root:
        for path in sorted((root / "attempts").glob("*/result.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("success"):
                rows.append(inspect(path))
    report = {
        "episodes": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "invalid": sum(not row["valid"] for row in rows),
        "total_messages": sum(sum(row["counts"].values()) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["episodes", "valid", "invalid", "total_messages"]}, indent=2))
    if report["invalid"] and not args.allow_invalid:
        raise SystemExit("invalid Pick and Place bags found")


if __name__ == "__main__":
    main()
