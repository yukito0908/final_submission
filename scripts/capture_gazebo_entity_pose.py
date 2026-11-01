#!/usr/bin/env python3
"""Stream one entity pose from Ignition Gazebo into a compact CSV file."""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="wood_cube_5cm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hz", type=float, default=20.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    process = subprocess.Popen(
        ["ign", "topic", "-e", "-t", "/world/default/pose/info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    started = time.monotonic()
    last_written = -1e9
    found_entity = False
    in_position = False
    xyz: dict[str, float] = {}
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wall_time_s", "x_m", "y_m", "z_m"])
            assert process.stdout is not None
            while not stop:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                stripped = line.strip()
                if stripped == f'name: "{args.entity}"':
                    found_entity = True
                    in_position = False
                    xyz.clear()
                    continue
                if found_entity and stripped == "position {":
                    in_position = True
                    continue
                if found_entity and in_position and stripped.startswith(("x:", "y:", "z:")):
                    key, value = stripped.split(":", 1)
                    xyz[key] = float(value.strip())
                    if len(xyz) == 3:
                        now = time.monotonic() - started
                        if now - last_written >= 1.0 / args.max_hz:
                            writer.writerow([f"{now:.6f}", xyz["x"], xyz["y"], xyz["z"]])
                            handle.flush()
                            last_written = now
                        found_entity = False
                        in_position = False
                        xyz.clear()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
