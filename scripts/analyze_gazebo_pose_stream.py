#!/usr/bin/env python3
"""Extract one Gazebo entity's position statistics from `ign topic -e` output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path)
    parser.add_argument("--entity", default="wood_cube_5cm")
    args = parser.parse_args()

    text = args.stream.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        rf'name: "{re.escape(args.entity)}".*?position \{{\s*'
        r"x: ([^\s]+)\s*y: ([^\s]+)\s*z: ([^\s]+)",
        re.DOTALL,
    )
    positions = [tuple(map(float, match)) for match in pattern.findall(text)]
    if not positions:
        raise SystemExit(f"No poses found for entity: {args.entity}")

    first = positions[0]
    last = positions[-1]
    result = {
        "entity": args.entity,
        "samples": len(positions),
        "first_xyz_m": first,
        "last_xyz_m": last,
        "min_xyz_m": [min(p[i] for p in positions) for i in range(3)],
        "max_xyz_m": [max(p[i] for p in positions) for i in range(3)],
        "delta_xyz_m": [last[i] - first[i] for i in range(3)],
        "max_lift_from_initial_m": max(p[2] for p in positions) - first[2],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
