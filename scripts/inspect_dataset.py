#!/usr/bin/env python3
"""Load a LeRobotDataset end-to-end and validate feature dimensions and values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


FEATURES = ["observation.state", "observation.environment_state", "action"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    values = {key: [] for key in FEATURES}
    errors = []
    for index in range(len(dataset)):
        sample = dataset[index]
        for key in FEATURES:
            value = np.asarray(sample[key], dtype=np.float32)
            if value.shape != (5,):
                errors.append(f"frame {index} {key} shape {value.shape}")
            if not np.isfinite(value).all():
                errors.append(f"frame {index} {key} contains non-finite values")
            values[key].append(value)
    arrays = {key: np.stack(value) for key, value in values.items()}
    report = {
        "repo_id": args.repo_id,
        "root": str(args.root),
        "fps": dataset.fps,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "features": {
            key: {
                "shape": list(array.shape),
                "min": array.min(axis=0).tolist(),
                "max": array.max(axis=0).tolist(),
                "mean": array.mean(axis=0).tolist(),
                "std": array.std(axis=0).tolist(),
            }
            for key, array in arrays.items()
        },
        "valid": not errors,
        "errors": errors[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["repo_id", "fps", "episodes", "frames", "valid"]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
