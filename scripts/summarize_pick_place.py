#!/usr/bin/env python3
"""Aggregate Pick and Place collection, training, and ACT evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/artifacts/pick_place"))
    args = parser.parse_args()
    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)

    collections = [
        load(args.root / "data/rosbags/collection_summary.json"),
        load(args.root / "data/rosbags_supplement/collection_summary.json"),
    ]
    conversion = load(args.root / "data/lerobot/conversion_report.json")
    training = load(args.root / "models/act_pick_place_deterministic/train_metrics.json")
    evaluation = load(args.root / "results/evaluation_final/evaluation_summary.json")
    # Report the exact 20 physically successful episodes used by the learner.
    successes = conversion["manifest"]
    lifts = np.asarray([row["pose_metrics"]["lift_m"] for row in successes])
    distances = np.asarray([row["pose_metrics"]["transport_xy_m"] for row in successes])
    final_x = np.asarray([row["pose_metrics"]["final_xyz_m"][0] for row in successes])
    final_y = np.asarray([row["pose_metrics"]["final_xyz_m"][1] for row in successes])

    summary = {
        "teacher": {
            "requested": sum(row["requested_episodes"] for row in collections),
            "physically_successful_candidates": sum(row["successful_episodes"] for row in collections),
            "attempts": sum(row["attempt_count"] for row in collections),
            "physical_success_rate": (
                sum(row["successful_episodes"] for row in collections)
                / sum(row["attempt_count"] for row in collections)
            ),
            "dataset_selected": len(successes),
            "rejected_incomplete_stage_events": len(conversion["rejected"]),
            "lift_m_mean": float(lifts.mean()),
            "lift_m_std": float(lifts.std()),
            "transport_xy_m_mean": float(distances.mean()),
            "transport_xy_m_std": float(distances.std()),
            "final_x_m_mean": float(final_x.mean()),
            "final_y_m_mean": float(final_y.mean()),
        },
        "dataset": {"episodes": conversion["episodes"], "frames": conversion["frames"], "fps": conversion["fps"]},
        "training": training,
        "evaluation": {
            "trials": evaluation["trials"],
            "successes": evaluation["successes"],
            "success_rate": evaluation["success_rate"],
            "stage_success_counts": evaluation["stage_success_counts"],
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    indices = np.arange(len(successes)) + 1
    axes[0].plot(indices, lifts * 100, "o-", label="maximum lift")
    axes[0].axhline(5, color="black", linestyle="--", label="success threshold")
    axes[0].set(xlabel="Episode", ylabel="Lift (cm)", title="Teacher: physical lift")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].scatter(final_x * 100, final_y * 100, label="placed block")
    axes[1].axvspan(15, 35, alpha=0.15, color="green", label="goal x range")
    axes[1].axhspan(-8, 8, alpha=0.10, color="green")
    axes[1].set(xlabel="Final x (cm)", ylabel="Final y (cm)", title="Teacher: placement positions")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "teacher_physics.png", dpi=180)
    plt.close(fig)

    train_rows = [json.loads(line) for line in (
        args.root / "models/act_pick_place_deterministic/train_log.jsonl"
    ).read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], alpha=0.6)
    ax.set(xlabel="CPU training step", ylabel="ACT loss", title="Pick and Place ACT training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)

    stages = ["approach", "grasp", "lift", "transport", "place"]
    values = [evaluation["stage_success_counts"][stage] for stage in stages]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stages, values, color="#4472c4")
    ax.axhline(evaluation["trials"], color="black", linestyle="--")
    ax.set(ylim=(0, evaluation["trials"] + 0.5), ylabel="Successful trials", title="ACT stage success")
    fig.tight_layout()
    fig.savefig(output / "act_stage_success.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
