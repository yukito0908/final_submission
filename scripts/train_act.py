#!/usr/bin/env python3
"""CPU-only, deterministic ACT training for the CRANE+ V2 datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


JOINT_NAMES = [
    "crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint4",
    "crane_plus_joint3", "crane_plus_joint_hand",
]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--use-vae", action=argparse.BooleanOptionalAction, default=True,
        help="Enable ACT's variational encoder (use --no-use-vae for deterministic demonstrations).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train_log.jsonl"
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    policy_features = dataset_to_policy_features(FEATURES)
    output_features = {key: value for key, value in policy_features.items() if value.type is FeatureType.ACTION}
    input_features = {key: value for key, value in policy_features.items() if key not in output_features}

    chunk_size = int(args.chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        # Execute the complete learned action chunk before re-planning.  With one
        # action step, the first target is almost the current state and the
        # closed-loop controller can settle at the initial posture.
        n_action_steps=chunk_size,
        pretrained_backbone_weights=None,
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        use_vae=args.use_vae,
        latent_dim=16,
        n_vae_encoder_layers=2,
    )
    policy = ACTPolicy(config, dataset_stats=metadata.stats).to(device)
    policy.train()

    delta_timestamps = {
        "observation.state": [0.0],
        "observation.environment_state": [0.0],
        "action": [index / metadata.fps for index in range(chunk_size)],
    }
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_timestamps)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    run_started = time.perf_counter()
    losses: list[float] = []
    step_times: list[float] = []
    step = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        while step < args.steps:
            for batch in loader:
                step_started = time.perf_counter()
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                batch["observation.state"] = batch["observation.state"][:, -1, :]
                batch["observation.environment_state"] = batch["observation.environment_state"][:, -1, :]
                loss, loss_dict = policy.forward(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                elapsed = time.perf_counter() - step_started
                value = float(loss.item())
                losses.append(value)
                step_times.append(elapsed)
                record = {
                    "step": step,
                    "loss": value,
                    "step_time_s": elapsed,
                    **{key: float(item) for key, item in loss_dict.items()},
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if step == 0 or (step + 1) % args.log_every == 0:
                    print(json.dumps(record), flush=True)
                step += 1
                if step >= args.steps:
                    break

    policy.save_pretrained(args.output)
    metrics = {
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "threads": args.threads,
        "parameter_count": parameter_count,
        "model": {
            "chunk_size": chunk_size,
            "use_vae": args.use_vae,
            "dim_model": 128,
            "n_heads": 4,
            "dim_feedforward": 512,
            "n_encoder_layers": 2,
            "n_decoder_layers": 1,
            "latent_dim": 16,
            "n_vae_encoder_layers": 2,
        },
        "wall_time_s": time.perf_counter() - run_started,
        "mean_step_time_s": float(np.mean(step_times)),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "mean_last_20_loss": float(np.mean(losses[-20:])),
    }
    atomic_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
