#!/usr/bin/env python3
"""Persistent JSON-lines ACT inference worker (runs in the LeRobot environment)."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    policy = ACTPolicy.from_pretrained(args.model).to("cpu")
    policy.eval()
    policy.reset()
    emit({"status": "ready", "model": args.model})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "reset":
                policy.reset()
                emit({"status": "reset"})
                continue
            state = torch.tensor(request["state"], dtype=torch.float32).reshape(1, 5)
            batch = {
                "observation.state": state,
                "observation.environment_state": state.clone(),
            }
            with torch.inference_mode():
                action = policy.select_action(batch)[0].cpu().tolist()
            emit({"action": action})
        except Exception as exc:  # Keep the protocol alive and make failures explicit.
            emit({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
