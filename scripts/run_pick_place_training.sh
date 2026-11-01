#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
source /workspace/ros2_ws/install/setup.bash

if [[ ! -f /workspace/artifacts/pick_place/data/rosbags/collection_summary.json ]]; then
  python3 /workspace/scripts/collect_pick_place.py \
    --config /workspace/configs/pick_place.yaml \
    --output /workspace/artifacts/pick_place/data/rosbags
fi

bag_roots=(/workspace/artifacts/pick_place/data/rosbags)
if [[ -f /workspace/artifacts/pick_place/data/rosbags_supplement/collection_summary.json ]]; then
  bag_roots+=(/workspace/artifacts/pick_place/data/rosbags_supplement)
fi

cd /home/ubuntu/pai_training_stage
python=/home/ubuntu/pai_training_stage/.pixi/envs/default/bin/python

"$python" /workspace/scripts/validate_pick_place_bags.py \
  --root "${bag_roots[@]}" \
  --output /workspace/artifacts/pick_place/results/bag_validation_combined.json \
  --allow-invalid

"$python" /workspace/scripts/convert_pick_place_dataset.py \
  --bags "${bag_roots[@]}" \
  --output /workspace/artifacts/pick_place/data/lerobot \
  --repo-id local/crane_plus_pick_place \
  --fps 20 --overwrite

"$python" /workspace/scripts/train_act.py \
  --dataset-root /workspace/artifacts/pick_place/data/lerobot \
  --repo-id local/crane_plus_pick_place \
  --output /workspace/artifacts/pick_place/models/act_pick_place_deterministic \
  --steps 5000 --batch-size 8 --chunk-size 32 --no-use-vae --threads 12
