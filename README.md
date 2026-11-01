# CRANE+ V2 ACT Pick and Place

ROS 2 Humble / Gazebo上のCRANE+ V2に、5 cmブロックの把持・持上げ・搬送・配置をACT模倣学習で実行させるPhysical AI最終課題です。

教師制御ではなく、収集した20エピソードからCPU学習したACTをGazeboへデプロイしています。正式評価は5試行すべてで接近・把持・持上げ・搬送・目標領域への配置に成功しました。

## Results

- ACT評価: 5/5成功
- 平均最大持上げ: 17.96 ± 1.10 cm
- 平均平面移動: 32.40 ± 0.53 cm
- Dataset: 20 episodes / 12,444 frames / 20 Hz
- Training: CPU 5,000 steps / 670,243 parameters / 277秒
- Video: 97秒、H.264、ACT推論による実行映像

## Final deliverables

- 最終モデル: `artifacts/pick_place/models/act_pick_place_deterministic/`
- Dataset: `artifacts/pick_place/data/lerobot/`
- 正式評価: `artifacts/pick_place/results/evaluation_final/`

## Repository layout

```text
assets/                         Gazebo用の開始・目標マーカー
configs/pick_place.yaml         収集・評価条件
ros2_ws/src/                    教師ノードと動画用launch
scripts/                        収集、変換、学習、評価、動画・PDF生成
tests/                          軌道補間とDataset変換の単体テスト
artifacts/pick_place/
  data/                         rosbag2とLeRobotDataset
  models/                       最終ACT checkpoint
  results/                      評価JSON/CSVとグラフ
  media/                        レポート画像
output/pdf/                     最終PDF
```

`build/`、`install/`、`log/`、Pixi環境、Python cacheはGit管理しません。講義用リポジトリとLeRobot環境はDocker側の `/home/ubuntu/airobot_ws` と `/home/ubuntu/pai_training_stage` を再利用します。

## Environment

- Windows 11 / Docker Desktop
- Container: `physical_ai_final`
- ROS 2 Humble / Gazebo Ignition / MoveIt 2
- CRANE+ V2
- LeRobot / PyTorch / ACT
- GPU不使用

## Build

```bash
docker exec physical_ai_final bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/ubuntu/airobot_ws/install/setup.bash
  cd /workspace/ros2_ws
  colcon build --packages-select crane_plus_pick_place_teacher --symlink-install
'
```

## Gazebo

通常表示:

```bash
docker exec -d -e DISPLAY=:0 -e XAUTHORITY=/home/ubuntu/.Xauthority \
  physical_ai_final bash -lc '
    source /opt/ros/humble/setup.bash
    source /home/ubuntu/airobot_ws/install/setup.bash
    ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py
  '
```

動画用表示:

```bash
docker exec -d -e DISPLAY=:0 -e XAUTHORITY=/home/ubuntu/.Xauthority \
  physical_ai_final bash -lc '
    source /opt/ros/humble/setup.bash
    source /home/ubuntu/airobot_ws/install/setup.bash
    source /workspace/ros2_ws/install/setup.bash
    ros2 launch crane_plus_pick_place_teacher video.launch.py
  '
```

## Dataset validation and conversion

```bash
docker exec physical_ai_final bash -lc '
  cd /home/ubuntu/pai_training_stage
  .pixi/envs/default/bin/python /workspace/scripts/validate_pick_place_bags.py \
    --root /workspace/artifacts/pick_place/data/rosbags \
           /workspace/artifacts/pick_place/data/rosbags_supplement \
    --output /workspace/artifacts/pick_place/results/bag_validation_combined.json \
    --allow-invalid

  .pixi/envs/default/bin/python /workspace/scripts/convert_pick_place_dataset.py \
    --bags /workspace/artifacts/pick_place/data/rosbags \
           /workspace/artifacts/pick_place/data/rosbags_supplement \
    --output /workspace/artifacts/pick_place/data/lerobot \
    --repo-id local/crane_plus_pick_place --fps 20 --overwrite
'
```

## CPU training

```bash
docker exec physical_ai_final bash -lc '
  cd /home/ubuntu/pai_training_stage
  .pixi/envs/default/bin/python /workspace/scripts/train_act.py \
    --dataset-root /workspace/artifacts/pick_place/data/lerobot \
    --repo-id local/crane_plus_pick_place \
    --output /workspace/artifacts/pick_place/models/act_pick_place_deterministic \
    --steps 5000 --batch-size 8 --chunk-size 32 --no-use-vae --threads 12
'
```

## ACT evaluation

Gazeboと3 controllerがactiveな状態で実行します。

```bash
docker exec physical_ai_final bash -lc '
  source /opt/ros/humble/setup.bash
  source /home/ubuntu/airobot_ws/install/setup.bash
  python3 /workspace/scripts/evaluate_pick_place_policy.py \
    --model /workspace/artifacts/pick_place/models/act_pick_place_deterministic \
    --output /workspace/artifacts/pick_place/results/evaluation_final \
    --trials 5 --run-duration 70 --max-step 0.50
'
```

## Tests

```bash
docker exec physical_ai_final bash -lc '
  source /opt/ros/humble/setup.bash
  cd /home/ubuntu/pai_training_stage
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
  .pixi/envs/default/bin/pytest -q /workspace/tests
'
```

## Scope and limitations

ACT入力は5次元関節状態のみで、ブロック位置は固定です。5試行中3試行では解放後の最終高さがテーブル面より2.5 cm高く残りました。位置変更への一般化や厳密な静置判定は今後の課題です。詳細はレポートに記録しています。

## Upstream projects

- [matsuolab/pai_training](https://github.com/matsuolab/pai_training)
- [matsuolab/physicalai_teleop](https://github.com/matsuolab/physicalai_teleop)
- [AI-Robot-Book-Humble/docker-ros2-desktop-ai-robot-book-humble](https://github.com/AI-Robot-Book-Humble/docker-ros2-desktop-ai-robot-book-humble)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)

外部へのpushや提出サイトへのアップロードは、このリポジトリ内では実行していません。
