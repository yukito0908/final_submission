#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
set -u
export DISPLAY=${DISPLAY:-:0}
exec ros2 launch crane_plus_gazebo no_moveit_crane_plus_with_table.launch.py
