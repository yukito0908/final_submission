import os

from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml


def load_file(package_name, file_path):
    path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_yaml(package_name, file_path):
    path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    node = Node(
        package="crane_plus_pick_place_teacher",
        executable="pick_place_teacher",
        name="pick_place_teacher",
        output="screen",
        parameters=[
            {"robot_description": RobotDescriptionLoader().load()},
            {"robot_description_semantic": load_file(
                "crane_plus_moveit_config", "config/crane_plus.srdf")},
            load_yaml("crane_plus_moveit_config", "config/kinematics.yaml"),
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        SetParameter(name="use_sim_time", value=use_sim_time),
        node,
    ])
