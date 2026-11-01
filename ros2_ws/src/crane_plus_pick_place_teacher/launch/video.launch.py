"""Launch CRANE+ V2 Gazebo with a submission-video-friendly GUI."""

import os

from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    gazebo_share = get_package_share_directory("crane_plus_gazebo")
    teacher_share = get_package_share_directory("crane_plus_pick_place_teacher")
    environment = {
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": os.environ["LD_LIBRARY_PATH"],
        "IGN_GAZEBO_RESOURCE_PATH": os.path.dirname(
            get_package_share_directory("crane_plus_description")
        ),
    }
    gazebo = ExecuteProcess(
        cmd=[
            "ign gazebo -r",
            os.path.join(gazebo_share, "worlds", "table.sdf"),
            "--gui-config",
            os.path.join(teacher_share, "config", "video_gui.config"),
        ],
        output="screen",
        additional_env=environment,
        shell=True,
    )
    loader = RobotDescriptionLoader()
    loader.use_gazebo = "true"
    loader.use_ignition = "true"
    loader.gz_control_config_package = "crane_plus_control"
    loader.gz_control_config_file_path = "config/crane_plus_controllers.yaml"
    description = loader.load()
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("crane_plus_moveit_config"),
            "/launch/run_move_group.launch.py",
        ]),
        launch_arguments={"loaded_description": description}.items(),
    )
    return LaunchDescription([
        SetParameter(name="use_sim_time", value=True),
        gazebo,
        Node(package="ros_gz_sim", executable="create", output="screen", arguments=[
            "-topic", "/robot_description", "-name", "crane_plus", "-z", "1.015",
            "-allow_renaming", "true",
        ]),
        move_group,
        ExecuteProcess(cmd=["ros2 run controller_manager spawner joint_state_broadcaster"], shell=True),
        ExecuteProcess(cmd=["ros2 run controller_manager spawner crane_plus_arm_controller"], shell=True),
        ExecuteProcess(cmd=["ros2 run controller_manager spawner crane_plus_gripper_controller"], shell=True),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
        ]),
    ])
