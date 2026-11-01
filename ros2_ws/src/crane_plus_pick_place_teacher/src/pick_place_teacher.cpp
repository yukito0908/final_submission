// Instrumented derivative of crane_plus_examples/src/pick_and_place.cpp.
// Copyright 2020 RT Corporation, modifications Copyright 2026.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "control_msgs/msg/joint_trajectory_controller_state.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using namespace std::chrono_literals;

double radians(const double degrees) { return degrees * M_PI / 180.0; }

void require_move(MoveGroupInterface & group, const std::string & label)
{
  const auto result = group.move();
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    throw std::runtime_error("MoveIt failed during " + label + " (code " +
      std::to_string(result.val) + ")");
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto arm_node = rclcpp::Node::make_shared("pick_place_teacher_arm", options);
  auto gripper_node = rclcpp::Node::make_shared("pick_place_teacher_gripper", options);
  auto stage_publisher = arm_node->create_publisher<std_msgs::msg::String>("/pick_place/stage", 10);
  auto action_publisher = arm_node->create_publisher<sensor_msgs::msg::JointState>(
    "/pick_place/action", 10);
  const std::vector<std::string> action_joint_names = {
    "crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint4",
    "crane_plus_joint3", "crane_plus_joint_hand"};
  std::unordered_map<std::string, double> desired_positions;
  auto state_callback = [&desired_positions](
    const control_msgs::msg::JointTrajectoryControllerState & message) {
      for (std::size_t index = 0;
        index < message.joint_names.size() && index < message.reference.positions.size(); ++index)
      {
        desired_positions[message.joint_names[index]] = message.reference.positions[index];
      }
    };
  auto arm_state_subscription = arm_node->create_subscription<
    control_msgs::msg::JointTrajectoryControllerState>(
    "/crane_plus_arm_controller/controller_state", 10, state_callback);
  auto gripper_state_subscription = arm_node->create_subscription<
    control_msgs::msg::JointTrajectoryControllerState>(
    "/crane_plus_gripper_controller/controller_state", 10, state_callback);
  auto action_timer = arm_node->create_wall_timer(50ms, [&]() {
    if (desired_positions.size() < action_joint_names.size()) {
      return;
    }
    sensor_msgs::msg::JointState message;
    message.header.stamp = arm_node->now();
    message.name = action_joint_names;
    for (const auto & name : action_joint_names) {
      message.position.push_back(desired_positions.at(name));
    }
    action_publisher->publish(message);
  });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(arm_node);
  executor.add_node(gripper_node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  int exit_code = 0;
  try {
    MoveGroupInterface arm(arm_node, "arm");
    MoveGroupInterface gripper(gripper_node, "gripper");
    arm.setMaxVelocityScalingFactor(0.8);
    arm.setMaxAccelerationScalingFactor(0.8);
    gripper.setMaxVelocityScalingFactor(0.8);
    gripper.setMaxAccelerationScalingFactor(0.8);
    arm.setPlanningTime(10.0);

    auto publish_stage = [&](const std::string & stage) {
      std_msgs::msg::String message;
      message.data = stage;
      stage_publisher->publish(message);
      RCLCPP_INFO(arm_node->get_logger(), "STAGE=%s", stage.c_str());
      std::this_thread::sleep_for(150ms);
    };
    auto gripper_values = gripper.getCurrentJointValues();
    const double gripper_default = 0.0;
    const double gripper_open = radians(-30.0);
    const double gripper_close = radians(10.0);
    auto set_gripper = [&](const double value, const std::string & label) {
      gripper_values[0] = value;
      gripper.setJointValueTarget(gripper_values);
      require_move(gripper, label);
    };

    const auto subscription_deadline = std::chrono::steady_clock::now() + 5s;
    while (stage_publisher->get_subscription_count() == 0 &&
      std::chrono::steady_clock::now() < subscription_deadline)
    {
      std::this_thread::sleep_for(100ms);
    }
    if (stage_publisher->get_subscription_count() == 0) {
      RCLCPP_WARN(arm_node->get_logger(), "No /pick_place/stage recorder discovered");
    }
    publish_stage("reset");
    arm.setNamedTarget("vertical");
    require_move(arm, "reset_vertical");
    set_gripper(gripper_default, "reset_gripper");

    publish_stage("approach");
    arm.setNamedTarget("home");
    require_move(arm, "approach_home");
    set_gripper(gripper_open, "open_before_approach");

    geometry_msgs::msg::Pose target;
    tf2::Quaternion quaternion;
    target.position.x = 0.0;
    target.position.y = -0.09;
    target.position.z = 0.17;
    quaternion.setRPY(radians(0), radians(90), radians(-90));
    target.orientation = tf2::toMsg(quaternion);
    arm.setPoseTarget(target);
    require_move(arm, "approach_above_cube");

    quaternion.setRPY(radians(0), radians(180), radians(-90));
    target.orientation = tf2::toMsg(quaternion);
    arm.setPoseTarget(target);
    require_move(arm, "align_gripper");

    target.position.z = 0.14;
    arm.setPoseTarget(target);
    require_move(arm, "descend_to_cube");

    publish_stage("grasp");
    set_gripper(gripper_close, "close_on_cube");

    publish_stage("lift");
    target.position.z = 0.17;
    arm.setPoseTarget(target);
    require_move(arm, "lift_cube");

    publish_stage("transport");
    arm.setNamedTarget("home");
    require_move(arm, "transport_home");

    target.position.x = 0.15;
    target.position.y = 0.0;
    target.position.z = 0.06;
    quaternion.setRPY(radians(0), radians(90), radians(0));
    target.orientation = tf2::toMsg(quaternion);
    arm.setPoseTarget(target);
    require_move(arm, "transport_to_goal");

    publish_stage("place");
    set_gripper(gripper_open, "release_cube");

    publish_stage("retreat");
    arm.setNamedTarget("home");
    require_move(arm, "retreat_home");
    arm.setNamedTarget("vertical");
    require_move(arm, "retreat_vertical");
    set_gripper(gripper_default, "finish_gripper");
    publish_stage("complete");
  } catch (const std::exception & error) {
    RCLCPP_ERROR(arm_node->get_logger(), "%s", error.what());
    exit_code = 1;
  }

  executor.cancel();
  if (spin_thread.joinable()) {
    spin_thread.join();
  }
  rclcpp::shutdown();
  return exit_code;
}
