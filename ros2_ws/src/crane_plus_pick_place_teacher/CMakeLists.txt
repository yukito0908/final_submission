cmake_minimum_required(VERSION 3.8)
project(crane_plus_pick_place_teacher)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(pick_place_teacher src/pick_place_teacher.cpp)
ament_target_dependencies(
  pick_place_teacher
  control_msgs
  geometry_msgs
  moveit_ros_planning_interface
  rclcpp
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
)

install(TARGETS pick_place_teacher DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
ament_package()
