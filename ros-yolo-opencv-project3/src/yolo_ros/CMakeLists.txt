cmake_minimum_required(VERSION 3.0.2)
project(yolo_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_generation
)

# --- ADD THIS SECTION ---
# Explicitly add the catkin include directories.
# This helps the compiler find header files like "ros/ros.h".
include_directories(
  ${catkin_INCLUDE_DIRS}
)
# --- END OF ADDITION ---

add_service_files(
  FILES
  TriggerSelection.srv
  MotorControl.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs cv_bridge message_runtime
)

add_executable(host_client src/host_client.cpp)

add_dependencies(host_client ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(host_client
  ${catkin_LIBRARIES}
)

install(PROGRAMS
  scripts/yolo_subscriber.py
  scripts/get_object_position.py
  scripts/interactive_object_selector.py
  scripts/object_selector_server.py
  motor_control/scripts/camera_tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)