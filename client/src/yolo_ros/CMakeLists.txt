cmake_minimum_required(VERSION 3.0.2)
project(yolo_ros)

## C++的 ROS 套件 (*roscpp)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_generation
)

## Service 檔案
add_service_files(
  FILES
  TriggerSelection.srv
)

## 生成 Service 訊息
generate_messages(
  DEPENDENCIES
  geometry_msgs
)

## Catkin 套件設定
catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs cv_bridge message_runtime
)

# 指定要編譯的 C++ 執行檔
add_executable(host_client src/host_client.cpp)

# 確保在編譯 C++ 檔案前，Service 標頭檔已經被生成
add_dependencies(host_client ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# C++ 執行檔所需的函式庫
target_link_libraries(host_client
  ${catkin_LIBRARIES}
)
