#!/bin/bash
# Setup Arduino Libraries - 在容器內執行此腳本生成 ros_lib

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Arduino ROS Libraries Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running inside container
if [ ! -f "/opt/ros/noetic/setup.bash" ]; then
    echo -e "${RED}錯誤: 此腳本必須在 Docker 容器內執行！${NC}"
    echo "請使用: docker exec -it ros-noetic-yolo-dev bash"
    exit 1
fi

# Source ROS
echo -e "${YELLOW}[1/5] Sourcing ROS environment...${NC}"
source /opt/ros/noetic/setup.bash
echo -e "${GREEN}✓${NC} ROS environment loaded"
echo ""

# Check if rosserial_arduino is installed
echo -e "${YELLOW}[2/5] Checking rosserial_arduino...${NC}"
if ! rospack find rosserial_arduino &>/dev/null; then
    echo -e "${RED}✗${NC} rosserial_arduino not found!"
    echo "Please rebuild the Docker container with updated Dockerfile"
    exit 1
fi
echo -e "${GREEN}✓${NC} rosserial_arduino found: $(rospack find rosserial_arduino)"
echo ""

# Create Arduino libraries directory
echo -e "${YELLOW}[3/5] Creating Arduino libraries directory...${NC}"
ARDUINO_LIBS_DIR="/root/Arduino/libraries"
mkdir -p "$ARDUINO_LIBS_DIR"
cd "$ARDUINO_LIBS_DIR"
echo -e "${GREEN}✓${NC} Directory created: $ARDUINO_LIBS_DIR"
echo ""

# Generate ros_lib
echo -e "${YELLOW}[4/5] Generating ros_lib...${NC}"
if [ -d "$ARDUINO_LIBS_DIR/ros_lib" ]; then
    echo -e "${YELLOW}⚠${NC}  ros_lib already exists. Removing old version..."
    rm -rf "$ARDUINO_LIBS_DIR/ros_lib"
fi

rosrun rosserial_arduino make_libraries.py .

if [ ! -d "$ARDUINO_LIBS_DIR/ros_lib" ]; then
    echo -e "${RED}✗${NC} Failed to generate ros_lib!"
    exit 1
fi
echo -e "${GREEN}✓${NC} ros_lib generated successfully"
echo ""

# Verify ros_lib
echo -e "${YELLOW}[5/5] Verifying ros_lib...${NC}"
if [ -f "$ARDUINO_LIBS_DIR/ros_lib/ros.h" ]; then
    echo -e "${GREEN}✓${NC} ros.h found"
else
    echo -e "${RED}✗${NC} ros.h not found!"
    exit 1
fi

# Count message types
MSG_COUNT=$(find "$ARDUINO_LIBS_DIR/ros_lib" -name "*.h" | wc -l)
echo -e "${GREEN}✓${NC} Generated $MSG_COUNT header files"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "ros_lib location: $ARDUINO_LIBS_DIR/ros_lib"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Copy ros_lib to host machine:"
echo -e "   ${BLUE}docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/${NC}"
echo ""
echo "2. Or copy to motor_control/arduino directory:"
echo -e "   ${BLUE}cp -r $ARDUINO_LIBS_DIR/ros_lib /root/catkin_ws/src/yolo_ros/motor_control/arduino/${NC}"
echo ""
echo "3. Then upload Arduino sketch using Arduino IDE on host machine"
echo ""
