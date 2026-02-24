#!/bin/bash
# System Test Script - Test each component of the tracking system

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Camera Tracking System Test Script"
echo "=========================================="
echo ""

# Function to check if a topic exists
check_topic() {
    local topic=$1
    local timeout=5
    local result=$(timeout $timeout rostopic list 2>/dev/null | grep "^${topic}$")

    if [ -n "$result" ]; then
        echo -e "${GREEN}✓${NC} Topic $topic exists"
        return 0
    else
        echo -e "${RED}✗${NC} Topic $topic NOT found"
        return 1
    fi
}

# Function to check if a node is running
check_node() {
    local node=$1
    local result=$(rosnode list 2>/dev/null | grep "$node")

    if [ -n "$result" ]; then
        echo -e "${GREEN}✓${NC} Node $node is running"
        return 0
    else
        echo -e "${RED}✗${NC} Node $node NOT running"
        return 1
    fi
}

# Test 1: Check if roscore is running
echo -e "${YELLOW}[1/8] Checking roscore...${NC}"
if pgrep -x "rosmaster" > /dev/null; then
    echo -e "${GREEN}✓${NC} roscore is running"
else
    echo -e "${RED}✗${NC} roscore is NOT running"
    echo "Please start roscore first: roscore"
    exit 1
fi
echo ""

# Test 2: Check Arduino connection
echo -e "${YELLOW}[2/8] Checking Arduino connection...${NC}"
if [ -e "/dev/ttyACM0" ]; then
    echo -e "${GREEN}✓${NC} Arduino device found at /dev/ttyACM0"
elif [ -e "/dev/ttyUSB0" ]; then
    echo -e "${GREEN}✓${NC} Arduino device found at /dev/ttyUSB0"
else
    echo -e "${RED}✗${NC} No Arduino device found"
    echo "Please check USB connection"
fi
echo ""

# Test 3: Check rosserial node
echo -e "${YELLOW}[3/8] Checking rosserial node...${NC}"
check_node "/serial_node"
echo ""

# Test 4: Check motor topics
echo -e "${YELLOW}[4/8] Checking motor topics...${NC}"
check_topic "/motor_position"
check_topic "/motor_status"
check_topic "/motor_step_command"
check_topic "/motor_enable"
check_topic "/motor_home"
echo ""

# Test 5: Check camera topics
echo -e "${YELLOW}[5/8] Checking camera topics...${NC}"
check_topic "/camera/color/image_raw"
check_topic "/camera/color/camera_info"
echo ""

# Test 6: Check tracking node
echo -e "${YELLOW}[6/8] Checking tracking node...${NC}"
check_node "/camera_tracker_node"
echo ""

# Test 7: Check tracking topics
echo -e "${YELLOW}[7/8] Checking tracking topics...${NC}"
check_topic "/tracking_enable"
check_topic "/tracking_home"
check_topic "/tracking_annotated_image"
echo ""

# Test 8: Test motor movement (optional)
echo -e "${YELLOW}[8/8] Motor movement test (optional)${NC}"
read -p "Do you want to test motor movement? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Enabling motor..."
    rostopic pub /motor_enable std_msgs/Bool "data: true" -1
    sleep 1

    echo "Moving motor +50 steps..."
    rostopic pub /motor_step_command std_msgs/Int32 "data: 50" -1
    sleep 2

    echo "Moving motor -50 steps (returning)..."
    rostopic pub /motor_step_command std_msgs/Int32 "data: -50" -1
    sleep 2

    echo -e "${GREEN}✓${NC} Motor movement test completed"
else
    echo "Motor movement test skipped"
fi
echo ""

# Summary
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo ""
echo "If all tests passed, you can start tracking with:"
echo "  rostopic pub /tracking_enable std_msgs/Bool \"data: true\" -1"
echo ""
echo "Or use the convenience script:"
echo "  ./enable_tracking.sh on"
echo ""
echo "To view the tracking image:"
echo "  rosrun image_view image_view image:=/tracking_annotated_image"
echo ""
