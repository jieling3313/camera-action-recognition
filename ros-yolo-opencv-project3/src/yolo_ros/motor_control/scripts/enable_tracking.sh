#!/bin/bash
# Quick script to enable/disable tracking

if [ "$1" = "on" ] || [ "$1" = "1" ] || [ "$1" = "true" ]; then
    echo "Enabling tracking..."
    rostopic pub /tracking_enable std_msgs/Bool "data: true" -1
elif [ "$1" = "off" ] || [ "$1" = "0" ] || [ "$1" = "false" ]; then
    echo "Disabling tracking..."
    rostopic pub /tracking_enable std_msgs/Bool "data: false" -1
elif [ "$1" = "home" ]; then
    echo "Homing motor..."
    rostopic pub /tracking_home std_msgs/Bool "data: true" -1
else
    echo "Usage: $0 {on|off|home}"
    echo "  on   - Enable tracking"
    echo "  off  - Disable tracking"
    echo "  home - Home the motor"
    exit 1
fi
