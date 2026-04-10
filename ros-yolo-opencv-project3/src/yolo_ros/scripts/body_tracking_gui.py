#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Body Tracking GUI - ROS Node Entry Point
Launches the Body Tracking Qt5 Window

Usage:
    rosrun yolo_ros body_tracking_gui.py

Or via launch file:
    roslaunch yolo_ros body_tracking_system.launch
"""

import sys
import os

# Add qt_gui to path
script_dir = os.path.dirname(os.path.abspath(__file__))
qt_gui_dir = os.path.join(os.path.dirname(script_dir), 'qt_gui')
sys.path.insert(0, qt_gui_dir)

from body_tracking_window import main

if __name__ == '__main__':
    main()
