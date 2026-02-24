#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
ROS-compatible Qt GUI launcher script
Wrapper that launches the Qt GUI from scripts directory
"""

import sys
import os

# Add qt_gui directory to Python path
qt_gui_path = os.path.join(os.path.dirname(__file__), '..', 'qt_gui')
sys.path.insert(0, os.path.abspath(qt_gui_path))

# Import and run the main GUI
from main import main

if __name__ == '__main__':
    main()
