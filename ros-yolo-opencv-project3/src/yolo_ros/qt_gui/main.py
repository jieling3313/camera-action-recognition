#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QT GUI 主程式入口
用於 One-Shot 動作辨識系統的人機介面

Author: Claude AI Assistant
Date: 2025-11-23
"""

import sys
import os

# CRITICAL: Fix OpenCV Qt plugin conflicts BEFORE any imports
# Remove cv2/qt/plugins to prevent Qt plugin loading conflicts
import shutil
cv2_qt_dir = '/usr/local/lib/python3.10/site-packages/cv2/qt'
if os.path.exists(cv2_qt_dir):
    try:
        plugins_path = os.path.join(cv2_qt_dir, 'plugins')
        if os.path.exists(plugins_path):
            if os.path.islink(plugins_path):
                os.unlink(plugins_path)
            elif os.path.isdir(plugins_path):
                shutil.rmtree(plugins_path)
    except:
        pass

import rospy
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """主程式入口"""
    # 初始化 ROS 節點
    rospy.init_node('action_recognition_gui', anonymous=True)

    # 建立 QT 應用程式
    app = QApplication(sys.argv)
    app.setApplicationName("One-Shot Action Recognition System")

    # 建立主視窗
    window = MainWindow()
    window.show()

    # 執行事件循環
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down GUI...")
        sys.exit(0)


if __name__ == '__main__':
    main()
