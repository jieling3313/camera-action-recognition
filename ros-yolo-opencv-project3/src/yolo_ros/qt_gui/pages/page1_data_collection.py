#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一個頁面：資料收集 (重構版)
功能：
- 點擊按鈕自動彈出 ROS 視窗顯示 D435i 和骨架
- 通過 ROS 服務進行拍照/錄影
- 避免 Qt X11 渲染負擔，提升效能

Author: Claude AI Assistant
Date: 2025-11-24 (Refactored)
"""

import os
import subprocess
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QLineEdit, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Page1DataCollection(QWidget):
    """第一個頁面：資料收集 (使用 ROS 視窗)"""

    def __init__(self):
        super().__init__()

        # 資料儲存路徑
        self.actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"
        os.makedirs(self.actionset_dir, exist_ok=True)

        # 錄影狀態
        self.is_recording = False

        # Camera display node 進程
        self.camera_node_process = None
        self.raw_view_process = None
        self.skeleton_view_process = None

        # ROS 發布器（檔名）
        self.filename_pub = rospy.Publisher(
            '/camera_display/filename',
            String,
            queue_size=1
        )

        # 建立 UI
        self._create_ui()

        rospy.loginfo("Page 1 (Data Collection - ROS Windows) initialized")

    def _create_ui(self):
        """建立使用者介面"""
        layout = QVBoxLayout(self)

        # 標題
        title = QLabel("Data Collection - Capture and Record Actions")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 說明文字
        info_label = QLabel(
            "Click 'Start Display' to launch ROS windows showing camera and skeleton.\n"
            "Use 'Capture' and 'Record' buttons to save data."
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #666; margin: 10px;")
        layout.addWidget(info_label)

        # 視窗控制區域
        display_group = QGroupBox("Display Control")
        display_layout = QHBoxLayout()

        self.btn_start_display = QPushButton("Start Display")
        self.btn_start_display.setFont(QFont("Arial", 12))
        self.btn_start_display.setMinimumHeight(50)
        self.btn_start_display.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_start_display.clicked.connect(self.start_display)
        display_layout.addWidget(self.btn_start_display)

        self.btn_stop_display = QPushButton("Stop Display")
        self.btn_stop_display.setFont(QFont("Arial", 12))
        self.btn_stop_display.setMinimumHeight(50)
        self.btn_stop_display.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.btn_stop_display.clicked.connect(self.stop_display)
        self.btn_stop_display.setEnabled(False)
        display_layout.addWidget(self.btn_stop_display)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 控制區域
        control_group = QGroupBox("Capture / Record Control")
        control_layout = QVBoxLayout()

        # 檔名輸入
        filename_layout = QHBoxLayout()
        filename_label = QLabel("Filename:")
        filename_label.setFont(QFont("Arial", 12))
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("Enter filename (without extension)")
        self.filename_input.setFont(QFont("Arial", 11))
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self.filename_input)
        control_layout.addLayout(filename_layout)

        # 按鈕區域
        button_layout = QHBoxLayout()

        # 拍攝按鈕
        self.btn_capture = QPushButton("Capture")
        self.btn_capture.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_capture.setMinimumHeight(60)
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.btn_capture.clicked.connect(self.capture_image)
        button_layout.addWidget(self.btn_capture)

        # 錄影按鈕
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_record.setMinimumHeight(60)
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        self.btn_record.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.btn_record)

        control_layout.addLayout(button_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 狀態顯示
        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin: 10px;
            }
        """)
        layout.addWidget(self.status_label)

        # 佔位空間
        layout.addStretch()

    def start_display(self):
        """啟動 ROS 相機顯示節點並使用 image_view 顯示"""
        if self.camera_node_process is not None:
            QMessageBox.warning(self, "Warning", "Display is already running")
            return

        try:
            # 準備環境變數（直接從當前環境繼承 DISPLAY）
            env = os.environ.copy()
            # Ensure DISPLAY is set (use current value or default to :0)
            if 'DISPLAY' not in env:
                env['DISPLAY'] = os.environ.get('DISPLAY', ':0')

            # 啟動 camera_display_node（處理影像，不顯示）
            node_script = "/root/catkin_ws/src/yolo_ros/scripts/camera_display_node.py"
            if not os.path.exists(node_script):
                QMessageBox.critical(
                    self, "Error",
                    f"Camera display node not found: {node_script}"
                )
                return

            self.camera_node_process = subprocess.Popen(
                ["python3.10", node_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )

            # 等待 skeleton topic 準備好
            import time
            time.sleep(1)

            # 啟動 image_view 顯示原始影像
            self.raw_view_process = subprocess.Popen(
                ["rosrun", "image_view", "image_view",
                 "image:=/camera/color/image_raw",
                 "_image_transport:=raw"],
                env=env
            )

            # 啟動 image_view 顯示骨架影像
            self.skeleton_view_process = subprocess.Popen(
                ["rosrun", "image_view", "image_view",
                 "image:=/camera/skeleton_image",
                 "_image_transport:=raw"],
                env=env
            )

            self.btn_start_display.setEnabled(False)
            self.btn_stop_display.setEnabled(True)
            self.status_label.setText("Status: Display Running (Check ROS windows)")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    margin: 10px;
                }
            """)

            rospy.loginfo("Camera display node started")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start display: {e}")
            rospy.logerr(f"Failed to start camera display node: {e}")

    def stop_display(self):
        """停止 ROS 相機顯示節點和 image_view 視窗"""
        try:
            # 停止 camera_display_node
            if self.camera_node_process is not None:
                self.camera_node_process.terminate()
                self.camera_node_process.wait(timeout=5)
                self.camera_node_process = None

            # 停止 raw image viewer
            if self.raw_view_process is not None:
                self.raw_view_process.terminate()
                self.raw_view_process.wait(timeout=2)
                self.raw_view_process = None

            # 停止 skeleton image viewer
            if self.skeleton_view_process is not None:
                self.skeleton_view_process.terminate()
                self.skeleton_view_process.wait(timeout=2)
                self.skeleton_view_process = None

            self.btn_start_display.setEnabled(True)
            self.btn_stop_display.setEnabled(False)
            self.status_label.setText("Status: Display Stopped")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    margin: 10px;
                }
            """)

            rospy.loginfo("Camera display node stopped")

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error stopping display: {e}")

    def capture_image(self):
        """呼叫 ROS 服務進行拍照"""
        filename = self.filename_input.text().strip()

        if not filename:
            QMessageBox.warning(self, "Error", "Please enter filename")
            return

        # 發布檔名
        self.filename_pub.publish(filename)
        rospy.sleep(0.1)  # 等待發布完成

        try:
            # 呼叫拍照服務
            rospy.wait_for_service('/camera_display/capture', timeout=2.0)
            capture_service = rospy.ServiceProxy('/camera_display/capture', Trigger)
            response = capture_service()

            if response.success:
                QMessageBox.information(self, "Success", response.message)
                self.status_label.setText(f"Status: Captured - {filename}")
                rospy.loginfo(f"Image captured: {filename}")
            else:
                QMessageBox.warning(self, "Error", response.message)

        except rospy.ROSException as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to call capture service: {e}\n\n"
                "Make sure camera display node is running."
            )
            rospy.logerr(f"Capture service error: {e}")

    def toggle_recording(self):
        """切換錄影狀態"""
        filename = self.filename_input.text().strip()

        if not self.is_recording:
            # 開始錄影
            if not filename:
                QMessageBox.warning(self, "Error", "Please enter filename")
                return

            # 發布檔名
            self.filename_pub.publish(filename)
            rospy.sleep(0.1)

            try:
                rospy.wait_for_service('/camera_display/start_recording', timeout=2.0)
                start_service = rospy.ServiceProxy('/camera_display/start_recording', Trigger)
                response = start_service()

                if response.success:
                    self.is_recording = True
                    self.btn_record.setText("Stop Recording")
                    self.btn_record.setStyleSheet("""
                        QPushButton {
                            background-color: #4CAF50;
                            color: white;
                            border-radius: 5px;
                        }
                        QPushButton:hover {
                            background-color: #45a049;
                        }
                    """)
                    self.status_label.setText(f"Status: Recording - {filename}")
                    self.status_label.setStyleSheet("""
                        QLabel {
                            padding: 10px;
                            background-color: #f44336;
                            color: white;
                            border-radius: 5px;
                            margin: 10px;
                        }
                    """)
                    rospy.loginfo(f"Recording started: {filename}")
                else:
                    QMessageBox.warning(self, "Error", response.message)

            except rospy.ROSException as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to start recording: {e}\n\n"
                    "Make sure camera display node is running."
                )

        else:
            # 停止錄影
            try:
                rospy.wait_for_service('/camera_display/stop_recording', timeout=2.0)
                stop_service = rospy.ServiceProxy('/camera_display/stop_recording', Trigger)
                response = stop_service()

                self.is_recording = False
                self.btn_record.setText("Start Recording")
                self.btn_record.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #e68900;
                    }
                """)

                if response.success:
                    QMessageBox.information(self, "Success", response.message)
                    self.status_label.setText("Status: Recording Stopped")
                    self.status_label.setStyleSheet("""
                        QLabel {
                            padding: 10px;
                            background-color: #f0f0f0;
                            border-radius: 5px;
                            margin: 10px;
                        }
                    """)
                    rospy.loginfo("Recording stopped")

                    # 發布訊息通知 actionset 已更新（讓 Page 2 刷新列表）
                    try:
                        actionset_update_pub = rospy.Publisher('/actionset/updated', String, queue_size=1, latch=True)
                        actionset_update_pub.publish(String(data="actionset_updated"))
                        rospy.loginfo("Published actionset update notification")
                    except Exception as e:
                        rospy.logwarn(f"Failed to publish actionset update: {e}")
                else:
                    QMessageBox.warning(self, "Error", response.message)

            except rospy.ROSException as e:
                QMessageBox.critical(self, "Error", f"Failed to stop recording: {e}")

    def on_stop(self):
        """處理全域 STOP 信號"""
        if self.is_recording:
            self.toggle_recording()

        self.stop_display()
        self.status_label.setText("Status: Stopped")

    def on_initial(self):
        """處理全域 Initial 信號"""
        self.filename_input.clear()
        self.status_label.setText("Status: Initialized")

    def closeEvent(self, event):
        """關閉時清理"""
        self.stop_display()
        event.accept()
