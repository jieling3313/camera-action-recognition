#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三個頁面：即時辨識 (重構版)
功能：
- 點擊按鈕自動彈出 ROS 視窗顯示辨識結果
- 顯示動作名稱和信心分數表格
- 避免 Qt X11 渲染負擔

Author: Claude AI Assistant
Date: 2025-11-24 (Refactored)
"""

import os
import subprocess
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Page3Recognition(QWidget):
    """第三個頁面：即時辨識 (使用 ROS 視窗)"""

    def __init__(self):
        super().__init__()

        # Recognition display node 進程
        self.recognition_node_process = None

        # 當前辨識結果
        self.current_action = "No recognition"
        self.current_confidence = 0.0

        # ROS 訂閱辨識結果
        self.result_sub = rospy.Subscriber(
            '/recognition_display/result',
            String,
            self.result_callback,
            queue_size=1
        )

        # 建立 UI
        self._create_ui()

        # 定時更新表格
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 每 100ms 更新

        rospy.loginfo("Page 3 (Recognition - ROS Windows) initialized")

    def _create_ui(self):
        """建立使用者介面"""
        layout = QVBoxLayout(self)

        # 標題
        title = QLabel("Live Recognition - Action Recognition Results")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 說明文字
        info_label = QLabel(
            "Click 'Start Recognition' to launch ROS window showing skeleton and recognition results.\n"
            "Make sure you have loaded actions in Page 2 first."
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #666; margin: 10px;")
        layout.addWidget(info_label)

        # 視窗控制區域
        display_group = QGroupBox("Recognition Control")
        display_layout = QHBoxLayout()

        self.btn_start_recognition = QPushButton("Start Recognition")
        self.btn_start_recognition.setFont(QFont("Arial", 12))
        self.btn_start_recognition.setMinimumHeight(50)
        self.btn_start_recognition.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_start_recognition.clicked.connect(self.start_recognition)
        display_layout.addWidget(self.btn_start_recognition)

        self.btn_stop_recognition = QPushButton("Stop Recognition")
        self.btn_stop_recognition.setFont(QFont("Arial", 12))
        self.btn_stop_recognition.setMinimumHeight(50)
        self.btn_stop_recognition.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.btn_stop_recognition.clicked.connect(self.stop_recognition)
        self.btn_stop_recognition.setEnabled(False)
        display_layout.addWidget(self.btn_stop_recognition)

        self.btn_load_actions = QPushButton("Load Actions")
        self.btn_load_actions.setFont(QFont("Arial", 12))
        self.btn_load_actions.setMinimumHeight(50)
        self.btn_load_actions.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.btn_load_actions.clicked.connect(self.load_actions)
        display_layout.addWidget(self.btn_load_actions)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 辨識結果區域
        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout()

        # 當前辨識
        self.current_label = QLabel("Current Recognition: No recognition")
        self.current_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_label.setAlignment(Qt.AlignCenter)
        self.current_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin: 10px;
            }
        """)
        result_layout.addWidget(self.current_label)

        # 信心分數
        self.confidence_label = QLabel("Confidence: 0.0%")
        self.confidence_label.setFont(QFont("Arial", 12))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        result_layout.addWidget(self.confidence_label)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

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

    def start_recognition(self):
        """啟動辨識節點

        注意：如果已經由 roslaunch 啟動了 recognition_display_node_v2，
        則直接使用現有節點，不啟動新進程。
        這樣可以保留 Page 2 設定的觸發動作等設定。
        """
        # 檢查是否已經有 recognition_display_node 在運行
        try:
            # 嘗試呼叫 load_actions 服務來檢查節點是否存在
            rospy.wait_for_service('/recognition_display/load_actions', timeout=1.0)

            # 節點已存在，直接使用
            self.btn_start_recognition.setEnabled(False)
            self.btn_stop_recognition.setEnabled(True)
            self.status_label.setText("Status: Recognition Running (Using existing node)")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    margin: 10px;
                }
            """)

            # 載入動作（確保使用最新設定）
            self.load_actions()

            rospy.loginfo("Using existing recognition_display_node_v2 (preserves trigger settings)")
            return

        except rospy.ROSException:
            # 節點不存在，需要啟動
            pass

        if self.recognition_node_process is not None:
            QMessageBox.warning(self, "Warning", "Recognition is already running")
            return

        try:
            # 啟動 recognition_display_node_v2（支援雙模式辨識）
            node_script = "/root/catkin_ws/src/yolo_ros/scripts/ros_nodes/recognition_display_node_v2.py"

            if not os.path.exists(node_script):
                QMessageBox.critical(
                    self, "Error",
                    f"Recognition display node not found: {node_script}"
                )
                return

            # 使用 subprocess 啟動節點
            self.recognition_node_process = subprocess.Popen(
                ["python3.10", node_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.btn_start_recognition.setEnabled(False)
            self.btn_stop_recognition.setEnabled(True)
            self.status_label.setText("Status: Recognition Running (New node started)")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    margin: 10px;
                }
            """)

            rospy.loginfo("Recognition display node started (new process)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recognition: {e}")
            rospy.logerr(f"Failed to start recognition display node: {e}")

    def stop_recognition(self):
        """停止辨識節點

        注意：如果使用的是 roslaunch 啟動的節點，則不終止它，
        只是切換 UI 狀態。
        """
        # 更新 UI 狀態
        self.btn_start_recognition.setEnabled(True)
        self.btn_stop_recognition.setEnabled(False)
        self.status_label.setText("Status: Recognition Stopped")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin: 10px;
            }
        """)

        # 只有當我們自己啟動的進程時才終止它
        if self.recognition_node_process is not None:
            try:
                self.recognition_node_process.terminate()
                self.recognition_node_process.wait(timeout=5)
                self.recognition_node_process = None
                rospy.loginfo("Recognition display node stopped (process terminated)")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Error stopping recognition: {e}")
        else:
            rospy.loginfo("Recognition UI stopped (roslaunch node still running)")

    def load_actions(self):
        """載入動作集"""
        try:
            rospy.wait_for_service('/recognition_display/load_actions', timeout=10.0)
            load_service = rospy.ServiceProxy('/recognition_display/load_actions', Trigger)
            response = load_service()

            if response.success:
                QMessageBox.information(self, "Success", response.message)
                self.status_label.setText("Status: Actions Loaded")
                rospy.loginfo("Actions loaded successfully")
            else:
                QMessageBox.warning(self, "Error", response.message)

        except rospy.ROSException as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load actions: {e}\n\n"
                "Make sure recognition display node is running."
            )
            rospy.logerr(f"Load actions service error: {e}")

    def result_callback(self, msg):
        """接收辨識結果"""
        try:
            # 格式: "action_name,confidence"
            parts = msg.data.split(',')
            if len(parts) == 2:
                self.current_action = parts[0]
                self.current_confidence = float(parts[1])
        except Exception as e:
            rospy.logerr(f"Error parsing result: {e}")

    def update_display(self):
        """更新顯示"""
        self.current_label.setText(f"Current Recognition: {self.current_action}")
        # 注意：current_confidence 已經是百分比 (0-100)，不需要再乘以 100
        self.confidence_label.setText(f"Confidence: {self.current_confidence:.1f}%")

        # 根據信心分數調整顏色（confidence 是 0-100 的百分比）
        if self.current_confidence > 70:
            color = "#4CAF50"  # 綠色
        elif self.current_confidence > 40:
            color = "#FF9800"  # 橙色
        else:
            color = "#f44336"  # 紅色

        self.current_label.setStyleSheet(f"""
            QLabel {{
                padding: 15px;
                background-color: {color};
                color: white;
                border-radius: 5px;
                margin: 10px;
            }}
        """)

    def on_stop(self):
        """處理全域 STOP 信號"""
        self.stop_recognition()
        self.status_label.setText("Status: Stopped")

    def on_initial(self):
        """處理全域 Initial 信號"""
        self.current_action = "No recognition"
        self.current_confidence = 0.0
        self.status_label.setText("Status: Initialized")

    def closeEvent(self, event):
        """關閉時清理"""
        self.stop_recognition()
        event.accept()
