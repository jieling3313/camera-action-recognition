#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四個頁面：Body Tracking - 人體追蹤與動作辨識整合

功能:
- 左側：追蹤控制面板（馬達控制、巡航模式）
- 中間：即時影像顯示（YOLO追蹤 + MediaPipe骨架 + 動作辨識）
- 右側：狀態監控與辨識結果

整合特點：
- YOLO 人體偵測用於馬達追蹤
- MediaPipe 骨架提取用於動作辨識
- 同時進行追蹤與姿態辨識

Author: Claude AI Assistant
Date: 2025
"""

import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QGroupBox, QSlider, QSpinBox, QCheckBox,
                              QFrame, QGridLayout, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
import rospy
from std_msgs.msg import Bool, Int32, String, Float32
from sensor_msgs.msg import Image


class CvBridgeSimple:
    """Simple replacement for cv_bridge to avoid NumPy version conflicts"""

    def imgmsg_to_cv2(self, img_msg, desired_encoding="bgr8"):
        """Convert ROS Image message to OpenCV image"""
        dtype = np.uint8
        if img_msg.encoding == "32FC1":
            dtype = np.float32
        elif img_msg.encoding == "16UC1":
            dtype = np.uint16

        img = np.frombuffer(img_msg.data, dtype=dtype)

        if img_msg.encoding in ["rgb8", "bgr8"]:
            img = img.reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding == "rgba8" or img_msg.encoding == "bgra8":
            img = img.reshape((img_msg.height, img_msg.width, 4))
        elif img_msg.encoding in ["mono8", "8UC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        elif img_msg.encoding in ["mono16", "16UC1", "32FC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        else:
            try:
                img = img.reshape((img_msg.height, img_msg.width, 3))
            except:
                img = img.reshape((img_msg.height, img_msg.width))

        if img_msg.encoding == "rgb8" and desired_encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "bgr8" and desired_encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class Page4BodyTracking(QWidget):
    """第四個頁面：Body Tracking - 人體追蹤與動作辨識整合"""

    # Signals for thread-safe updates
    status_updated = pyqtSignal(str)
    person_count_updated = pyqtSignal(int)
    motor_position_updated = pyqtSignal(int)
    recognition_updated = pyqtSignal(str, float)  # action_name, confidence

    def __init__(self):
        super().__init__()

        # CV Bridge
        self.bridge = CvBridgeSimple()

        # ROS initialization
        self.ros_initialized = False
        self._init_ros()

        # Current frame storage
        self.current_tracking_frame = None
        self.current_recognition_frame = None

        # 建立 UI
        self._create_ui()

        # Connect signals
        self.status_updated.connect(self._update_status_display)
        self.person_count_updated.connect(self._update_person_count)
        self.motor_position_updated.connect(self._update_motor_position)
        self.recognition_updated.connect(self._update_recognition_display)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._check_ros_status)
        self.status_timer.start(500)

        # Frame update timer (for video display)
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_video_display)
        self.frame_timer.start(33)  # ~30 FPS

        rospy.loginfo("Page 4 (Body Tracking) initialized")

    def _init_ros(self):
        """Initialize ROS publishers and subscribers"""
        try:
            # Check if ROS master is running
            rospy.get_master().getPid()

            # Publishers for tracking control
            # Note: camera_tracker_node uses relative topic names
            self.tracking_enable_pub = rospy.Publisher('tracking_enable', Bool, queue_size=10)
            self.patrol_enable_pub = rospy.Publisher('patrol_enable', Bool, queue_size=10)
            self.auto_patrol_pub = rospy.Publisher('auto_patrol_enable', Bool, queue_size=10)
            self.patrol_speed_pub = rospy.Publisher('patrol_speed', Float32, queue_size=10)
            self.track_largest_pub = rospy.Publisher('track_largest_person', Bool, queue_size=10)
            self.motor_home_pub = rospy.Publisher('tracking_home', Bool, queue_size=10)
            self.motor_enable_pub = rospy.Publisher('motor_enable', Bool, queue_size=10)

            # Subscribers for status
            # Note: camera_tracker_node uses relative topic names
            rospy.Subscriber('body_tracking_status', String, self._status_callback)
            rospy.Subscriber('detected_person_count', Int32, self._person_count_callback)
            rospy.Subscriber('motor_position', Int32, self._motor_position_callback)

            # Subscribers for video feeds
            # Note: camera_tracker_node uses relative topic names, so we try both
            rospy.Subscriber('tracking_annotated_image', Image, self._tracking_image_callback, queue_size=1)
            rospy.Subscriber('/recognition_display/output_image', Image, self._recognition_image_callback, queue_size=1)

            # Subscriber for recognition results
            rospy.Subscriber('/recognition_display/result', String, self._recognition_result_callback)

            self.ros_initialized = True
            rospy.loginfo("Page 4: ROS initialized")

        except Exception as e:
            rospy.logwarn(f"Page 4: ROS not initialized - {e}")
            self.ros_initialized = False

    def _create_ui(self):
        """建立使用者介面"""
        layout = QVBoxLayout(self)

        # 標題
        title = QLabel("Body Tracking & Action Recognition")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 主要內容區域（三欄布局）
        content_layout = QHBoxLayout()

        # 左側：追蹤控制面板
        content_layout.addWidget(self._create_control_panel(), 1)

        # 中間：影像顯示區域
        content_layout.addWidget(self._create_video_panel(), 2)

        # 右側：狀態與辨識結果
        content_layout.addWidget(self._create_status_panel(), 1)

        layout.addLayout(content_layout)

    def _create_control_panel(self):
        """建立左側控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ===== 追蹤控制 =====
        tracking_group = QGroupBox("Body Tracking Control")
        tracking_group.setFont(QFont("Arial", 11, QFont.Bold))
        tracking_layout = QVBoxLayout()

        # 啟動/停止追蹤
        btn_layout = QHBoxLayout()
        self.btn_start_tracking = QPushButton("Start Tracking")
        self.btn_start_tracking.setStyleSheet(self._button_style("#27ae60", "#229954"))
        self.btn_start_tracking.clicked.connect(self._start_tracking)
        btn_layout.addWidget(self.btn_start_tracking)

        self.btn_stop_tracking = QPushButton("Stop Tracking")
        self.btn_stop_tracking.setStyleSheet(self._button_style("#e74c3c", "#c0392b"))
        self.btn_stop_tracking.clicked.connect(self._stop_tracking)
        btn_layout.addWidget(self.btn_stop_tracking)
        tracking_layout.addLayout(btn_layout)

        # 追蹤最大人體選項
        self.chk_track_largest = QCheckBox("Track Largest Person")
        self.chk_track_largest.setChecked(True)
        self.chk_track_largest.stateChanged.connect(self._toggle_track_largest)
        tracking_layout.addWidget(self.chk_track_largest)

        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)

        # ===== 巡航控制 =====
        patrol_group = QGroupBox("Patrol / Cruise Mode")
        patrol_group.setFont(QFont("Arial", 11, QFont.Bold))
        patrol_layout = QVBoxLayout()

        # 巡航按鈕
        patrol_btn_layout = QHBoxLayout()
        self.btn_start_patrol = QPushButton("Start Patrol")
        self.btn_start_patrol.setStyleSheet(self._button_style("#f39c12", "#d68910"))
        self.btn_start_patrol.clicked.connect(self._start_patrol)
        patrol_btn_layout.addWidget(self.btn_start_patrol)

        self.btn_stop_patrol = QPushButton("Stop Patrol")
        self.btn_stop_patrol.setStyleSheet(self._button_style("#95a5a6", "#7f8c8d"))
        self.btn_stop_patrol.clicked.connect(self._stop_patrol)
        patrol_btn_layout.addWidget(self.btn_stop_patrol)
        patrol_layout.addLayout(patrol_btn_layout)

        # 自動巡航選項
        self.chk_auto_patrol = QCheckBox("Auto-Patrol (when no person)")
        self.chk_auto_patrol.stateChanged.connect(self._toggle_auto_patrol)
        patrol_layout.addWidget(self.chk_auto_patrol)

        # 巡航速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))

        self.patrol_speed_slider = QSlider(Qt.Horizontal)
        self.patrol_speed_slider.setMinimum(10)
        self.patrol_speed_slider.setMaximum(90)
        self.patrol_speed_slider.setValue(36)
        self.patrol_speed_slider.valueChanged.connect(self._update_patrol_speed)
        speed_layout.addWidget(self.patrol_speed_slider)

        self.patrol_speed_spin = QSpinBox()
        self.patrol_speed_spin.setMinimum(10)
        self.patrol_speed_spin.setMaximum(90)
        self.patrol_speed_spin.setValue(36)
        self.patrol_speed_spin.setSuffix(" °/s")
        self.patrol_speed_spin.valueChanged.connect(self._update_patrol_speed_from_spin)
        speed_layout.addWidget(self.patrol_speed_spin)

        patrol_layout.addLayout(speed_layout)
        patrol_group.setLayout(patrol_layout)
        layout.addWidget(patrol_group)

        # ===== 馬達控制 =====
        motor_group = QGroupBox("Motor Control")
        motor_group.setFont(QFont("Arial", 11, QFont.Bold))
        motor_layout = QVBoxLayout()

        motor_btn_layout = QHBoxLayout()
        self.btn_motor_home = QPushButton("Home")
        self.btn_motor_home.setStyleSheet(self._button_style("#9b59b6", "#8e44ad"))
        self.btn_motor_home.clicked.connect(self._home_motor)
        motor_btn_layout.addWidget(self.btn_motor_home)

        self.btn_motor_enable = QPushButton("Enable")
        self.btn_motor_enable.setStyleSheet(self._button_style("#3498db", "#2980b9"))
        self.btn_motor_enable.clicked.connect(self._enable_motor)
        motor_btn_layout.addWidget(self.btn_motor_enable)

        self.btn_motor_disable = QPushButton("Disable")
        self.btn_motor_disable.setStyleSheet(self._button_style("#7f8c8d", "#6c7a89"))
        self.btn_motor_disable.clicked.connect(self._disable_motor)
        motor_btn_layout.addWidget(self.btn_motor_disable)

        motor_layout.addLayout(motor_btn_layout)
        motor_group.setLayout(motor_layout)
        layout.addWidget(motor_group)

        # ===== STOP ALL 按鈕 =====
        self.btn_stop_all = QPushButton("STOP ALL")
        self.btn_stop_all.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #a93226;
            }
        """)
        self.btn_stop_all.clicked.connect(self._stop_all)
        layout.addWidget(self.btn_stop_all)

        layout.addStretch()
        return panel

    def _create_video_panel(self):
        """建立中間影像顯示區域"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 影像來源選擇
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Video Source:"))
        self.video_source_combo = QComboBox()
        self.video_source_combo.addItems([
            "Tracking + Recognition (Split)",
            "Tracking Only",
            "Recognition Only"
        ])
        self.video_source_combo.currentIndexChanged.connect(self._on_video_source_changed)
        source_layout.addWidget(self.video_source_combo)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # 影像顯示標籤
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #bdc3c7; background-color: #2c3e50;")
        self.video_label.setText("Waiting for video feed...\n\nMake sure tracking node is running")
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #bdc3c7;
                background-color: #2c3e50;
                color: #ecf0f1;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.video_label)

        # 影像資訊
        self.video_info_label = QLabel("No video")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        self.video_info_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.video_info_label)

        return panel

    def _create_status_panel(self):
        """建立右側狀態面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ===== 系統狀態 =====
        status_group = QGroupBox("System Status")
        status_group.setFont(QFont("Arial", 11, QFont.Bold))
        status_layout = QGridLayout()

        # ROS 狀態
        self.ros_status_label = QLabel("Checking...")
        self.ros_status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(QLabel("ROS:"), 0, 0)
        status_layout.addWidget(self.ros_status_label, 0, 1)

        # 追蹤模式
        self.mode_label = QLabel("STANDBY")
        self.mode_label.setStyleSheet("font-weight: bold; color: #3498db;")
        status_layout.addWidget(QLabel("Mode:"), 1, 0)
        status_layout.addWidget(self.mode_label, 1, 1)

        # 偵測人數
        self.person_count_label = QLabel("0")
        self.person_count_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        status_layout.addWidget(QLabel("Persons:"), 2, 0)
        status_layout.addWidget(self.person_count_label, 2, 1)

        # 馬達位置
        self.motor_pos_label = QLabel("0")
        status_layout.addWidget(QLabel("Motor Pos:"), 3, 0)
        status_layout.addWidget(self.motor_pos_label, 3, 1)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # ===== 動作辨識結果 =====
        recognition_group = QGroupBox("Action Recognition")
        recognition_group.setFont(QFont("Arial", 11, QFont.Bold))
        recognition_layout = QVBoxLayout()

        # 當前辨識動作
        self.action_label = QLabel("No Action")
        self.action_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
            }
        """)
        recognition_layout.addWidget(self.action_label)

        # 信心度
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #7f8c8d;")
        recognition_layout.addWidget(self.confidence_label)

        # 辨識狀態
        self.recognition_status_label = QLabel("Recognition: Idle")
        self.recognition_status_label.setStyleSheet("color: #95a5a6;")
        recognition_layout.addWidget(self.recognition_status_label)

        recognition_group.setLayout(recognition_layout)
        layout.addWidget(recognition_group)

        # ===== 指令輸出 =====
        command_group = QGroupBox("Command Output")
        command_group.setFont(QFont("Arial", 11, QFont.Bold))
        command_layout = QVBoxLayout()

        self.command_label = QLabel("No Command")
        self.command_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.command_label.setAlignment(Qt.AlignCenter)
        self.command_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        command_layout.addWidget(self.command_label)

        # 指令歷史
        self.command_history_label = QLabel("History: -")
        self.command_history_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        self.command_history_label.setWordWrap(True)
        command_layout.addWidget(self.command_history_label)

        command_group.setLayout(command_layout)
        layout.addWidget(command_group)

        layout.addStretch()
        return panel

    def _button_style(self, bg_color, hover_color):
        """按鈕樣式生成器"""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 8px 15px;
                font-size: 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
        """

    # ===== ROS Callbacks =====
    def _status_callback(self, msg):
        """Handle status updates from tracker node"""
        self.status_updated.emit(msg.data)

    def _person_count_callback(self, msg):
        """Handle person count updates"""
        self.person_count_updated.emit(msg.data)

    def _motor_position_callback(self, msg):
        """Handle motor position updates"""
        self.motor_position_updated.emit(msg.data)

    def _tracking_image_callback(self, msg):
        """Handle tracking annotated image"""
        try:
            self.current_tracking_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to convert tracking image: {e}")

    def _recognition_image_callback(self, msg):
        """Handle recognition output image"""
        try:
            self.current_recognition_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Failed to convert recognition image: {e}")

    def _recognition_result_callback(self, msg):
        """Handle recognition result"""
        result = msg.data
        if result.startswith("COMMAND:"):
            command = result.replace("COMMAND:", "")
            self.command_label.setText(command)
            self.command_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: #27ae60;
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
            # Update history
            current_history = self.command_history_label.text()
            if "History:" in current_history:
                history = current_history.replace("History: ", "")
                if history == "-":
                    history = command
                else:
                    history = f"{command}, {history}"
                    # Limit history length
                    if len(history) > 50:
                        history = history[:50] + "..."
                self.command_history_label.setText(f"History: {history}")
        else:
            # Regular recognition result
            parts = result.split(":")
            if len(parts) >= 2:
                action = parts[0]
                try:
                    confidence = float(parts[1])
                    self.recognition_updated.emit(action, confidence)
                except:
                    pass

    # ===== UI Update Methods =====
    def _update_status_display(self, status_str):
        """Update status display from status string"""
        try:
            status = eval(status_str)
            mode = status.get('mode', 'UNKNOWN')
            self.mode_label.setText(mode)

            if mode == "TRACKING":
                self.mode_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            elif "PATROL" in mode:
                self.mode_label.setStyleSheet("font-weight: bold; color: #f39c12;")
            else:
                self.mode_label.setStyleSheet("font-weight: bold; color: #3498db;")
        except:
            pass

    def _update_person_count(self, count):
        """Update person count display"""
        self.person_count_label.setText(str(count))
        if count > 0:
            self.person_count_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        else:
            self.person_count_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

    def _update_motor_position(self, position):
        """Update motor position display"""
        self.motor_pos_label.setText(str(position))

    def _update_recognition_display(self, action, confidence):
        """Update recognition result display"""
        self.action_label.setText(action)
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")

        if action != "Unknown" and action != "No Action":
            self.action_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: #27ae60;
                    padding: 15px;
                    border-radius: 5px;
                }
            """)
        else:
            self.action_label.setStyleSheet("""
                QLabel {
                    color: #2c3e50;
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                }
            """)

    def _check_ros_status(self):
        """Check ROS connection status"""
        try:
            rospy.get_master().getPid()
            self.ros_status_label.setText("Connected")
            self.ros_status_label.setStyleSheet("font-weight: bold; color: #27ae60;")

            if not self.ros_initialized:
                self._init_ros()
        except:
            self.ros_status_label.setText("Disconnected")
            self.ros_status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
            self.ros_initialized = False

    def _update_video_display(self):
        """Update video display based on selected source"""
        source_index = self.video_source_combo.currentIndex()

        if source_index == 0:  # Split view
            frame = self._create_split_view()
        elif source_index == 1:  # Tracking only
            frame = self.current_tracking_frame
        else:  # Recognition only
            frame = self.current_recognition_frame

        if frame is not None:
            self._display_frame(frame)
            h, w = frame.shape[:2]
            self.video_info_label.setText(f"Resolution: {w}x{h}")
        else:
            self.video_info_label.setText("No video feed")

    def _create_split_view(self):
        """Create split view with tracking and recognition"""
        tracking = self.current_tracking_frame
        recognition = self.current_recognition_frame

        if tracking is None and recognition is None:
            return None

        # If only one is available, use that
        if tracking is None:
            return recognition
        if recognition is None:
            return tracking

        # Resize to same height
        h1, w1 = tracking.shape[:2]
        h2, w2 = recognition.shape[:2]

        target_h = min(h1, h2, 360)  # Limit height for split view

        # Resize both
        scale1 = target_h / h1
        scale2 = target_h / h2

        tracking_resized = cv2.resize(tracking, (int(w1 * scale1), target_h))
        recognition_resized = cv2.resize(recognition, (int(w2 * scale2), target_h))

        # Add labels
        cv2.putText(tracking_resized, "YOLO Tracking", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(recognition_resized, "Action Recognition", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Concatenate horizontally
        combined = np.hstack([tracking_resized, recognition_resized])
        return combined

    def _display_frame(self, frame):
        """Display frame in the video label"""
        if frame is None:
            return

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage and scale to fit
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def _on_video_source_changed(self, index):
        """Handle video source selection change"""
        sources = ["Split View", "Tracking Only", "Recognition Only"]
        rospy.loginfo(f"Video source changed to: {sources[index]}")

    # ===== Control Methods =====
    def _start_tracking(self):
        """Start body tracking"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = True
            self.tracking_enable_pub.publish(msg)
            self.recognition_status_label.setText("Recognition: Active (Tracking)")
            rospy.loginfo("Tracking started - Recognition continues running")

    def _stop_tracking(self):
        """Stop body tracking"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.tracking_enable_pub.publish(msg)
            self.recognition_status_label.setText("Recognition: Ready")
            rospy.loginfo("Tracking stopped")

    def _toggle_track_largest(self, state):
        """Toggle track largest person mode"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = (state == Qt.Checked)
            self.track_largest_pub.publish(msg)

    def _start_patrol(self):
        """Start manual patrol mode"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = True
            self.patrol_enable_pub.publish(msg)
            self.recognition_status_label.setText("Recognition: Active (Patrol)")
            rospy.loginfo("Patrol started")

    def _stop_patrol(self):
        """Stop patrol mode"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.patrol_enable_pub.publish(msg)
            self.recognition_status_label.setText("Recognition: Ready")
            rospy.loginfo("Patrol stopped")

    def _toggle_auto_patrol(self, state):
        """Toggle auto-patrol mode"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = (state == Qt.Checked)
            self.auto_patrol_pub.publish(msg)

    def _update_patrol_speed(self, value):
        """Update patrol speed from slider"""
        self.patrol_speed_spin.blockSignals(True)
        self.patrol_speed_spin.setValue(value)
        self.patrol_speed_spin.blockSignals(False)

        if self.ros_initialized:
            msg = Float32()
            msg.data = float(value)
            self.patrol_speed_pub.publish(msg)

    def _update_patrol_speed_from_spin(self, value):
        """Update patrol speed from spinbox"""
        self.patrol_speed_slider.blockSignals(True)
        self.patrol_speed_slider.setValue(value)
        self.patrol_speed_slider.blockSignals(False)

        if self.ros_initialized:
            msg = Float32()
            msg.data = float(value)
            self.patrol_speed_pub.publish(msg)

    def _home_motor(self):
        """Home the motor"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = True
            self.motor_home_pub.publish(msg)
            rospy.loginfo("Motor homing")

    def _enable_motor(self):
        """Enable the motor"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = True
            self.motor_enable_pub.publish(msg)
            rospy.loginfo("Motor enabled")

    def _disable_motor(self):
        """Disable the motor"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.motor_enable_pub.publish(msg)
            rospy.loginfo("Motor disabled")

    def _stop_all(self):
        """Stop all operations"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.tracking_enable_pub.publish(msg)
            self.patrol_enable_pub.publish(msg)
            self.auto_patrol_pub.publish(msg)
            self.recognition_status_label.setText("Recognition: Ready")
            self.command_label.setText("STOPPED")
            self.command_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: #e74c3c;
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
            rospy.logwarn("All operations stopped")

    # ===== Slot Methods for External Signals =====
    @pyqtSlot()
    def on_stop(self):
        """接收 STOP 信號"""
        rospy.logwarn("Page 4: Received STOP signal")
        self._stop_all()

    @pyqtSlot()
    def on_initial(self):
        """接收 Initial 信號"""
        rospy.loginfo("Page 4: Received Initial signal")
        self.action_label.setText("No Action")
        self.confidence_label.setText("Confidence: --")
        self.command_label.setText("No Command")
        self.command_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.command_history_label.setText("History: -")
