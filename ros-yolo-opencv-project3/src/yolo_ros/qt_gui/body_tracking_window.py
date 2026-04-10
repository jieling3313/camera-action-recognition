#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Body Tracking Window - Independent Qt5 GUI for body tracking with D435i
Features:
- Body tracking control (start/stop)
- Track largest person mode toggle
- Patrol/Cruise mode control
- Auto-patrol mode
- Live video display via external window (ROS image_view)

Author: Claude AI Assistant
Date: 2025
"""

import sys
import os
import subprocess
import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QSlider, QSpinBox, QCheckBox,
    QFrame, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

import rospy
from std_msgs.msg import Bool, Int32, String, Float32


class BodyTrackingWindow(QMainWindow):
    """Independent Body Tracking Control Window"""

    # Signals for thread-safe updates
    status_updated = pyqtSignal(str)
    person_count_updated = pyqtSignal(int)
    motor_position_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Body Tracking Control - D435i + NEMA17")
        self.setGeometry(100, 100, 600, 700)
        self.setMinimumSize(500, 600)

        # External viewer process
        self.viewer_process = None

        # ROS initialization check
        self.ros_initialized = False
        self._init_ros()

        # Build UI
        self._build_ui()

        # Connect signals
        self.status_updated.connect(self._update_status_display)
        self.person_count_updated.connect(self._update_person_count)
        self.motor_position_updated.connect(self._update_motor_position)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._check_ros_status)
        self.status_timer.start(500)

    def _init_ros(self):
        """Initialize ROS publishers and subscribers"""
        try:
            # Check if ROS master is running
            rospy.get_master().getPid()

            # Publishers
            self.tracking_enable_pub = rospy.Publisher('/tracking_enable', Bool, queue_size=10)
            self.patrol_enable_pub = rospy.Publisher('/patrol_enable', Bool, queue_size=10)
            self.auto_patrol_pub = rospy.Publisher('/auto_patrol_enable', Bool, queue_size=10)
            self.patrol_speed_pub = rospy.Publisher('/patrol_speed', Float32, queue_size=10)
            self.track_largest_pub = rospy.Publisher('/track_largest_person', Bool, queue_size=10)
            self.motor_home_pub = rospy.Publisher('/tracking_home', Bool, queue_size=10)
            self.motor_enable_pub = rospy.Publisher('/motor_enable', Bool, queue_size=10)

            # Subscribers
            rospy.Subscriber('/body_tracking_status', String, self._status_callback)
            rospy.Subscriber('/detected_person_count', Int32, self._person_count_callback)
            rospy.Subscriber('/motor_position', Int32, self._motor_position_callback)

            self.ros_initialized = True
            rospy.loginfo("Body Tracking Window: ROS initialized")

        except Exception as e:
            rospy.logwarn(f"Body Tracking Window: ROS not initialized - {e}")
            self.ros_initialized = False

    def _build_ui(self):
        """Build the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # Title
        title = QLabel("Body Tracking Control")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title)

        # Status Group
        main_layout.addWidget(self._create_status_group())

        # Tracking Control Group
        main_layout.addWidget(self._create_tracking_group())

        # Patrol Control Group
        main_layout.addWidget(self._create_patrol_group())

        # Motor Control Group
        main_layout.addWidget(self._create_motor_group())

        # Video Display Group
        main_layout.addWidget(self._create_video_group())

        # Bottom buttons
        main_layout.addWidget(self._create_bottom_buttons())

        main_layout.addStretch()

    def _create_status_group(self):
        """Create status display group"""
        group = QGroupBox("System Status")
        group.setStyleSheet(self._group_style())
        layout = QGridLayout()

        # ROS status
        self.ros_status_label = QLabel("Checking...")
        self.ros_status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(QLabel("ROS Status:"), 0, 0)
        layout.addWidget(self.ros_status_label, 0, 1)

        # Mode status
        self.mode_label = QLabel("STANDBY")
        self.mode_label.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(QLabel("Current Mode:"), 1, 0)
        layout.addWidget(self.mode_label, 1, 1)

        # Person count
        self.person_count_label = QLabel("0")
        self.person_count_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        layout.addWidget(QLabel("Detected Persons:"), 2, 0)
        layout.addWidget(self.person_count_label, 2, 1)

        # Motor position
        self.motor_pos_label = QLabel("0")
        layout.addWidget(QLabel("Motor Position:"), 3, 0)
        layout.addWidget(self.motor_pos_label, 3, 1)

        group.setLayout(layout)
        return group

    def _create_tracking_group(self):
        """Create tracking control group"""
        group = QGroupBox("Body Tracking")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout()

        # Start/Stop tracking button
        btn_layout = QHBoxLayout()

        self.btn_start_tracking = QPushButton("Start Tracking")
        self.btn_start_tracking.setStyleSheet(self._button_style("#27ae60", "#229954"))
        self.btn_start_tracking.clicked.connect(self._start_tracking)
        btn_layout.addWidget(self.btn_start_tracking)

        self.btn_stop_tracking = QPushButton("Stop Tracking")
        self.btn_stop_tracking.setStyleSheet(self._button_style("#e74c3c", "#c0392b"))
        self.btn_stop_tracking.clicked.connect(self._stop_tracking)
        btn_layout.addWidget(self.btn_stop_tracking)

        layout.addLayout(btn_layout)

        # Track largest person checkbox
        self.chk_track_largest = QCheckBox("Track Largest Person (by bounding box area)")
        self.chk_track_largest.setChecked(True)
        self.chk_track_largest.stateChanged.connect(self._toggle_track_largest)
        layout.addWidget(self.chk_track_largest)

        group.setLayout(layout)
        return group

    def _create_patrol_group(self):
        """Create patrol control group"""
        group = QGroupBox("Patrol / Cruise Mode")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout()

        # Patrol buttons
        btn_layout = QHBoxLayout()

        self.btn_start_patrol = QPushButton("Start Patrol")
        self.btn_start_patrol.setStyleSheet(self._button_style("#f39c12", "#d68910"))
        self.btn_start_patrol.clicked.connect(self._start_patrol)
        btn_layout.addWidget(self.btn_start_patrol)

        self.btn_stop_patrol = QPushButton("Stop Patrol")
        self.btn_stop_patrol.setStyleSheet(self._button_style("#95a5a6", "#7f8c8d"))
        self.btn_stop_patrol.clicked.connect(self._stop_patrol)
        btn_layout.addWidget(self.btn_stop_patrol)

        layout.addLayout(btn_layout)

        # Auto-patrol checkbox
        self.chk_auto_patrol = QCheckBox("Auto-Patrol (start patrol when no person detected)")
        self.chk_auto_patrol.stateChanged.connect(self._toggle_auto_patrol)
        layout.addWidget(self.chk_auto_patrol)

        # Patrol speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Patrol Speed:"))

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
        self.patrol_speed_spin.setSuffix(" deg/s")
        self.patrol_speed_spin.valueChanged.connect(self._update_patrol_speed_from_spin)
        speed_layout.addWidget(self.patrol_speed_spin)

        layout.addLayout(speed_layout)

        group.setLayout(layout)
        return group

    def _create_motor_group(self):
        """Create motor control group"""
        group = QGroupBox("Motor Control")
        group.setStyleSheet(self._group_style())
        layout = QHBoxLayout()

        self.btn_motor_home = QPushButton("Home Motor")
        self.btn_motor_home.setStyleSheet(self._button_style("#9b59b6", "#8e44ad"))
        self.btn_motor_home.clicked.connect(self._home_motor)
        layout.addWidget(self.btn_motor_home)

        self.btn_motor_enable = QPushButton("Enable Motor")
        self.btn_motor_enable.setStyleSheet(self._button_style("#3498db", "#2980b9"))
        self.btn_motor_enable.clicked.connect(self._enable_motor)
        layout.addWidget(self.btn_motor_enable)

        self.btn_motor_disable = QPushButton("Disable Motor")
        self.btn_motor_disable.setStyleSheet(self._button_style("#7f8c8d", "#6c7a89"))
        self.btn_motor_disable.clicked.connect(self._disable_motor)
        layout.addWidget(self.btn_motor_disable)

        group.setLayout(layout)
        return group

    def _create_video_group(self):
        """Create video display control group"""
        group = QGroupBox("Video Display")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout()

        info_label = QLabel(
            "Note: Video is displayed in a separate window via ROS image_view.\n"
            "Click 'Open Video Window' to start the viewer."
        )
        info_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        btn_layout = QHBoxLayout()

        self.btn_open_video = QPushButton("Open Video Window")
        self.btn_open_video.setStyleSheet(self._button_style("#1abc9c", "#16a085"))
        self.btn_open_video.clicked.connect(self._open_video_window)
        btn_layout.addWidget(self.btn_open_video)

        self.btn_close_video = QPushButton("Close Video Window")
        self.btn_close_video.setStyleSheet(self._button_style("#95a5a6", "#7f8c8d"))
        self.btn_close_video.clicked.connect(self._close_video_window)
        btn_layout.addWidget(self.btn_close_video)

        layout.addLayout(btn_layout)

        # Video status
        self.video_status_label = QLabel("Video window: Not opened")
        self.video_status_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.video_status_label)

        group.setLayout(layout)
        return group

    def _create_bottom_buttons(self):
        """Create bottom control buttons"""
        frame = QFrame()
        layout = QHBoxLayout(frame)

        self.btn_stop_all = QPushButton("STOP ALL")
        self.btn_stop_all.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.btn_stop_all.clicked.connect(self._stop_all)
        layout.addWidget(self.btn_stop_all)

        return frame

    # Style helpers
    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """

    def _button_style(self, bg_color, hover_color):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 13px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
        """

    # ROS Callbacks
    def _status_callback(self, msg):
        """Handle status updates from tracker node"""
        self.status_updated.emit(msg.data)

    def _person_count_callback(self, msg):
        """Handle person count updates"""
        self.person_count_updated.emit(msg.data)

    def _motor_position_callback(self, msg):
        """Handle motor position updates"""
        self.motor_position_updated.emit(msg.data)

    # UI Update methods
    def _update_status_display(self, status_str):
        """Update status display from status string"""
        try:
            # Parse status dict from string
            status = eval(status_str)
            mode = status.get('mode', 'UNKNOWN')
            self.mode_label.setText(mode)

            # Update mode label color
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

        # Check video window status
        if self.viewer_process is not None:
            poll = self.viewer_process.poll()
            if poll is not None:
                self.viewer_process = None
                self.video_status_label.setText("Video window: Closed")
                self.video_status_label.setStyleSheet("color: #7f8c8d;")

    # Control methods
    def _start_tracking(self):
        """Start body tracking"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = True
            self.tracking_enable_pub.publish(msg)
            rospy.loginfo("Tracking started")

    def _stop_tracking(self):
        """Stop body tracking"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.tracking_enable_pub.publish(msg)
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
            rospy.loginfo("Patrol started")

    def _stop_patrol(self):
        """Stop patrol mode"""
        if self.ros_initialized:
            msg = Bool()
            msg.data = False
            self.patrol_enable_pub.publish(msg)
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

    def _open_video_window(self):
        """Open external video window using ROS image_view"""
        if self.viewer_process is not None:
            rospy.logwarn("Video window already open")
            return

        try:
            # Launch image_view for tracking image
            self.viewer_process = subprocess.Popen(
                ['rosrun', 'image_view', 'image_view', 'image:=/tracking_annotated_image'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            self.video_status_label.setText("Video window: Opened")
            self.video_status_label.setStyleSheet("color: #27ae60;")
            rospy.loginfo("Video window opened")
        except Exception as e:
            rospy.logerr(f"Failed to open video window: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open video window:\n{e}")

    def _close_video_window(self):
        """Close external video window"""
        if self.viewer_process is not None:
            try:
                os.killpg(os.getpgid(self.viewer_process.pid), signal.SIGTERM)
                self.viewer_process = None
                self.video_status_label.setText("Video window: Closed")
                self.video_status_label.setStyleSheet("color: #7f8c8d;")
                rospy.loginfo("Video window closed")
            except Exception as e:
                rospy.logerr(f"Failed to close video window: {e}")

    def _stop_all(self):
        """Stop all operations"""
        if self.ros_initialized:
            # Stop tracking
            msg = Bool()
            msg.data = False
            self.tracking_enable_pub.publish(msg)
            self.patrol_enable_pub.publish(msg)
            self.auto_patrol_pub.publish(msg)
            rospy.logwarn("All operations stopped")

    def closeEvent(self, event):
        """Handle window close"""
        self._stop_all()
        self._close_video_window()
        event.accept()


def main():
    """Main entry point"""
    # Initialize ROS node
    try:
        rospy.init_node('body_tracking_gui', anonymous=True)
    except rospy.exceptions.ROSException:
        print("Warning: ROS node already initialized or ROS master not running")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Create and show window
    window = BodyTrackingWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
