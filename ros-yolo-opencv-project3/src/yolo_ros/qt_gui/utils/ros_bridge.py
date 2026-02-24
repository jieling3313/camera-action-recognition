#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 影像橋接器
用於訂閱 ROS 影像主題並轉換為 OpenCV 影像

Author: Claude AI Assistant
Date: 2025-11-23
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ROSImageBridge(QObject):
    """ROS 影像橋接器"""

    # 自訂信號：當接收到新影像時發射
    image_received = pyqtSignal(np.ndarray)

    def __init__(self, topic='/camera/color/image_raw'):
        super().__init__()

        self.topic = topic
        self.bridge = CvBridge()
        self.subscriber = None

        rospy.loginfo(f"ROS Image Bridge initialized for topic: {topic}")

    def start(self):
        """開始訂閱 ROS 影像主題"""
        if self.subscriber is None:
            self.subscriber = rospy.Subscriber(
                self.topic,
                Image,
                self._image_callback,
                queue_size=1
            )
            rospy.loginfo(f"Started subscribing to {self.topic}")

    def stop(self):
        """停止訂閱"""
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None
            rospy.loginfo(f"Stopped subscribing to {self.topic}")

    def _image_callback(self, msg):
        """ROS 影像回調函數"""
        try:
            # 將 ROS 影像訊息轉換為 OpenCV 影像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 發射信號
            self.image_received.emit(cv_image)

        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def __del__(self):
        """解構函數"""
        self.stop()
