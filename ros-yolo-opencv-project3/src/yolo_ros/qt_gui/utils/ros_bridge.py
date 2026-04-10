#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 影像橋接器
用於訂閱 ROS 影像主題並轉換為 OpenCV 影像

Author: Claude AI Assistant
Date: 2025-11-23
"""

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import rospy
from sensor_msgs.msg import Image
# 注意: 不使用 cv_bridge，改用 CvBridgeSimple 避免 NumPy 版本衝突


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

    def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
        """Convert OpenCV image to ROS Image message"""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding

        if len(cv_image.shape) == 3:
            img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        else:
            img_msg.step = cv_image.shape[1]

        img_msg.data = cv_image.tobytes()
        return img_msg


class ROSImageBridge(QObject):
    """ROS 影像橋接器"""

    # 自訂信號：當接收到新影像時發射
    image_received = pyqtSignal(np.ndarray)

    def __init__(self, topic='/camera/color/image_raw'):
        super().__init__()

        self.topic = topic
        self.bridge = CvBridgeSimple()
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
