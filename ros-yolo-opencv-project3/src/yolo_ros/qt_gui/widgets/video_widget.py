#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影像顯示元件
用於顯示 ROS 影像訊息或 OpenCV 影像

Author: Claude AI Assistant
Date: 2025-11-23
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap


class VideoWidget(QLabel):
    """影像顯示元件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")

        # 預設顯示文字
        self.setText("等待影像...")

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """更新顯示的影像幀

        Args:
            frame: OpenCV 影像 (numpy array, BGR format)
        """
        if frame is None:
            return

        # 轉換為 QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        q_image = QImage(rgb_frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)

        # 縮放以適應元件大小，保持長寬比
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)

        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """視窗大小改變事件"""
        super().resizeEvent(event)
        # 重新縮放影像
        if self.pixmap():
            scaled_pixmap = self.pixmap().scaled(self.size(), Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
