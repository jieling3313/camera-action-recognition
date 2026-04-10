#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骨架預覽元件
用於預覽動作的骨架序列（支援播放動畫）

Author: Claude AI Assistant
Date: 2025-11-23
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap


class SkeletonPreviewWidget(QLabel):
    """骨架預覽元件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1;")

        # 預設顯示文字
        self.setText("請選擇動作進行預覽")

        # 骨架序列資料
        self.skeleton_sequence = None
        self.current_frame_index = 0

        # 播放計時器
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._next_frame)
        self.is_playing = False

        # MediaPipe 骨架連接定義 (33 個關鍵點)
        self.connections = [
            # 臉部
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # 軀幹
            (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            # 左腿
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]

    def load_skeleton_sequence(self, skeleton_sequence):
        """載入骨架序列

        Args:
            skeleton_sequence: numpy array, shape (T, 33, 3) for MediaPipe
                               或 (T, 17, 3) for YOLOv8-Pose
        """
        self.skeleton_sequence = skeleton_sequence
        self.current_frame_index = 0

        # 顯示第一幀
        if skeleton_sequence is not None and len(skeleton_sequence) > 0:
            self._draw_skeleton(skeleton_sequence[0])

    def play(self):
        """開始播放骨架動畫"""
        if self.skeleton_sequence is None:
            return

        self.is_playing = True
        self.play_timer.start(33)  # 約 30 FPS

    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.play_timer.stop()
        self.current_frame_index = 0

        # 重置到第一幀
        if self.skeleton_sequence is not None and len(self.skeleton_sequence) > 0:
            self._draw_skeleton(self.skeleton_sequence[0])

    def _next_frame(self):
        """顯示下一幀"""
        if self.skeleton_sequence is None:
            return

        self.current_frame_index = (self.current_frame_index + 1) % len(self.skeleton_sequence)
        self._draw_skeleton(self.skeleton_sequence[self.current_frame_index])

    def _draw_skeleton(self, keypoints):
        """繪製骨架

        Args:
            keypoints: numpy array, shape (33, 3) 或 (17, 3)
                       格式：(x, y, confidence) 或 (x, y, z)
        """
        # 建立空白影像
        img_size = 400
        image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        if keypoints is None or len(keypoints) == 0:
            self._display_image(image)
            return

        # 正規化座標到影像大小
        normalized_kp = keypoints[:, :2].copy()

        # 找到座標範圍（支援負值座標，如 hip-centering 後的數據）
        min_x, min_y = normalized_kp.min(axis=0)
        max_x, max_y = normalized_kp.max(axis=0)

        # 計算範圍
        range_x = max_x - min_x
        range_y = max_y - min_y

        # 正規化並縮放（保持長寬比）
        if range_x > 1e-6 and range_y > 1e-6:
            # 使用較大的範圍來保持長寬比
            max_range = max(range_x, range_y)

            # 將座標移到正數範圍並正規化
            normalized_kp[:, 0] = (normalized_kp[:, 0] - min_x) / max_range
            normalized_kp[:, 1] = (normalized_kp[:, 1] - min_y) / max_range

            # 縮放到影像大小，留邊距
            margin = 40
            scale = img_size - 2 * margin
            normalized_kp = normalized_kp * scale + margin

            # 置中調整
            center_offset_x = (scale - range_x / max_range * scale) / 2
            center_offset_y = (scale - range_y / max_range * scale) / 2
            normalized_kp[:, 0] += center_offset_x
            normalized_kp[:, 1] += center_offset_y

        # 繪製骨架連接
        num_keypoints = len(keypoints)

        # 根據關鍵點數量選擇連接方式
        if num_keypoints == 33:  # MediaPipe
            connections = self.connections
        elif num_keypoints == 17:  # YOLOv8-Pose (COCO)
            connections = self._get_coco_connections()
        else:
            connections = []

        # 判斷第三個通道是置信度還是 z 座標
        # 如果值普遍在 0-1 之間且接近 1，可能是置信度
        # 如果值範圍較大或包含負值，可能是 z 座標
        third_channel = keypoints[:, 2]
        is_confidence = np.all((third_channel >= 0) & (third_channel <= 1))

        for start_idx, end_idx in connections:
            if start_idx >= num_keypoints or end_idx >= num_keypoints:
                continue

            start_point = tuple(normalized_kp[start_idx].astype(int))
            end_point = tuple(normalized_kp[end_idx].astype(int))

            # 檢查置信度（如果是置信度格式）或直接繪製（如果是 z 座標）
            if is_confidence:
                if keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3:
                    cv2.line(image, start_point, end_point, (0, 200, 0), 2)
            else:
                # z 座標格式，直接繪製所有連接
                cv2.line(image, start_point, end_point, (0, 200, 0), 2)

        # 繪製關鍵點
        for i, kp in enumerate(normalized_kp):
            point = tuple(kp.astype(int))
            if is_confidence:
                if keypoints[i, 2] > 0.3:
                    cv2.circle(image, point, 4, (0, 0, 255), -1)
            else:
                # z 座標格式，繪製所有點
                cv2.circle(image, point, 4, (0, 0, 255), -1)

        # 顯示幀數資訊
        if self.skeleton_sequence is not None:
            total_frames = len(self.skeleton_sequence)
            frame_text = f"Frame: {self.current_frame_index + 1}/{total_frames}"
            cv2.putText(image, frame_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (100, 100, 100), 1, cv2.LINE_AA)

        self._display_image(image)

    def _get_coco_connections(self):
        """取得 COCO 17 關鍵點連接"""
        return [
            (0, 1), (0, 2),     # 鼻子-眼睛
            (1, 3), (2, 4),     # 眼睛-耳朵
            (0, 5), (0, 6),     # 鼻子-肩膀
            (5, 7), (7, 9),     # 左臂
            (6, 8), (8, 10),    # 右臂
            (5, 11), (6, 12),   # 肩膀-臀部
            (11, 13), (13, 15), # 左腿
            (12, 14), (14, 16)  # 右腿
        ]

    def _display_image(self, image):
        """顯示影像"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width

        # BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        q_image = QImage(rgb_image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)

        self.setPixmap(scaled_pixmap)
