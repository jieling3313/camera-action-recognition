#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 骨骼處理器
用於從影像中提取骨骼關鍵點

Author: Claude AI Assistant
Date: 2025-11-23
"""

import cv2
import numpy as np
import mediapipe as mp
import rospy


class MediaPipeProcessor:
    """MediaPipe 骨骼處理器"""

    def __init__(self):
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        rospy.loginfo("MediaPipe Pose initialized")

    def process(self, image):
        """處理影像並提取骨骼

        Args:
            image: OpenCV 影像 (BGR format)

        Returns:
            skeleton_image: 繪製骨骼後的影像
            keypoints: 關鍵點陣列 (33, 3) - (x, y, visibility)
        """
        if image is None:
            return None, None

        # 轉換為 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe 處理
        results = self.pose.process(image_rgb)

        # 複製影像以繪製骨骼
        skeleton_image = image.copy()

        keypoints = None

        if results.pose_landmarks:
            # 繪製骨骼
            self.mp_drawing.draw_landmarks(
                skeleton_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 提取關鍵點座標
            keypoints = []
            height, width = image.shape[:2]

            for landmark in results.pose_landmarks.landmark:
                # 正規化座標 (0-1)
                x = landmark.x
                y = landmark.y
                visibility = landmark.visibility

                keypoints.append([x, y, visibility])

            keypoints = np.array(keypoints)

        return skeleton_image, keypoints

    def __del__(self):
        """解構函數"""
        if hasattr(self, 'pose'):
            self.pose.close()
