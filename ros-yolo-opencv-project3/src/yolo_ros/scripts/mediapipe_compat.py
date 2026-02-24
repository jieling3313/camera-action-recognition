#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
MediaPipe 兼容層 - 讓舊版 solutions API 在新版 MediaPipe (0.10.x+) 上工作
"""

import mediapipe as mp
import numpy as np
from typing import NamedTuple

class PoseLandmark:
    """模擬 mp.solutions.pose.PoseLandmark"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class Landmark(NamedTuple):
    x: float
    y: float
    z: float
    visibility: float = 1.0

class PoseLandmarks:
    def __init__(self, landmarks):
        self.landmark = [
            Landmark(lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0))
            for lm in landmarks
        ]

class PoseResults:
    def __init__(self):
        self.pose_landmarks = None
        self.pose_world_landmarks = None

class Pose:
    """兼容舊版 mp.solutions.pose.Pose API"""

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # 創建 PoseLandmarker
        base_options = python.BaseOptions(
            model_asset_path='/root/.mediapipe/models/pose_landmarker.task'
        )

        # 根據參數設定運行模式
        if static_image_mode:
            running_mode = vision.RunningMode.IMAGE
        else:
            running_mode = vision.RunningMode.VIDEO

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self.frame_counter = 0

    def process(self, image):
        """處理影像並返回姿勢估計結果"""
        # 轉換 BGR 到 RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image[:, :, ::-1].copy()
        else:
            rgb_image = image.copy()

        # 創建 MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # 執行檢測
        if self.static_image_mode:
            detection_result = self.detector.detect(mp_image)
        else:
            # VIDEO 模式需要時間戳
            timestamp_ms = self.frame_counter
            self.frame_counter += 33  # ~30fps
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        # 轉換結果為舊版格式
        results = PoseResults()

        if detection_result.pose_landmarks:
            # 取第一個檢測到的姿勢
            landmarks = detection_result.pose_landmarks[0]
            results.pose_landmarks = PoseLandmarks(landmarks)

        if detection_result.pose_world_landmarks:
            world_landmarks = detection_result.pose_world_landmarks[0]
            results.pose_world_landmarks = PoseLandmarks(world_landmarks)

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()

class DrawingUtils:
    """兼容舊版 mp.solutions.drawing_utils"""

    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # 臉部左側
        (0, 4), (4, 5), (5, 6), (6, 8),  # 臉部右側
        (9, 10),  # 嘴巴
        (11, 12),  # 肩膀
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # 左臂
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # 右臂
        (11, 23), (12, 24), (23, 24),  # 軀幹
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # 左腿
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # 右腿
    ]

    @staticmethod
    def draw_landmarks(
        image,
        landmark_list,
        connections=None,
        landmark_drawing_spec=None,
        connection_drawing_spec=None
    ):
        """繪製關鍵點和連接線"""
        import cv2

        if landmark_list is None or not hasattr(landmark_list, 'landmark'):
            return

        h, w, _ = image.shape
        landmarks = landmark_list.landmark

        # 繪製連接線
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))

                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # 繪製關鍵點
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# 創建兼容的 solutions 模塊
class Solutions:
    class pose:
        Pose = Pose
        PoseLandmark = PoseLandmark
        POSE_CONNECTIONS = DrawingUtils.POSE_CONNECTIONS

    drawing_utils = DrawingUtils()

# 將兼容層注入到 mediapipe
if not hasattr(mp, 'solutions'):
    mp.solutions = Solutions()
