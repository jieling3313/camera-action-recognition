# -*- coding: utf-8 -*-
"""
MediaPipe 33-point Models - D435i + MediaPipe Pose format

This module contains the implementation using MediaPipe 33-point skeleton
format, which provides:
- Full hand tracking (fingers)
- Detailed foot tracking
- Compatible with Intel RealSense D435i + MediaPipe Pose

Files:
- skeleton_model_mediapipe33.py: One-Shot Action Recognition model (MediaPipe 33)
- extract_mediapipe_from_ntu.py: Script to extract MediaPipe skeletons from NTU RGB videos
- train_mediapipe33.py: Training script for MediaPipe 33-point format (to be created)
"""

from .skeleton_model_mediapipe33 import (
    OneShotActionRecognitionMediaPipe,
    MediaPipeSkeletonEmbedding,
    MediaPipeGraph,
    MediaPipeJoint,
    preprocess_mediapipe_skeleton,
    ntu_to_mediapipe_skeleton
)
