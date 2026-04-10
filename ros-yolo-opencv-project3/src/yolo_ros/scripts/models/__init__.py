# -*- coding: utf-8 -*-
"""
One-Shot Action Recognition Models

This package contains skeleton-based action recognition models with different
joint formats:

- legacy/: COCO 17-point format (YOLOv8-Pose based)
- ntu25/: NTU RGB+D 25-point format (Azure Kinect DK compatible)
- mediapipe33/: MediaPipe 33-point format (D435i + MediaPipe)
- common/: Shared utilities and class definitions
"""

from .common.ntu_rgbd_classes import NTU_RGBD_60_CLASSES
