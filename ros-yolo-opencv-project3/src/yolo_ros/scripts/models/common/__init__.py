# -*- coding: utf-8 -*-
"""
Common Utilities - Shared modules for all model formats

This module contains shared utilities and definitions used across
different skeleton formats.

Files:
- ntu_rgbd_classes.py: NTU RGB+D 60/120 action class definitions
- mediapipe_compat.py: MediaPipe compatibility layer for new API
- convert_mediapipe_to_coco.py: Format conversion utilities
"""

from .ntu_rgbd_classes import NTU_RGBD_60_CLASSES
from .mediapipe_compat import Pose, PoseLandmark, DrawingUtils
