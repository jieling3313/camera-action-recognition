# -*- coding: utf-8 -*-
"""
Legacy Models - COCO 17-point format

This module contains the original implementation using COCO 17-point skeleton
format extracted via YOLOv8-Pose.

Note: This format has limited compatibility with NTU RGB+D training data
due to joint mapping loss (25 -> 17 points).

Files:
- skeleton_model.py: One-Shot Action Recognition model (COCO 17)
- skeleton_extractor.py: YOLOv8-Pose based skeleton extractor
- train_ntu_rgbd.py: Training script with NTU->COCO mapping
"""

from .skeleton_model import OneShotActionRecognition, SkeletonEmbedding, COCOGraph
from .skeleton_extractor import SkeletonExtractor, SkeletonBuffer
