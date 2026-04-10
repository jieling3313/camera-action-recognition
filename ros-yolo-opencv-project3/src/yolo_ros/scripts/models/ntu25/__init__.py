# -*- coding: utf-8 -*-
"""
NTU 25-point Models - Native NTU RGB+D / Azure Kinect DK format

This module contains the implementation using the native NTU RGB+D 25-point
skeleton format, which is directly compatible with:
- NTU RGB+D training dataset (Kinect v2)
- Azure Kinect DK (32 points -> 25 points mapping)

Files:
- skeleton_model_ntu25.py: One-Shot Action Recognition model (NTU 25)
- skeleton_extractor_azure_kinect.py: Azure Kinect DK skeleton extractor
- train_ntu25.py: Training script for NTU 25-point format (to be created)
"""

from .skeleton_model_ntu25 import (
    OneShotActionRecognitionNTU25,
    NTUSkeletonEmbedding,
    NTUGraph,
    NTUJoint,
    preprocess_ntu_skeleton,
    load_ntu_skeleton_file
)

from .skeleton_extractor_azure_kinect import (
    AzureKinectSkeletonExtractor,
    AzureKinectSkeletonBuffer,
    AzureKinectJoint,
    AZURE_TO_NTU_MAP,
    azure_to_ntu_skeleton
)
