#!/usr/bin/env python3.10
"""
Convert MediaPipe 33 keypoints to COCO 17 keypoints format.

MediaPipe Pose Landmarks (33 points):
https://google.github.io/mediapipe/solutions/pose.html

COCO Keypoints (17 points):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import numpy as np

# MediaPipe to COCO keypoint mapping
# MediaPipe index -> COCO index
MEDIAPIPE_TO_COCO_MAP = {
    0: 0,   # nose
    2: 1,   # left_eye_inner -> left_eye
    5: 2,   # right_eye_inner -> right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
}


def convert_mediapipe_to_coco(mediapipe_skeleton):
    """
    Convert MediaPipe skeleton (33 keypoints) to COCO format (17 keypoints).

    Args:
        mediapipe_skeleton: numpy array of shape (..., 33, 3)
            where last dim is [x, y, visibility]

    Returns:
        coco_skeleton: numpy array of shape (..., 17, 3)
    """
    original_shape = mediapipe_skeleton.shape

    # Handle different input shapes
    if len(original_shape) == 2:
        # Single frame: (33, 3)
        mediapipe_skeleton = mediapipe_skeleton[np.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False

    # Get dimensions
    *batch_dims, num_keypoints, coords = mediapipe_skeleton.shape

    if num_keypoints != 33:
        raise ValueError(f"Expected 33 MediaPipe keypoints, got {num_keypoints}")

    if coords != 3:
        raise ValueError(f"Expected 3 coordinates (x, y, visibility), got {coords}")

    # Create output array
    output_shape = (*batch_dims, 17, 3)
    coco_skeleton = np.zeros(output_shape, dtype=mediapipe_skeleton.dtype)

    # Map keypoints
    for mediapipe_idx, coco_idx in MEDIAPIPE_TO_COCO_MAP.items():
        coco_skeleton[..., coco_idx, :] = mediapipe_skeleton[..., mediapipe_idx, :]

    if squeeze_output:
        coco_skeleton = coco_skeleton[0]

    return coco_skeleton

mediapipe_to_coco_single_frame = convert_mediapipe_to_coco


def convert_skeleton_sequence_file(input_path, output_path):
    """
    Convert a skeleton sequence .npy file from MediaPipe to COCO format.

    Args:
        input_path: Path to input .npy file (MediaPipe format)
        output_path: Path to output .npy file (COCO format)
    """
    # Load MediaPipe skeleton
    mediapipe_skeleton = np.load(input_path)
    print(f"Loaded: {input_path}")
    print(f"  Shape: {mediapipe_skeleton.shape}")

    # Convert to COCO
    coco_skeleton = convert_mediapipe_to_coco(mediapipe_skeleton)
    print(f"Converted to COCO format")
    print(f"  Shape: {coco_skeleton.shape}")

    # Save
    np.save(output_path, coco_skeleton)
    print(f"Saved: {output_path}")

    return coco_skeleton


def main():
    """Test conversion on existing data."""
    import os

    actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"

    # Actions to convert
    actions = ["stop", "left", "right", "test1125_v1", "test1125_stop_v1"]

    print("="*60)
    print("MediaPipe to COCO Converter")
    print("="*60)

    for action in actions:
        action_dir = os.path.join(actionset_dir, action)

        # Find skeleton sequence file
        skeleton_file = os.path.join(action_dir, f"{action}_skeleton_sequence.npy")

        if not os.path.exists(skeleton_file):
            print(f"\n✗ Skipping {action}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {action}")
        print(f"{'='*60}")

        # Create COCO version filename
        coco_file = os.path.join(action_dir, f"{action}_skeleton_sequence_coco17.npy")

        try:
            coco_skeleton = convert_skeleton_sequence_file(skeleton_file, coco_file)
            print(f"✓ Success: {action}")
        except Exception as e:
            print(f"✗ Failed: {action}")
            print(f"  Error: {e}")

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


if __name__ == "__main__":
    main()
