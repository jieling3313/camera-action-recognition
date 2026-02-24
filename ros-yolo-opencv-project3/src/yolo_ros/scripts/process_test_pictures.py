#!/usr/bin/env python3.10
"""
Process test_picture images with MediaPipe to create actionset data.

Usage:
    python3.10 process_test_pictures.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Paths
TEST_PICTURE_DIR = "/root/catkin_ws/src/yolo_ros/test_picture"
ACTIONSET_DIR = "/root/catkin_ws/src/yolo_ros/actionset"

# Action labels
ACTION_LABELS = {
    "stop": [1, 2, 3, 4],
    "left": [5, 6, 7],
    "right": [8, 9]
}


def extract_skeleton_from_image(image_path, pose):
    """
    Extract MediaPipe skeleton from an image.

    Args:
        image_path: Path to image file
        pose: MediaPipe Pose instance

    Returns:
        skeleton_data: numpy array of shape (33, 3) or None if no pose detected
        annotated_image: image with skeleton drawn
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None, None

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"Warning: No pose detected in {image_path}")
        return None, None

    # Extract skeleton coordinates
    skeleton = []
    h, w = image.shape[:2]

    for landmark in results.pose_landmarks.landmark:
        # Convert normalized coordinates to pixel coordinates
        x = landmark.x * w
        y = landmark.y * h
        visibility = landmark.visibility
        skeleton.append([x, y, visibility])

    skeleton_data = np.array(skeleton, dtype=np.float32)  # Shape: (33, 3)

    # Draw skeleton on image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

    return skeleton_data, annotated_image


def process_action(action_name, image_numbers, pose):
    """
    Process a set of images for one action.

    Args:
        action_name: Name of the action (stop, left, right)
        image_numbers: List of image numbers to process
        pose: MediaPipe Pose instance
    """
    print(f"\n{'='*60}")
    print(f"Processing action: {action_name}")
    print(f"Images: {image_numbers}")
    print(f"{'='*60}")

    # Create action directory
    action_dir = os.path.join(ACTIONSET_DIR, action_name)
    os.makedirs(action_dir, exist_ok=True)

    # Collect all skeletons
    all_skeletons = []

    for img_num in image_numbers:
        img_path = os.path.join(TEST_PICTURE_DIR, f"{img_num}.jpg")

        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue

        print(f"  Processing {img_num}.jpg...")

        # Extract skeleton
        skeleton_data, annotated_image = extract_skeleton_from_image(img_path, pose)

        if skeleton_data is None:
            continue

        all_skeletons.append(skeleton_data)

        # Save individual skeleton image
        skeleton_img_path = os.path.join(action_dir, f"{action_name}_{img_num}_skeleton.jpg")
        cv2.imwrite(skeleton_img_path, annotated_image)
        print(f"    ✓ Saved skeleton image: {skeleton_img_path}")

        # Save individual skeleton data
        skeleton_npy_path = os.path.join(action_dir, f"{action_name}_{img_num}_skeleton.npy")
        np.save(skeleton_npy_path, skeleton_data)
        print(f"    ✓ Saved skeleton data: {skeleton_npy_path}")

        # Save original image
        original_img_path = os.path.join(action_dir, f"{action_name}_{img_num}_raw.jpg")
        original_img = cv2.imread(img_path)
        cv2.imwrite(original_img_path, original_img)
        print(f"    ✓ Saved original image: {original_img_path}")

    if len(all_skeletons) == 0:
        print(f"  ✗ No valid skeletons found for action: {action_name}")
        return False

    # Create skeleton sequence (stack all frames)
    # Shape: (num_frames, 33, 3)
    skeleton_sequence = np.stack(all_skeletons, axis=0)

    # Save skeleton sequence
    sequence_path = os.path.join(action_dir, f"{action_name}_skeleton_sequence.npy")
    np.save(sequence_path, skeleton_sequence)

    print(f"\n  ✓ Created skeleton sequence: {sequence_path}")
    print(f"    Shape: {skeleton_sequence.shape}")
    print(f"    Frames: {skeleton_sequence.shape[0]}")
    print(f"    Keypoints: {skeleton_sequence.shape[1]}")
    print(f"    Coords: {skeleton_sequence.shape[2]}")

    return True


def main():
    """Main processing function."""
    print("\n" + "="*60)
    print("MediaPipe Test Picture Processor")
    print("="*60)

    # Check directories
    if not os.path.exists(TEST_PICTURE_DIR):
        print(f"Error: Test picture directory not found: {TEST_PICTURE_DIR}")
        return

    os.makedirs(ACTIONSET_DIR, exist_ok=True)

    # Initialize MediaPipe Pose
    print("\nInitializing MediaPipe Pose...")
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        print("✓ MediaPipe Pose initialized")

        # Process each action
        results = {}
        for action_name, image_numbers in ACTION_LABELS.items():
            success = process_action(action_name, image_numbers, pose)
            results[action_name] = success

    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for action_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {action_name}: {status}")

    print("\n" + "="*60)
    print("All actions processed!")
    print(f"Output directory: {ACTIONSET_DIR}")
    print("="*60)

    # List created directories
    print("\nCreated actionset directories:")
    for action_name in ACTION_LABELS.keys():
        action_dir = os.path.join(ACTIONSET_DIR, action_name)
        if os.path.exists(action_dir):
            files = os.listdir(action_dir)
            print(f"\n  {action_name}/ ({len(files)} files)")
            for f in sorted(files):
                print(f"    - {f}")


if __name__ == "__main__":
    main()
