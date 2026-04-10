#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 actionset 中的動作資料
檢查每個動作的幀數、時長，以及骨架的變化量
"""

import os
import numpy as np

def analyze_action(npy_path, action_name):
    """分析單一動作的骨架資料"""
    data = np.load(npy_path)
    frames = data.shape[0]
    seconds = frames / 30.0  # 假設 30fps

    # 計算動作的變化量（運動量）
    if frames > 1:
        # 計算相鄰幀之間的位移
        diffs = np.diff(data, axis=0)
        motion = np.mean(np.abs(diffs))
        max_motion = np.max(np.abs(diffs))
    else:
        motion = 0
        max_motion = 0

    # 計算手部位置範圍（判斷是否有明顯手部動作）
    # MediaPipe: 左手腕=15, 右手腕=16, 左肩=11, 右肩=12
    left_wrist_y = data[:, 15, 1]
    right_wrist_y = data[:, 16, 1]
    left_shoulder_y = data[:, 11, 1]
    right_shoulder_y = data[:, 12, 1]

    # 手腕相對肩膀的位置變化
    left_range = np.max(left_wrist_y - left_shoulder_y) - np.min(left_wrist_y - left_shoulder_y)
    right_range = np.max(right_wrist_y - right_shoulder_y) - np.min(right_wrist_y - right_shoulder_y)

    print(f"{action_name}:")
    print(f"  - Frames: {frames} ({seconds:.2f}s)")
    print(f"  - Average motion: {motion:.4f}")
    print(f"  - Max motion: {max_motion:.4f}")
    print(f"  - Left hand Y range: {left_range:.3f}")
    print(f"  - Right hand Y range: {right_range:.3f}")
    print()

    return {
        'name': action_name,
        'frames': frames,
        'seconds': seconds,
        'motion': motion,
        'max_motion': max_motion,
        'left_range': left_range,
        'right_range': right_range
    }

def main():
    base_path = "/home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/src/yolo_ros/actionset"

    print("=" * 60)
    print("ActionSet 分析報告")
    print("=" * 60)
    print()

    results = []

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('skeleton_sequence.npy') and not 'coco17' in f:
                    npy_path = os.path.join(folder_path, f)
                    result = analyze_action(npy_path, folder)
                    results.append(result)

    print("=" * 60)
    print("摘要")
    print("=" * 60)

    # 依幀數排序
    results.sort(key=lambda x: x['frames'])

    print("\n依幀數排序（少→多）:")
    for r in results:
        status = "⚠️ 太短" if r['frames'] < 90 else "✓"
        print(f"  {status} {r['name']}: {r['frames']} frames ({r['seconds']:.2f}s)")

    print("\n建議:")
    print("- 30fps 時，5秒錄製應該有 150 幀")
    print("- 如果 support 動作幀數太少，均勻取樣到 32 幀時會過度壓縮")
    print("- 建議 support 動作至少錄製 3-5 秒（90-150 幀）")

if __name__ == "__main__":
    main()
