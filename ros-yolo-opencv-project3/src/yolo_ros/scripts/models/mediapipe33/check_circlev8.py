#!/usr/bin/env python3
"""
檢查 circlev8 的骨架數據，確認姿勢是否正確
"""

import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"

# 載入 circlev8
circle_path = os.path.join(actionset_dir, 'circlev8', 'circlev8_skeleton_sequence.npy')
circle_data = np.load(circle_path)

print("=" * 60)
print("circlev8 骨架分析")
print("=" * 60)
print(f"幀數: {circle_data.shape[0]}")
print(f"形狀: {circle_data.shape}")

# MediaPipe 關鍵點索引
# 11=左肩, 12=右肩, 13=左肘, 14=右肘, 15=左手腕, 16=右手腕
# 0=鼻子

# 取中間幀分析（最穩定）
mid_frame = circle_data[circle_data.shape[0] // 2]

print("\n中間幀關鍵點位置:")
print(f"  鼻子 (0):     y = {mid_frame[0, 1]:.3f}")
print(f"  左肩 (11):    y = {mid_frame[11, 1]:.3f}")
print(f"  右肩 (12):    y = {mid_frame[12, 1]:.3f}")
print(f"  左手腕 (15):  y = {mid_frame[15, 1]:.3f}")
print(f"  右手腕 (16):  y = {mid_frame[16, 1]:.3f}")

# 判斷手是否舉起（手腕 y 值小於肩膀 y 值表示舉起）
# 注意：MediaPipe 的 y 軸是從上到下，所以 y 值小 = 位置高
left_raised = mid_frame[15, 1] < mid_frame[11, 1]
right_raised = mid_frame[16, 1] < mid_frame[12, 1]

print(f"\n手臂狀態:")
print(f"  左手舉起: {left_raised} (手腕 y={mid_frame[15, 1]:.3f} vs 肩膀 y={mid_frame[11, 1]:.3f})")
print(f"  右手舉起: {right_raised} (手腕 y={mid_frame[16, 1]:.3f} vs 肩膀 y={mid_frame[12, 1]:.3f})")

# 比較所有幀的手腕位置
left_wrist_y_all = circle_data[:, 15, 1]
right_wrist_y_all = circle_data[:, 16, 1]
left_shoulder_y_all = circle_data[:, 11, 1]
right_shoulder_y_all = circle_data[:, 12, 1]

print(f"\n所有幀統計:")
print(f"  左手腕 y 範圍: [{left_wrist_y_all.min():.3f}, {left_wrist_y_all.max():.3f}], 平均: {left_wrist_y_all.mean():.3f}")
print(f"  右手腕 y 範圍: [{right_wrist_y_all.min():.3f}, {right_wrist_y_all.max():.3f}], 平均: {right_wrist_y_all.mean():.3f}")
print(f"  左肩 y 平均: {left_shoulder_y_all.mean():.3f}")
print(f"  右肩 y 平均: {right_shoulder_y_all.mean():.3f}")

# 計算手腕相對於肩膀的位置（負值 = 手舉起）
left_relative = left_wrist_y_all.mean() - left_shoulder_y_all.mean()
right_relative = right_wrist_y_all.mean() - right_shoulder_y_all.mean()

print(f"\n手腕相對於肩膀的 y 位移（負值=舉起）:")
print(f"  左手: {left_relative:.3f}")
print(f"  右手: {right_relative:.3f}")

if left_relative < -0.1 and right_relative < -0.1:
    print("\n✓ 雙手明顯舉起")
elif left_relative < 0 and right_relative < 0:
    print("\n⚠️ 雙手略微舉起，但不夠明顯")
else:
    print("\n✗ 雙手沒有舉起！這可能是問題所在")

# 與 stop 比較
print("\n" + "=" * 60)
print("與 stop 動作比較")
print("=" * 60)

stop_path = os.path.join(actionset_dir, 'stopv6', 'stopv6_skeleton_sequence.npy')
if os.path.exists(stop_path):
    stop_data = np.load(stop_path)
    stop_mid = stop_data[stop_data.shape[0] // 2]

    print(f"\nstop 中間幀:")
    print(f"  左手腕 y: {stop_mid[15, 1]:.3f}")
    print(f"  右手腕 y: {stop_mid[16, 1]:.3f}")
    print(f"  左肩 y: {stop_mid[11, 1]:.3f}")
    print(f"  右肩 y: {stop_mid[12, 1]:.3f}")

    stop_left_relative = stop_mid[15, 1] - stop_mid[11, 1]
    stop_right_relative = stop_mid[16, 1] - stop_mid[12, 1]
    print(f"  左手相對肩膀: {stop_left_relative:.3f}")
    print(f"  右手相對肩膀: {stop_right_relative:.3f}")

    print(f"\n差異:")
    print(f"  circle 左手 vs stop 左手: {left_relative:.3f} vs {stop_left_relative:.3f}")
    print(f"  circle 右手 vs stop 右手: {right_relative:.3f} vs {stop_right_relative:.3f}")
