#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骨架序列平滑處理工具

對 actionset 中的骨架序列進行平滑濾波，減少抖動
"""

import os
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

def smooth_skeleton_sequence(skeleton_data, method='savgol', window_size=5):
    """
    對骨架序列進行平滑處理

    Args:
        skeleton_data: (T, 33, 3) 骨架序列
        method: 'savgol' (Savitzky-Golay) 或 'moving_avg' (移動平均)
        window_size: 平滑窗口大小（必須是奇數）

    Returns:
        平滑後的骨架序列 (T, 33, 3)
    """
    T, num_joints, dims = skeleton_data.shape
    smoothed = np.zeros_like(skeleton_data)

    # 確保窗口大小是奇數且不超過序列長度
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, T if T % 2 == 1 else T - 1)

    if window_size < 3:
        return skeleton_data  # 序列太短，無法平滑

    for j in range(num_joints):
        for d in range(dims):
            signal = skeleton_data[:, j, d]

            if method == 'savgol':
                # Savitzky-Golay 濾波器：保留信號形狀的同時去除噪音
                # polyorder 必須小於 window_size
                polyorder = min(3, window_size - 1)
                smoothed[:, j, d] = savgol_filter(signal, window_size, polyorder)
            else:
                # 移動平均
                smoothed[:, j, d] = uniform_filter1d(signal, size=window_size, mode='nearest')

    return smoothed


def analyze_motion(skeleton_data, name=""):
    """分析骨架序列的運動量"""
    if skeleton_data.shape[0] < 2:
        return 0, 0

    diffs = np.diff(skeleton_data, axis=0)
    mean_motion = np.mean(np.abs(diffs))
    max_motion = np.max(np.abs(diffs))

    print(f"{name}: mean_motion={mean_motion:.4f}, max_motion={max_motion:.4f}")
    return mean_motion, max_motion


def process_actionset(base_path, window_size=5, backup=True):
    """
    處理整個 actionset 目錄

    Args:
        base_path: actionset 目錄路徑
        window_size: 平滑窗口大小
        backup: 是否備份原始檔案
    """
    print("=" * 60)
    print(f"骨架序列平滑處理 (window_size={window_size})")
    print("=" * 60)

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if f.endswith('skeleton_sequence.npy') and 'coco17' not in f and 'backup' not in f:
                npy_path = os.path.join(folder_path, f)

                # 載入原始數據
                data = np.load(npy_path)
                print(f"\n處理: {folder}/{f}")
                print(f"  Shape: {data.shape}")

                # 分析原始運動量
                print("  原始: ", end="")
                orig_mean, orig_max = analyze_motion(data, "")

                # 平滑處理
                smoothed = smooth_skeleton_sequence(data, method='savgol', window_size=window_size)

                # 分析平滑後運動量
                print("  平滑後: ", end="")
                smooth_mean, smooth_max = analyze_motion(smoothed, "")

                # 計算減少百分比
                if orig_mean > 0:
                    reduction = (1 - smooth_mean / orig_mean) * 100
                    print(f"  運動量減少: {reduction:.1f}%")

                # 備份原始檔案
                if backup:
                    backup_path = npy_path.replace('.npy', '_backup.npy')
                    if not os.path.exists(backup_path):
                        np.save(backup_path, data)
                        print(f"  備份: {backup_path}")

                # 儲存平滑後的數據
                np.save(npy_path, smoothed)
                print(f"  已儲存平滑後數據")


def preview_comparison(npy_path, window_size=5):
    """預覽平滑前後的差異（不儲存）"""
    data = np.load(npy_path)
    smoothed = smooth_skeleton_sequence(data, method='savgol', window_size=window_size)

    print(f"檔案: {npy_path}")
    print(f"Shape: {data.shape}")
    print("\n原始數據運動量:")
    analyze_motion(data, "原始")
    print("\n平滑後運動量:")
    analyze_motion(smoothed, "平滑")

    # 顯示前 10 幀的某個關節點變化
    joint_idx = 15  # 左手腕
    print(f"\n左手腕 (joint {joint_idx}) Y 座標變化 (前10幀):")
    print(f"  原始:   {data[:10, joint_idx, 1]}")
    print(f"  平滑後: {smoothed[:10, joint_idx, 1]}")


if __name__ == "__main__":
    import sys

    base_path = "/root/catkin_ws/src/yolo_ros/actionset"

    if len(sys.argv) > 1:
        if sys.argv[1] == "preview":
            # 預覽模式：只顯示不儲存
            if len(sys.argv) > 2:
                preview_comparison(sys.argv[2])
            else:
                # 預覽第一個找到的檔案
                for folder in os.listdir(base_path):
                    folder_path = os.path.join(base_path, folder)
                    if os.path.isdir(folder_path):
                        for f in os.listdir(folder_path):
                            if f.endswith('skeleton_sequence.npy') and 'coco17' not in f:
                                preview_comparison(os.path.join(folder_path, f))
                                sys.exit(0)
        elif sys.argv[1] == "process":
            # 處理模式：平滑並儲存
            window = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            process_actionset(base_path, window_size=window)
    else:
        print("用法:")
        print("  python3 smooth_skeleton.py preview [npy_path]  # 預覽平滑效果")
        print("  python3 smooth_skeleton.py process [window]   # 處理所有檔案")
        print()
        print("範例:")
        print("  python3 smooth_skeleton.py preview")
        print("  python3 smooth_skeleton.py process 5")
