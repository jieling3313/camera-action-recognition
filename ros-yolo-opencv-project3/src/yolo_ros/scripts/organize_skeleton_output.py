#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
組織 skeleton_output 圖片到 actionset 目錄
將骨架圖片按照動作類別分類

規則：
- 1, 2, 3, 4_skeleton → stop 動作
- 5, 6, 7_skeleton → left 動作
- 8, 9_skeleton → right 動作
"""

import os
import shutil
import numpy as np

def organize_skeleton_output():
    """組織骨架輸出檔案到 actionset"""

    # 路徑設定
    skeleton_output_dir = "/root/catkin_ws/src/yolo_ros/scripts/skeleton_output"
    actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"

    # 檢查來源目錄
    if not os.path.exists(skeleton_output_dir):
        print(f"❌ Error: skeleton_output directory not found: {skeleton_output_dir}")
        return False

    # 建立 actionset 目錄
    os.makedirs(actionset_dir, exist_ok=True)

    # 定義動作分類規則
    action_mapping = {
        'stop': [1, 2, 3, 4],
        'left': [5, 6, 7],
        'right': [8, 9]
    }

    print("=" * 60)
    print("Organizing skeleton_output files to actionset")
    print("=" * 60)

    # 處理每個動作類別
    for action_name, skeleton_ids in action_mapping.items():
        print(f"\n📁 Processing action: {action_name}")

        # 建立動作目錄
        action_dir = os.path.join(actionset_dir, action_name)
        os.makedirs(action_dir, exist_ok=True)

        # 複製對應的骨架檔案
        for skeleton_id in skeleton_ids:
            jpg_file = f"{skeleton_id}_skeleton.jpg"
            npy_file = f"{skeleton_id}_skeleton.npy"

            jpg_src = os.path.join(skeleton_output_dir, jpg_file)
            npy_src = os.path.join(skeleton_output_dir, npy_file)

            # 新檔名（加上動作名稱）
            jpg_dst = os.path.join(action_dir, f"{action_name}_{skeleton_id}_skeleton.jpg")
            npy_dst = os.path.join(action_dir, f"{action_name}_{skeleton_id}_skeleton.npy")

            # 複製 JPG
            if os.path.exists(jpg_src):
                shutil.copy2(jpg_src, jpg_dst)
                print(f"  ✅ Copied: {jpg_file} → {action_name}/{os.path.basename(jpg_dst)}")
            else:
                print(f"  ⚠️  Not found: {jpg_file}")

            # 複製 NPY
            if os.path.exists(npy_src):
                shutil.copy2(npy_src, npy_dst)
                print(f"  ✅ Copied: {npy_file} → {action_name}/{os.path.basename(npy_dst)}")
            else:
                print(f"  ⚠️  Not found: {npy_file}")

        # 檢查動作目錄內容
        action_files = os.listdir(action_dir)
        npy_files = [f for f in action_files if f.endswith('.npy')]
        print(f"  📊 Total samples in '{action_name}': {len(npy_files)}")

    print("\n" + "=" * 60)
    print("✅ Organization completed!")
    print("=" * 60)

    # 顯示最終結構
    print("\n📂 Final actionset structure:")
    for action_name in action_mapping.keys():
        action_dir = os.path.join(actionset_dir, action_name)
        if os.path.exists(action_dir):
            files = os.listdir(action_dir)
            jpg_count = len([f for f in files if f.endswith('.jpg')])
            npy_count = len([f for f in files if f.endswith('.npy')])
            print(f"  - {action_name}/")
            print(f"      Images: {jpg_count}")
            print(f"      Skeletons: {npy_count}")

    return True


def verify_skeleton_data():
    """驗證骨架數據格式"""
    print("\n" + "=" * 60)
    print("Verifying skeleton data format")
    print("=" * 60)

    actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"
    actions = ['stop', 'left', 'right']

    for action_name in actions:
        action_dir = os.path.join(actionset_dir, action_name)
        if not os.path.exists(action_dir):
            continue

        print(f"\n📊 Action: {action_name}")
        npy_files = [f for f in os.listdir(action_dir) if f.endswith('.npy')]

        for npy_file in npy_files[:2]:  # 只檢查前 2 個
            npy_path = os.path.join(action_dir, npy_file)
            try:
                data = np.load(npy_path)
                print(f"  - {npy_file}: shape {data.shape}, dtype {data.dtype}")
            except Exception as e:
                print(f"  ❌ Error loading {npy_file}: {e}")


if __name__ == '__main__':
    # 執行組織
    success = organize_skeleton_output()

    if success:
        # 驗證數據
        verify_skeleton_data()

        print("\n" + "=" * 60)
        print("✨ All done! You can now use these actions in Page 2.")
        print("=" * 60)
    else:
        print("\n❌ Failed to organize skeleton_output")
