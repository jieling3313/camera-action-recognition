import numpy as np
import os

base_path = "/root/catkin_ws/src/yolo_ros/actionset"

# MediaPipe 33 關節點索引
# 手臂相關: 11,12(肩), 13,14(肘), 15,16(手腕), 17-22(手指)
ARM_JOINTS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# 載入各動作
actions_full = {}
actions_arms = {}

for folder in ['circlev8', 'forwardv6', 'rightv6', 'stopv6']:
    folder_path = os.path.join(base_path, folder)
    for f in os.listdir(folder_path):
        if f.endswith('skeleton_sequence.npy') and 'coco17' not in f:
            npy_path = os.path.join(folder_path, f)
            data = np.load(npy_path)

            # Hip centering
            hip_center = (data[:, 23, :2] + data[:, 24, :2]) / 2
            data[:, :, :2] = data[:, :, :2] - hip_center[:, np.newaxis, :]

            # 均勻取樣到 32 幀
            T = data.shape[0]
            indices = np.linspace(0, T - 1, 32, dtype=int)
            sampled = data[indices]

            actions_full[folder] = sampled
            # 只取手臂關節
            actions_arms[folder] = sampled[:, ARM_JOINTS, :]

print("=== 全身骨架相似度 ===")
action_names = list(actions_full.keys())
for i, name1 in enumerate(action_names):
    for name2 in action_names[i+1:]:
        flat1 = actions_full[name1].flatten()
        flat2 = actions_full[name2].flatten()
        cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
        print(f"{name1} vs {name2}: {cos_sim:.4f}")

print("\n=== 只用手臂關節的相似度 ===")
for i, name1 in enumerate(action_names):
    for name2 in action_names[i+1:]:
        flat1 = actions_arms[name1].flatten()
        flat2 = actions_arms[name2].flatten()
        cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
        print(f"{name1} vs {name2}: {cos_sim:.4f}")

print("\n=== 各動作手臂特徵 ===")
for name, arm_data in actions_arms.items():
    # 左手腕 (joint 15 在 ARM_JOINTS 中的索引是 4)
    left_wrist_idx = ARM_JOINTS.index(15)
    right_wrist_idx = ARM_JOINTS.index(16)

    left_x = arm_data[:, left_wrist_idx, 0]
    left_y = arm_data[:, left_wrist_idx, 1]
    right_x = arm_data[:, right_wrist_idx, 0]
    right_y = arm_data[:, right_wrist_idx, 1]

    print(f"{name}:")
    print(f"  左手腕: X=[{left_x.min():.2f}, {left_x.max():.2f}], Y=[{left_y.min():.2f}, {left_y.max():.2f}]")
    print(f"  右手腕: X=[{right_x.min():.2f}, {right_x.max():.2f}], Y=[{right_y.min():.2f}, {right_y.max():.2f}]")
