import numpy as np
import os

base_path = "/root/catkin_ws/src/yolo_ros/actionset"

# 載入各動作並比較
actions = {}

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

            actions[folder] = sampled

            # 分析左手腕位置
            joint = 15
            x_mean = sampled[:, joint, 0].mean()
            y_mean = sampled[:, joint, 1].mean()
            x_range = sampled[:, joint, 0].max() - sampled[:, joint, 0].min()
            y_range = sampled[:, joint, 1].max() - sampled[:, joint, 1].min()

            print(f"{folder}:")
            print(f"  原始幀數: {T}")
            print(f"  左手腕 - X mean: {x_mean:.3f}, Y mean: {y_mean:.3f}")
            print(f"  左手腕 - X range: {x_range:.3f}, Y range: {y_range:.3f}")
            print()

# 計算動作之間的簡單相似度（不用模型，直接比較骨架）
print("=== 動作間的骨架相似度（餘弦相似度）===")
action_names = list(actions.keys())
for i, name1 in enumerate(action_names):
    for name2 in action_names[i+1:]:
        flat1 = actions[name1].flatten()
        flat2 = actions[name2].flatten()
        cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
        print(f"{name1} vs {name2}: {cos_sim:.4f}")
