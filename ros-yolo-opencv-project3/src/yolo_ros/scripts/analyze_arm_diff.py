import numpy as np
import os

base_path = "/root/catkin_ws/src/yolo_ros/actionset"
ARM_JOINTS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# 找到最新版本的每個動作
action_folders = {
    'circle': 'circlev10',
    'forward': 'forwardv9',
    'left': 'leftv9',
    'right': 'rightv9',
    'stop': 'stopv9'
}

actions = {}

for name, folder in action_folders.items():
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

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

            actions[name] = sampled
            break

print("=" * 60)
print("各動作的手腕位置特徵")
print("=" * 60)

for name, sampled in actions.items():
    # 左手腕 (15), 右手腕 (16)
    left_wrist = sampled[:, 15, :]
    right_wrist = sampled[:, 16, :]

    print(f"\n{name}:")
    print(f"  左手腕 X: [{left_wrist[:, 0].min():.2f}, {left_wrist[:, 0].max():.2f}] mean={left_wrist[:, 0].mean():.2f}")
    print(f"  左手腕 Y: [{left_wrist[:, 1].min():.2f}, {left_wrist[:, 1].max():.2f}] mean={left_wrist[:, 1].mean():.2f}")
    print(f"  右手腕 X: [{right_wrist[:, 0].min():.2f}, {right_wrist[:, 0].max():.2f}] mean={right_wrist[:, 0].mean():.2f}")
    print(f"  右手腕 Y: [{right_wrist[:, 1].min():.2f}, {right_wrist[:, 1].max():.2f}] mean={right_wrist[:, 1].mean():.2f}")

print("\n" + "=" * 60)
print("建議的動作區分特徵")
print("=" * 60)
print("""
理想的動作應該有明顯不同的特徵：
- circle: 雙手高舉（Y 很負，如 -0.9 ~ -1.0）
- forward: 雙手向前伸（Y 在中間，如 -0.4 ~ -0.6）
- right: 左手向右伸（左手 X 很正，如 +0.3 ~ +0.5）
- left: 右手向左伸（右手 X 很負，如 -0.3 ~ -0.5）
- stop: 雙手舉起但低於 circle（Y 如 -0.5 ~ -0.7）
""")

print("\n" + "=" * 60)
print("動作相似度矩陣（只用手腕 X, Y）")
print("=" * 60)

# 只用手腕的 X, Y 座標
wrist_features = {}
for name, sampled in actions.items():
    left_wrist = sampled[:, 15, :2]  # 只取 X, Y
    right_wrist = sampled[:, 16, :2]
    wrist_features[name] = np.concatenate([left_wrist.flatten(), right_wrist.flatten()])

names = list(wrist_features.keys())
print("\n      ", "  ".join([f"{n:8s}" for n in names]))
for i, n1 in enumerate(names):
    row = f"{n1:6s}"
    for n2 in names:
        f1 = wrist_features[n1]
        f2 = wrist_features[n2]
        sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        row += f"  {sim:8.3f}"
    print(row)
