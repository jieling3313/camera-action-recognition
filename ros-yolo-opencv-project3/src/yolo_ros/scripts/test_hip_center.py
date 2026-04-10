import numpy as np

npy_path = "/root/catkin_ws/src/yolo_ros/actionset/rightv5/rightv5_skeleton_sequence.npy"
data = np.load(npy_path)

print("=== 原始數據 ===")
print(f"Shape: {data.shape}")

# 左手腕
joint = 15
x_range = data[:, joint, 0].max() - data[:, joint, 0].min()
y_range = data[:, joint, 1].max() - data[:, joint, 1].min()
print(f"左手腕 X range: {x_range:.3f}, Y range: {y_range:.3f}")

# Hip centering (與 recognition node 一致)
hip_center = (data[:, 23, :2] + data[:, 24, :2]) / 2
centered = data.copy()
centered[:, :, :2] = centered[:, :, :2] - hip_center[:, np.newaxis, :]

print("\n=== Hip-Centering 後 ===")
joint = 15
x_range = centered[:, joint, 0].max() - centered[:, joint, 0].min()
y_range = centered[:, joint, 1].max() - centered[:, joint, 1].min()
print(f"左手腕 X range: {x_range:.3f}, Y range: {y_range:.3f}")

# 均勻取樣到 32 幀
T = data.shape[0]
indices = np.linspace(0, T - 1, 32, dtype=int)
sampled = centered[indices]

print("\n=== 均勻取樣 32 幀後 ===")
print(f"Shape: {sampled.shape}")
joint = 15
x_range = sampled[:, joint, 0].max() - sampled[:, joint, 0].min()
y_range = sampled[:, joint, 1].max() - sampled[:, joint, 1].min()
print(f"左手腕 X range: {x_range:.3f}, Y range: {y_range:.3f}")

# 顯示取樣的幀索引
print(f"\n取樣幀索引: {indices}")
print(f"原始幀數: {T}, 取樣間隔: 約每 {T/32:.1f} 幀取一幀")
