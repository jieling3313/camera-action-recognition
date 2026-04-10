import numpy as np

npy_path = "/root/catkin_ws/src/yolo_ros/actionset/rightv5/rightv5_skeleton_sequence.npy"
data = np.load(npy_path)

print(f"Shape: {data.shape}")
print()

joint = 15
x_min = data[:, joint, 0].min()
x_max = data[:, joint, 0].max()
y_min = data[:, joint, 1].min()
y_max = data[:, joint, 1].max()
print(f"左手腕 (joint {joint}) 座標範圍:")
print(f"  X: min={x_min:.3f}, max={x_max:.3f}, range={x_max - x_min:.3f}")
print(f"  Y: min={y_min:.3f}, max={y_max:.3f}, range={y_max - y_min:.3f}")
print()

joint = 16
x_min = data[:, joint, 0].min()
x_max = data[:, joint, 0].max()
y_min = data[:, joint, 1].min()
y_max = data[:, joint, 1].max()
print(f"右手腕 (joint {joint}) 座標範圍:")
print(f"  X: min={x_min:.3f}, max={x_max:.3f}, range={x_max - x_min:.3f}")
print(f"  Y: min={y_min:.3f}, max={y_max:.3f}, range={y_max - y_min:.3f}")
