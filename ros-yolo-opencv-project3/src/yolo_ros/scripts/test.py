#!/usr/bin/env python3.10
import numpy as np

npy_path = "/root/catkin_ws/src/yolo_ros/actionset/rightv5/rightv5_skeleton_sequence.npy"
data = np.load(npy_path)

print(f"Shape: {data.shape}")
print()

joint = 15
print(f"左手腕 (joint {joint}) 座標範圍:")
print(f"  X: min={data[:, joint, 0].min():.3f}, max={data[:, joint, 0].max():.3f}, range={data[:, joint, 0].max() - 
data[:, joint, 0].min():.3f}")
print(f"  Y: min={data[:, joint, 1].min():.3f}, max={data[:, joint, 1].max():.3f}, range={data[:, joint, 1].max() - 
data[:, joint, 1].min():.3f}")
print()

joint = 16
print(f"右手腕 (joint {joint}) 座標範圍:")
print(f"  X: min={data[:, joint, 0].min():.3f}, max={data[:, joint, 0].max():.3f}, range={data[:, joint, 0].max() - 
data[:, joint, 0].min():.3f}")
print(f"  Y: min={data[:, joint, 1].min():.3f}, max={data[:, joint, 1].max():.3f}, range={data[:, joint, 1].max() - 
data[:, joint, 1].min():.3f}")
