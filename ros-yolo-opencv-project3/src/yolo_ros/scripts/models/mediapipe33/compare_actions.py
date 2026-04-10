#!/usr/bin/env python3
"""
比較 support 動作的差異
診斷為什麼 stopv5 和 rightv5 相似度這麼高
"""

import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    actionset_dir = os.path.join(base_dir, '..', '..', '..', 'actionset')

    actions = ['forwardv5', 'stopv5', 'rightv5']

    print("=" * 70)
    print("Support 動作數據分析")
    print("=" * 70)

    for action_name in actions:
        action_path = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
        if os.path.exists(action_path):
            data = np.load(action_path)
            print(f"\n【{action_name}】")
            print(f"  總幀數: {data.shape[0]}")
            print(f"  形狀: {data.shape}")

            # 分析關鍵部位的運動
            # 左手腕=15, 右手腕=16, 左肩=11, 右肩=12
            left_wrist = data[:, 15, :2]
            right_wrist = data[:, 16, :2]
            left_shoulder = data[:, 11, :2]
            right_shoulder = data[:, 12, :2]

            # 計算手腕相對於肩膀的位置
            left_rel = left_wrist - left_shoulder
            right_rel = right_wrist - right_shoulder

            print(f"  左手腕相對位置 (mean): x={left_rel[:,0].mean():.4f}, y={left_rel[:,1].mean():.4f}")
            print(f"  右手腕相對位置 (mean): x={right_rel[:,0].mean():.4f}, y={right_rel[:,1].mean():.4f}")

            # 計算運動範圍
            left_motion = np.diff(left_wrist, axis=0)
            right_motion = np.diff(right_wrist, axis=0)
            left_motion_mag = np.sqrt(np.sum(left_motion**2, axis=1))
            right_motion_mag = np.sqrt(np.sum(right_motion**2, axis=1))

            print(f"  左手運動量: avg={left_motion_mag.mean():.4f}, max={left_motion_mag.max():.4f}")
            print(f"  右手運動量: avg={right_motion_mag.mean():.4f}, max={right_motion_mag.max():.4f}")

            # 判斷動作類型
            total_motion = left_motion_mag.mean() + right_motion_mag.mean()
            if total_motion < 0.005:
                print(f"  ⚠️ 警告: 這是一個【靜態動作】，運動量很小")

            # 分析關鍵幀
            print(f"  第一幀手腕位置: L=({left_wrist[0,0]:.3f}, {left_wrist[0,1]:.3f}), R=({right_wrist[0,0]:.3f}, {right_wrist[0,1]:.3f})")
            print(f"  最後幀手腕位置: L=({left_wrist[-1,0]:.3f}, {left_wrist[-1,1]:.3f}), R=({right_wrist[-1,0]:.3f}, {right_wrist[-1,1]:.3f})")

    print("\n" + "=" * 70)
    print("問題診斷")
    print("=" * 70)
    print("""
可能的問題：
1. stop 和 right 動作在姿勢上太相似
   - stop: 雙手交叉靜止
   - right: 右手指向右邊

   如果 right 動作錄製時，手沒有明顯伸展，會與 stop 混淆

2. 建議改善：
   a) 重新錄製 rightv5：確保右手完全伸直指向右邊
   b) 重新錄製 stopv5：確保雙手明顯交叉在胸前
   c) 或者使用更具區分度的動作

3. 動作設計建議：
   - forward: 雙手向前推 (動態)
   - stop: 雙手舉高呈 X 形 (靜態，但姿勢明顯)
   - right: 右手完全伸直指向右方，左手下垂 (明顯不對稱)
""")

if __name__ == '__main__':
    main()
