#!/usr/bin/env python3
"""
快速測試 testv1 辨識問題
在容器內執行: python3 quick_test.py

重要：現在加入了髖部中心化預處理（與訓練一致）
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def hip_center_normalize(skeleton):
    """髖部中心化正規化（與訓練時一致）"""
    skeleton = skeleton.copy()
    # MediaPipe: 23=左髖, 24=右髖
    hip_center = (skeleton[:, 23, :2] + skeleton[:, 24, :2]) / 2
    skeleton[:, :, :2] = skeleton[:, :, :2] - hip_center[:, np.newaxis, :]
    return skeleton

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    actionset_dir = os.path.join(base_dir, '..', '..', '..', 'actionset')
    checkpoint_path = os.path.join(base_dir, 'checkpoints_mediapipe33', 'best.pth')

    # 載入模型
    print("載入模型...")
    model = OneShotActionRecognitionMediaPipe(in_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # 載入測試序列
    test_path = os.path.join(actionset_dir, 'testv1', 'testv1_skeleton_sequence.npy')
    test_data = np.load(test_path)
    print(f"testv1: shape={test_data.shape}, 總幀數={test_data.shape[0]}")

    # 載入支持動作
    print("\n載入支持動作...")
    support_features = {}
    for action_name in ['forwardv5', 'stopv5', 'rightv5']:
        action_path = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
        if os.path.exists(action_path):
            data = np.load(action_path)
            print(f"  {action_name}: shape={data.shape}")

            # 調整為 32 幀（與即時辨識一致）
            if data.shape[0] >= 32:
                sample = data[:32]
            else:
                pad_len = 32 - data.shape[0]
                sample = np.concatenate([data, np.repeat(data[-1:], pad_len, axis=0)], axis=0)

            # 關鍵：髖部中心化（與訓練一致）
            sample = hip_center_normalize(sample)

            tensor = torch.FloatTensor(sample).unsqueeze(0)
            with torch.no_grad():
                feature = model.embedding(tensor).numpy()[0]
            support_features[action_name] = feature
            print(f"    feature norm: {np.linalg.norm(feature):.4f}")

    # 檢查支持動作之間的相似度
    print("\n支持動作之間的相似度:")
    names = list(support_features.keys())
    for i, a1 in enumerate(names):
        for a2 in names[i+1:]:
            f1, f2 = support_features[a1], support_features[a2]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            status = "❌" if sim > 0.9 else "⚠️" if sim > 0.8 else "✓"
            print(f"  {a1} vs {a2}: {sim:.4f} {status}")

    # 滑動窗口測試 - 使用較小窗口以更精確捕捉動作
    print("\n" + "=" * 60)
    print("滑動窗口辨識 (窗口=32, 步長=16)")
    print("預期: forward -> Unknown -> stop -> Unknown -> right")
    print("=" * 60)

    window_size = 32  # 與即時辨識一致
    step_size = 16    # 較小步長避免錯過邊界

    for start in range(0, len(test_data) - window_size + 1, step_size):
        end = start + window_size
        window = test_data[start:end]

        # 關鍵：髖部中心化（與訓練一致）
        window = hip_center_normalize(window)

        # 提取特徵
        tensor = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            query_feature = model.embedding(tensor).numpy()[0]

        query_norm = np.linalg.norm(query_feature)

        # 計算相似度
        sims = {}
        for name, sup_feat in support_features.items():
            sim = np.dot(query_feature, sup_feat) / (query_norm * np.linalg.norm(sup_feat) + 1e-8)
            sims[name] = sim

        # 計算最高和第二高相似度
        sorted_sims = sorted(sims.values(), reverse=True)
        max_sim = sorted_sims[0]
        second_sim = sorted_sims[1] if len(sorted_sims) > 1 else 0
        sim_margin = max_sim - second_sim
        best_action = max(sims.items(), key=lambda x: x[1])[0]

        # 雙重閾值判斷（與 recognition_display_node_v2.py 一致）
        threshold = 0.85
        margin_threshold = 0.05

        if max_sim < threshold:
            result = "Unknown"
            reason = f"sim<{threshold}"
        elif sim_margin < margin_threshold:
            result = "Unknown"
            reason = f"margin<{margin_threshold}"
        else:
            result = best_action
            reason = "OK"

        print(f"幀 {start:3d}-{end:3d}: {result:12s} (max_sim={max_sim:.4f}, margin={sim_margin:.4f}, {reason})")
        for name, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
            marker = "←" if name == result and result != "Unknown" else ""
            print(f"    {name}: {sim:.4f} {marker}")

    print("\n" + "=" * 60)
    print("判斷邏輯:")
    print(f"  1. max_sim >= {threshold} (相似度閾值)")
    print(f"  2. margin >= {margin_threshold} (差距閾值)")
    print("  兩個條件都滿足才會辨識為具體動作，否則為 Unknown")
    print("=" * 60)

if __name__ == '__main__':
    main()
