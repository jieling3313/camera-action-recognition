#!/usr/bin/env python3
"""
診斷 circle 動作辨識問題
在容器內執行: python3.10 diagnose_circle.py
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def hip_center_normalize(skeleton):
    """髖部中心化正規化"""
    skeleton = skeleton.copy()
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

    # 列出所有可用動作
    print("\n可用動作資料夾:")
    action_dirs = [d for d in os.listdir(actionset_dir)
                   if os.path.isdir(os.path.join(actionset_dir, d))]
    for d in sorted(action_dirs):
        npy_path = os.path.join(actionset_dir, d, f'{d}_skeleton_sequence.npy')
        if os.path.exists(npy_path):
            data = np.load(npy_path)
            print(f"  {d}: {data.shape[0]} frames")

    # 載入 stop 和 circle 的特徵
    print("\n" + "=" * 60)
    print("比較 circle 與 stop 的特徵")
    print("=" * 60)

    actions_to_compare = ['circlev6', 'stopv6', 'forwardv6', 'rightv6']
    support_features = {}

    for action_name in actions_to_compare:
        action_path = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
        if os.path.exists(action_path):
            data = np.load(action_path)
            print(f"\n{action_name}:")
            print(f"  原始幀數: {data.shape[0]}")
            print(f"  x範圍: [{data[...,0].min():.3f}, {data[...,0].max():.3f}]")
            print(f"  y範圍: [{data[...,1].min():.3f}, {data[...,1].max():.3f}]")

            # 均勻取樣 32 幀（與 model_manager.py 一致）
            T = data.shape[0]
            if T >= 32:
                indices = np.linspace(0, T - 1, 32, dtype=int)
                sample = data[indices]
                print(f"  均勻取樣索引: {indices[:5]}...{indices[-5:]}")
            else:
                pad_len = 32 - T
                sample = np.concatenate([data, np.repeat(data[-1:], pad_len, axis=0)], axis=0)

            # 髖部中心化
            sample = hip_center_normalize(sample)
            print(f"  髖部中心化後 x範圍: [{sample[...,0].min():.3f}, {sample[...,0].max():.3f}]")

            # 提取特徵
            tensor = torch.FloatTensor(sample).unsqueeze(0)
            with torch.no_grad():
                feature = model.embedding(tensor).numpy()[0]
            support_features[action_name] = feature
            print(f"  feature norm: {np.linalg.norm(feature):.4f}")

    # 計算所有動作之間的相似度
    print("\n" + "=" * 60)
    print("動作間相似度矩陣")
    print("=" * 60)

    names = list(support_features.keys())
    print(f"\n{'':15}", end='')
    for n in names:
        print(f"{n[:10]:12}", end='')
    print()

    for i, a1 in enumerate(names):
        print(f"{a1[:15]:15}", end='')
        for a2 in names:
            f1, f2 = support_features[a1], support_features[a2]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            print(f"{sim:.3f}       ", end='')
        print()

    # 分析 circle 與 stop 的差異
    if 'circlev6' in support_features and 'stopv6' in support_features:
        circle_f = support_features['circlev6']
        stop_f = support_features['stopv6']

        sim = np.dot(circle_f, stop_f) / (np.linalg.norm(circle_f) * np.linalg.norm(stop_f) + 1e-8)

        print("\n" + "=" * 60)
        print("問題分析")
        print("=" * 60)
        print(f"\ncircle vs stop 相似度: {sim:.4f}")

        if sim > 0.85:
            print("⚠️ 相似度太高！模型無法區分 circle 和 stop")
            print("\n可能原因:")
            print("  1. circle 錄製時包含太多靜止幀（開始或結束時站著不動）")
            print("  2. circle 動作的手臂運動幅度不夠大")
            print("  3. circle 動作太短，沒有完整的圓形軌跡")
            print("\n建議:")
            print("  - 重新錄製 circle，確保整個過程都在畫圓")
            print("  - 可以錄製更長時間（例如畫 2-3 圈）")
            print("  - 確保手臂有明顯的運動")
        else:
            print("✓ 相似度在可接受範圍內")

    # 滑動窗口測試 circle 動作
    print("\n" + "=" * 60)
    print("滑動窗口測試 circle 序列")
    print("=" * 60)

    circle_path = os.path.join(actionset_dir, 'circlev6', 'circlev6_skeleton_sequence.npy')
    if os.path.exists(circle_path):
        circle_data = np.load(circle_path)
        print(f"circle 總幀數: {circle_data.shape[0]}")

        for start in range(0, min(circle_data.shape[0] - 32, 100), 16):
            end = start + 32
            window = circle_data[start:end]
            window = hip_center_normalize(window)

            tensor = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                query_feature = model.embedding(tensor).numpy()[0]

            # 計算與各動作的相似度
            sims = {}
            for name, sup_feat in support_features.items():
                sim = np.dot(query_feature, sup_feat) / (np.linalg.norm(query_feature) * np.linalg.norm(sup_feat) + 1e-8)
                sims[name] = sim

            best = max(sims.items(), key=lambda x: x[1])
            sorted_sims = sorted(sims.values(), reverse=True)
            margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0

            status = "✓" if best[0] == 'circlev6' else "✗"
            print(f"幀 {start:3d}-{end:3d}: {best[0]:12s} (sim={best[1]:.4f}, margin={margin:.4f}) {status}")

if __name__ == '__main__':
    main()
