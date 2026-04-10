#!/usr/bin/env python3
"""
詳細分析 testv1，輸出文字結果和 CSV 供後續分析
執行: python3 detailed_analysis.py
"""

import numpy as np
import torch
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def calculate_motion_per_frame(skeleton_sequence):
    """計算每幀的運動量"""
    upper_body_idx = list(range(11, 23))
    motions = [0]
    for i in range(1, len(skeleton_sequence)):
        diff = skeleton_sequence[i, upper_body_idx, :2] - skeleton_sequence[i-1, upper_body_idx, :2]
        motion = np.sqrt(np.sum(diff ** 2))
        motions.append(motion)
    return np.array(motions)

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
    total_frames = test_data.shape[0]
    print(f"testv1: shape={test_data.shape}, 總幀數={total_frames}")

    # 計算每幀運動量
    motions = calculate_motion_per_frame(test_data)
    print(f"平均運動量: {motions.mean():.4f}, 最大: {motions.max():.4f}")

    # 載入支持動作
    print("\n載入支持動作...")
    support_features = {}
    support_data = {}
    for action_name in ['forwardv5', 'stopv5', 'rightv5']:
        action_path = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
        if os.path.exists(action_path):
            data = np.load(action_path)
            support_data[action_name] = data
            print(f"  {action_name}: shape={data.shape}")

            # 調整為 32 幀（與 model_manager.py 一致）
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

            # 分析支持動作的運動量
            sup_motions = calculate_motion_per_frame(data)
            print(f"    運動量: avg={sup_motions.mean():.4f}, max={sup_motions.max():.4f}")

    # 檢查支持動作之間的相似度
    print("\n" + "=" * 60)
    print("支持動作之間的相似度 (問題診斷)")
    print("=" * 60)
    names = list(support_features.keys())
    inter_similarities = []
    for i, a1 in enumerate(names):
        for a2 in names[i+1:]:
            f1, f2 = support_features[a1], support_features[a2]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            inter_similarities.append(sim)
            status = "❌ 太高無法區分" if sim > 0.95 else "⚠️ 偏高" if sim > 0.9 else "✓ 可區分"
            print(f"  {a1} vs {a2}: {sim:.4f} {status}")

    avg_inter_sim = np.mean(inter_similarities)
    print(f"\n平均相似度: {avg_inter_sim:.4f}")

    # 計算建議的閾值
    # 如果要區分「有做動作」vs「沒做動作」，閾值應該在 inter_similarity 附近
    suggested_threshold = avg_inter_sim - 0.02
    print(f"建議閾值: {suggested_threshold:.4f} (略低於平均相似度)")

    # 逐幀分析（更細粒度）
    print("\n" + "=" * 60)
    print("滑動窗口逐幀分析 (窗口=32, 步長=16)")
    print("=" * 60)

    window_size = 32
    step_size = 16
    results = []

    for start in range(0, total_frames - window_size + 1, step_size):
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

        max_sim = max(sims.values())
        best_action = max(sims.items(), key=lambda x: x[1])[0]

        # 計算窗口內的平均運動量
        window_motion = motions[start:end].mean()

        # 動態閾值：根據 inter_similarity 調整
        # 如果 max_sim 顯著高於其他相似度，更可能是正確辨識
        sim_values = list(sims.values())
        sim_diff = max_sim - sorted(sim_values)[-2]  # 與第二高的差距

        results.append({
            'frame_start': start,
            'frame_end': end,
            'best_action': best_action,
            'max_similarity': max_sim,
            'sim_diff': sim_diff,
            'motion': window_motion,
            'query_norm': query_norm,
            'sims': sims.copy()
        })

    # 輸出結果
    print(f"\n{'Frame':^10} {'Motion':^8} {'Best':^12} {'MaxSim':^8} {'Diff':^8} | forward  stop     right")
    print("-" * 80)

    for r in results:
        sims = r['sims']
        print(f"{r['frame_start']:4d}-{r['frame_end']:4d} "
              f"{r['motion']:7.4f} "
              f"{r['best_action']:12s} "
              f"{r['max_similarity']:7.4f} "
              f"{r['sim_diff']:7.4f} | "
              f"{sims['forwardv5']:7.4f}  {sims['stopv5']:7.4f}  {sims['rightv5']:7.4f}")

    # 儲存 CSV
    csv_path = os.path.join(base_dir, 'testv1_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_start', 'frame_end', 'motion', 'best_action', 'max_sim', 'sim_diff',
                        'forward_sim', 'stop_sim', 'right_sim', 'query_norm'])
        for r in results:
            writer.writerow([
                r['frame_start'], r['frame_end'], f"{r['motion']:.4f}",
                r['best_action'], f"{r['max_similarity']:.4f}", f"{r['sim_diff']:.4f}",
                f"{r['sims']['forwardv5']:.4f}", f"{r['sims']['stopv5']:.4f}",
                f"{r['sims']['rightv5']:.4f}", f"{r['query_norm']:.2f}"
            ])
    print(f"\n結果已儲存: {csv_path}")

    # 診斷總結
    print("\n" + "=" * 60)
    print("診斷總結")
    print("=" * 60)

    max_sims = [r['max_similarity'] for r in results]
    sim_diffs = [r['sim_diff'] for r in results]

    print(f"Max similarity 範圍: [{min(max_sims):.4f}, {max(max_sims):.4f}]")
    print(f"Sim diff 範圍: [{min(sim_diffs):.4f}, {max(sim_diffs):.4f}]")

    if avg_inter_sim > 0.95:
        print("\n❌ 核心問題: 支持動作之間相似度太高 (>0.95)")
        print("   解決方案:")
        print("   1. 重新訓練模型，使用對比學習增加區分度")
        print("   2. 錄製差異更大的動作")
        print("   3. 使用 sim_diff（與第二高相似度的差距）作為辨識依據")
    elif min(max_sims) > 0.9:
        print("\n⚠️ 問題: 即使不做動作，相似度仍然很高")
        print("   解決方案:")
        print("   1. 提高閾值到", f"{min(max_sims) + 0.02:.2f}")
        print("   2. 使用相對差距而非絕對閾值")
    else:
        print("\n✓ 數據看起來可以區分")
        print(f"   建議閾值: {suggested_threshold:.2f}")


if __name__ == '__main__':
    main()
