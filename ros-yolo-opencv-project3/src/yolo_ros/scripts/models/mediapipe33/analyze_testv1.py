#!/usr/bin/env python3
"""
分析 testv1 骨架序列，診斷動作辨識問題

測試序列內容: forward動作 -> 回正 -> stop動作 -> 回正 -> right動作
目標: 確認模型能否正確辨識出這個序列中的動作轉換
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def load_model(checkpoint_path):
    """載入模型"""
    model = OneShotActionRecognitionMediaPipe(in_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def extract_feature(model, skeleton_data):
    """提取特徵向量"""
    tensor = torch.FloatTensor(skeleton_data).unsqueeze(0)
    with torch.no_grad():
        feature = model.embedding(tensor)
        return feature.numpy()[0]

def cosine_similarity(f1, f2):
    """計算餘弦相似度"""
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)

def hip_center_normalize(skeleton):
    """髖部中心化正規化（與訓練時一致）"""
    skeleton = skeleton.copy()
    # MediaPipe: 23=左髖, 24=右髖
    hip_center = (skeleton[:, 23, :2] + skeleton[:, 24, :2]) / 2
    skeleton[:, :, :2] = skeleton[:, :, :2] - hip_center[:, np.newaxis, :]
    return skeleton

def analyze_motion(skeleton_sequence):
    """分析每幀的運動量"""
    # 計算幀間差異（上半身關鍵點）
    upper_body_idx = list(range(11, 23))  # 肩膀、手肘、手腕、髖部

    motions = []
    for i in range(1, len(skeleton_sequence)):
        diff = skeleton_sequence[i, upper_body_idx, :2] - skeleton_sequence[i-1, upper_body_idx, :2]
        motion = np.sqrt(np.sum(diff ** 2))
        motions.append(motion)

    return np.array(motions)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    actionset_dir = os.path.join(base_dir, '..', '..', '..', 'actionset')
    checkpoint_path = os.path.join(base_dir, 'checkpoints_mediapipe33', 'best.pth')

    # 載入測試序列
    test_path = os.path.join(actionset_dir, 'testv1', 'testv1_skeleton_sequence.npy')
    test_data = np.load(test_path)

    print("=" * 70)
    print("TestV1 骨架序列分析")
    print("=" * 70)
    print(f"序列形狀: {test_data.shape}")
    print(f"總幀數: {test_data.shape[0]}")
    print(f"預期動作序列: forward -> 無動作 -> stop -> 無動作 -> right")

    # 載入模型
    print("\n載入模型...")
    model = load_model(checkpoint_path)

    # 載入支持動作的特徵
    print("\n載入支持動作特徵...")
    support_features = {}
    for action_name in ['forwardv5', 'stopv5', 'rightv5']:
        action_path = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
        if os.path.exists(action_path):
            data = np.load(action_path)
            # 使用與 model_manager.py 相同的方式：取前 32 幀或補齊
            if data.shape[0] >= 32:
                sample = data[:32]
            else:
                pad_len = 32 - data.shape[0]
                sample = np.concatenate([data, np.repeat(data[-1:], pad_len, axis=0)], axis=0)

            # 關鍵：髖部中心化（與訓練一致）
            sample = hip_center_normalize(sample)

            feature = extract_feature(model, sample)
            support_features[action_name] = feature
            print(f"  {action_name}: shape={data.shape}, feature_norm={np.linalg.norm(feature):.4f}")

    # 分析運動量
    print("\n" + "=" * 70)
    print("運動量分析（用於找出動作邊界）")
    print("=" * 70)

    motions = analyze_motion(test_data)

    # 找出運動量的峰值和谷值
    motion_threshold = np.mean(motions) * 0.5

    # 簡單的狀態機找出動作段落
    is_moving = motions > motion_threshold

    # 找出狀態變化點
    state_changes = []
    current_state = is_moving[0]
    for i, moving in enumerate(is_moving):
        if moving != current_state:
            state_changes.append((i, 'start' if moving else 'end'))
            current_state = moving

    print(f"運動閾值: {motion_threshold:.4f}")
    print(f"狀態變化點: {state_changes}")

    # 滑動窗口分析（模擬即時辨識）
    print("\n" + "=" * 70)
    print("滑動窗口辨識分析（窗口=32幀，步長=16幀）")
    print("=" * 70)

    window_size = 32
    step_size = 16

    results = []

    for start in range(0, len(test_data) - window_size + 1, step_size):
        end = start + window_size
        window = test_data[start:end]

        # 關鍵：髖部中心化（與訓練一致）
        window = hip_center_normalize(window)

        # 提取特徵
        query_feature = extract_feature(model, window)
        query_norm = np.linalg.norm(query_feature)

        # 計算與所有支持動作的相似度
        similarities = {}
        for name, support_feat in support_features.items():
            sim = cosine_similarity(query_feature, support_feat)
            similarities[name] = sim

        # 找出最高相似度和差距
        sorted_sims = sorted(similarities.values(), reverse=True)
        max_sim = sorted_sims[0]
        second_sim = sorted_sims[1] if len(sorted_sims) > 1 else 0
        sim_margin = max_sim - second_sim
        best_action = max(similarities.items(), key=lambda x: x[1])

        # 計算窗口內的平均運動量
        window_motion = motions[start:end-1].mean() if end-1 <= len(motions) else 0

        results.append({
            'frame_start': start,
            'frame_end': end,
            'best_action': best_action[0],
            'max_similarity': max_sim,
            'sim_margin': sim_margin,
            'similarities': similarities,
            'motion': window_motion,
            'query_norm': query_norm
        })

        # 雙重閾值判斷（與 recognition_display_node_v2.py 一致）
        threshold = 0.85
        margin_threshold = 0.05

        if max_sim < threshold:
            recognized = "Unknown"
        elif sim_margin < margin_threshold:
            recognized = "Unknown"
        else:
            recognized = best_action[0]

        print(f"幀 {start:4d}-{end:4d}: {recognized:12s} "
              f"(sim={max_sim:.4f}, margin={sim_margin:.4f}, motion={window_motion:.4f})")
        for name, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {sim:.4f}")

    # 視覺化結果
    print("\n" + "=" * 70)
    print("生成視覺化圖表...")
    print("=" * 70)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # 1. 運動量
    ax1 = axes[0]
    ax1.plot(motions, 'b-', linewidth=0.5)
    ax1.axhline(y=motion_threshold, color='r', linestyle='--', label=f'threshold={motion_threshold:.4f}')
    ax1.set_ylabel('Motion')
    ax1.set_title('Frame-to-Frame Motion (Upper Body)')
    ax1.legend()

    # 2. 各動作的相似度
    ax2 = axes[1]
    frames = [r['frame_start'] for r in results]
    for action_name in support_features.keys():
        sims = [r['similarities'][action_name] for r in results]
        ax2.plot(frames, sims, label=action_name, linewidth=2)
    ax2.axhline(y=0.85, color='k', linestyle='--', label='threshold=0.85')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Cosine Similarity with Support Actions')
    ax2.legend()
    ax2.set_ylim([0.7, 1.0])

    # 3. 最高相似度
    ax3 = axes[2]
    max_sims = [r['max_similarity'] for r in results]
    ax3.plot(frames, max_sims, 'g-', linewidth=2)
    ax3.axhline(y=0.85, color='r', linestyle='--', label='threshold=0.85')
    ax3.set_ylabel('Max Similarity')
    ax3.set_title('Maximum Similarity (for Unknown Detection)')
    ax3.legend()
    ax3.set_ylim([0.7, 1.0])

    # 4. 辨識結果
    ax4 = axes[3]
    action_to_num = {'forwardv5': 1, 'stopv5': 2, 'rightv5': 3, 'Unknown': 0}
    recognized = []
    for r in results:
        if r['max_similarity'] >= 0.85:
            recognized.append(action_to_num[r['best_action']])
        else:
            recognized.append(0)

    ax4.plot(frames, recognized, 'ko-', markersize=4)
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(['Unknown', 'forward', 'stop', 'right'])
    ax4.set_ylabel('Recognized Action')
    ax4.set_xlabel('Frame')
    ax4.set_title('Recognition Result (threshold=0.85)')

    plt.tight_layout()
    output_path = os.path.join(base_dir, 'testv1_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"圖表已儲存: {output_path}")

    # 診斷問題
    print("\n" + "=" * 70)
    print("診斷分析")
    print("=" * 70)

    # 檢查支持特徵之間的相似度
    print("\n支持動作之間的相似度（應該要低）:")
    action_names = list(support_features.keys())
    for i, a1 in enumerate(action_names):
        for a2 in action_names[i+1:]:
            sim = cosine_similarity(support_features[a1], support_features[a2])
            status = "❌ 太高!" if sim > 0.9 else "⚠️ 偏高" if sim > 0.8 else "✓ 正常"
            print(f"  {a1} vs {a2}: {sim:.4f} {status}")

    # 檢查 feature norm 一致性
    print("\n特徵 norm 分析:")
    support_norms = [np.linalg.norm(f) for f in support_features.values()]
    query_norms = [r['query_norm'] for r in results]
    print(f"  Support feature norms: {support_norms}")
    print(f"  Query feature norm range: [{min(query_norms):.2f}, {max(query_norms):.2f}]")
    print(f"  Norm 比例: {np.mean(query_norms) / np.mean(support_norms):.2f}x")

    if np.mean(query_norms) / np.mean(support_norms) > 1.5:
        print("  ❌ Feature norm 不一致！這可能導致辨識問題。")
    else:
        print("  ✓ Feature norm 大致一致")


if __name__ == '__main__':
    main()
