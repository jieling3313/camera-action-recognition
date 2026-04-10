#!/usr/bin/env python3
"""
離線測試動作辨識模型
測試 actionset 中的動作是否可以被正確區分
"""

import numpy as np
import torch
import sys
import os

# 添加路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def load_model(checkpoint_path):
    """載入模型"""
    model = OneShotActionRecognitionMediaPipe(in_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded: Epoch {checkpoint.get('epoch', 'N/A')}, Acc {checkpoint.get('best_acc', 'N/A')}%")
    return model

def sample_to_fixed_length(sample, target_length=64):
    """將骨架序列調整為固定長度"""
    T = sample.shape[0]
    if T == target_length:
        return sample
    elif T > target_length:
        indices = np.linspace(0, T - 1, target_length, dtype=int)
        return sample[indices]
    else:
        pad_length = target_length - T
        padding = np.repeat(sample[-1:], pad_length, axis=0)
        return np.concatenate([sample, padding], axis=0)

def extract_feature(model, skeleton_data):
    """提取特徵向量"""
    # 調整為 64 幀
    skeleton_data = sample_to_fixed_length(skeleton_data, 64)

    # 轉換為 tensor
    tensor = torch.FloatTensor(skeleton_data).unsqueeze(0)  # (1, 64, 33, 3)

    with torch.no_grad():
        feature = model.embedding(tensor)  # (1, 256)
        feature = feature.numpy()[0]

    return feature

def cosine_similarity(f1, f2):
    """計算餘弦相似度"""
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)

def softmax_confidence(similarities, temperature=0.05):
    """使用 softmax 計算相對信心度

    Args:
        similarities: dict {action_name: similarity}
        temperature: 溫度參數，越小差異越大

    Returns:
        dict {action_name: confidence%}
    """
    sims = np.array(list(similarities.values()))
    names = list(similarities.keys())

    # Softmax with temperature
    exp_sims = np.exp((sims - sims.max()) / temperature)
    softmax_scores = exp_sims / exp_sims.sum()

    return {name: score * 100 for name, score in zip(names, softmax_scores)}

def main():
    # 路徑設定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoints_mediapipe33', 'latest.pth')
    actionset_dir = os.path.join(base_dir, '..', '..', '..', 'actionset')

    print("=" * 70)
    print("動作辨識模型離線測試")
    print("=" * 70)

    # 載入模型
    model = load_model(checkpoint_path)

    # 測試指定的動作（可通過命令行參數指定）
    # 用法: python test_action_recognition.py forwardv4 rightv4 stop0409
    actions = []

    if len(sys.argv) > 1:
        # 使用命令行參數指定的動作
        target_actions = sys.argv[1:]
        print(f"測試指定動作: {target_actions}")
        for action_name in target_actions:
            npy_file = os.path.join(actionset_dir, action_name, f'{action_name}_skeleton_sequence.npy')
            if os.path.exists(npy_file):
                actions.append((action_name, npy_file))
            else:
                print(f"  ⚠️ 找不到: {npy_file}")
    else:
        # 預設：找所有 v4 動作 + stop0409
        for d in sorted(os.listdir(actionset_dir)):
            if 'v4' in d or d == 'stop0409':  # v4 版本 + stop0409
                npy_file = os.path.join(actionset_dir, d, f'{d}_skeleton_sequence.npy')
                if os.path.exists(npy_file):
                    actions.append((d, npy_file))

    if not actions:
        print("找不到指定動作，嘗試所有動作...")
        for d in sorted(os.listdir(actionset_dir)):
            if d.startswith('.'):
                continue
            npy_file = os.path.join(actionset_dir, d, f'{d}_skeleton_sequence.npy')
            if os.path.exists(npy_file):
                actions.append((d, npy_file))

    print(f"\n找到 {len(actions)} 個動作:")
    for name, path in actions:
        print(f"  - {name}")

    # 載入並分析數據
    print("\n" + "=" * 70)
    print("數據分析")
    print("=" * 70)

    data_dict = {}
    for name, path in actions:
        data = np.load(path)
        data_dict[name] = data

        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  X range: [{data[..., 0].min():.4f}, {data[..., 0].max():.4f}]")
        print(f"  Y range: [{data[..., 1].min():.4f}, {data[..., 1].max():.4f}]")
        print(f"  Visibility range: [{data[..., 2].min():.4f}, {data[..., 2].max():.4f}]")

        # 檢查是否需要正規化
        if data[..., 0].max() > 2 or data[..., 1].max() > 2:
            print(f"  ⚠️  座標超過 2，可能是像素座標！")
        else:
            print(f"  ✓ 座標在正規化範圍內")

    # 運動量分析
    analyze_motion(data_dict)

    # 提取特徵
    print("\n" + "=" * 70)
    print("特徵提取")
    print("=" * 70)

    features = {}
    for name, data in data_dict.items():
        feature = extract_feature(model, data)
        features[name] = feature
        print(f"{name}: mean={feature.mean():.4f}, std={feature.std():.4f}, norm={np.linalg.norm(feature):.4f}")

    # 計算相似度矩陣
    print("\n" + "=" * 70)
    print("相似度矩陣（應該是：同動作高，不同動作低）")
    print("=" * 70)

    action_names = list(features.keys())
    n = len(action_names)

    # 打印表頭
    header = "            " + "  ".join([f"{name[:10]:>10}" for name in action_names])
    print(header)

    for i, a1 in enumerate(action_names):
        row = f"{a1[:10]:>10}  "
        for j, a2 in enumerate(action_names):
            sim = cosine_similarity(features[a1], features[a2])
            conf = (sim + 1) * 50
            row += f"{sim:>10.4f}  "
        print(row)

    # 分析結果
    print("\n" + "=" * 70)
    print("診斷結果")
    print("=" * 70)

    # 計算不同動作間的平均相似度
    same_action_sims = []
    diff_action_sims = []

    for i, a1 in enumerate(action_names):
        for j, a2 in enumerate(action_names):
            sim = cosine_similarity(features[a1], features[a2])
            if i == j:
                same_action_sims.append(sim)
            else:
                diff_action_sims.append(sim)

    avg_same = np.mean(same_action_sims) if same_action_sims else 0
    avg_diff = np.mean(diff_action_sims) if diff_action_sims else 0

    print(f"同動作平均相似度: {avg_same:.4f}")
    print(f"不同動作平均相似度: {avg_diff:.4f}")
    print(f"區分度 (差值): {avg_same - avg_diff:.4f}")

    if avg_diff > 0.9:
        print("\n❌ 問題：不同動作之間相似度太高 (>0.9)")
        print("   可能原因：")
        print("   1. 模型沒有正確訓練來區分這些動作")
        print("   2. 動作之間差異太小")
        print("   3. 數據預處理不一致")
    elif avg_diff > 0.7:
        print("\n⚠️  警告：不同動作之間相似度較高 (0.7-0.9)")
        print("   模型可能難以區分某些動作")
    else:
        print("\n✓ 模型應該能夠區分這些動作")

    # 模擬即時辨識
    print("\n" + "=" * 70)
    print("模擬即時辨識（使用每個動作的不同片段）")
    print("=" * 70)

    for query_name, query_data in data_dict.items():
        print(f"\n查詢動作: {query_name}")

        # 使用動作的後半部分作為查詢（模擬即時輸入）
        T = query_data.shape[0]
        query_segment = query_data[T//2:]  # 後半部分
        query_feature = extract_feature(model, query_segment)

        # 與所有支持集比較（原始相似度）
        raw_sims = {}
        for support_name, support_feature in features.items():
            sim = cosine_similarity(query_feature, support_feature)
            raw_sims[support_name] = sim

        # 計算 softmax 信心度
        softmax_conf = softmax_confidence(raw_sims, temperature=0.05)

        # 排序結果
        sorted_results = sorted(softmax_conf.items(), key=lambda x: x[1], reverse=True)

        predicted = sorted_results[0][0]
        correct = "✓" if predicted == query_name else "❌"

        print(f"  預測結果: {predicted} {correct}")
        print(f"  [原始相似度] vs [Softmax 信心度]:")
        for name, conf in sorted_results:
            sim = raw_sims[name]
            marker = "←" if name == query_name else ""
            print(f"    {name}: sim={sim:.4f} -> conf={conf:.1f}% {marker}")

    # 測試不同的溫度參數
    print("\n" + "=" * 70)
    print("溫度參數測試（調整區分度）")
    print("=" * 70)

    test_temps = [0.1, 0.05, 0.02, 0.01]
    query_name = list(data_dict.keys())[0]
    query_data = data_dict[query_name]
    T = query_data.shape[0]
    query_segment = query_data[T//2:]
    query_feature = extract_feature(model, query_segment)

    raw_sims = {}
    for support_name, support_feature in features.items():
        raw_sims[support_name] = cosine_similarity(query_feature, support_feature)

    print(f"查詢動作: {query_name}")
    print(f"原始相似度: {raw_sims}")
    print()

    for temp in test_temps:
        conf = softmax_confidence(raw_sims, temperature=temp)
        sorted_conf = sorted(conf.items(), key=lambda x: x[1], reverse=True)
        print(f"Temperature={temp}:")
        for name, c in sorted_conf:
            print(f"  {name}: {c:.1f}%")
        print()

    # 模擬即時辨識場景（使用滑動窗口的最後 64 幀）
    print("\n" + "=" * 70)
    print("模擬即時滑動窗口辨識")
    print("=" * 70)

    print("\n這模擬了即時辨識時使用最後 64 幀的情況：")

    for query_name, query_data in data_dict.items():
        print(f"\n動作: {query_name}")
        T = query_data.shape[0]

        # 測試不同的窗口位置
        positions = ['開始', '中間', '結束']
        segments = [
            query_data[:64],           # 開始 64 幀
            query_data[T//2-32:T//2+32] if T >= 64 else query_data,  # 中間 64 幀
            query_data[-64:] if T >= 64 else query_data,  # 結束 64 幀
        ]

        for pos, segment in zip(positions, segments):
            if len(segment) < 64:
                segment = sample_to_fixed_length(segment, 64)

            query_feature = extract_feature(model, segment)

            raw_sims = {}
            for support_name, support_feature in features.items():
                raw_sims[support_name] = cosine_similarity(query_feature, support_feature)

            conf = softmax_confidence(raw_sims, temperature=0.02)
            sorted_conf = sorted(conf.items(), key=lambda x: x[1], reverse=True)
            predicted = sorted_conf[0][0]
            correct = "✓" if predicted == query_name else "❌"

            print(f"  {pos}片段: 預測={predicted} {correct}")
            for name, c in sorted_conf:
                sim = raw_sims[name]
                marker = "←" if name == query_name else ""
                print(f"    {name}: sim={sim:.4f}, conf={c:.1f}% {marker}")

def analyze_motion(data_dict):
    """分析每個動作的運動量"""
    print("\n" + "=" * 70)
    print("動作運動量分析（判斷是動態還是靜態動作）")
    print("=" * 70)

    for name, data in data_dict.items():
        # 計算幀間變化量（運動量）
        frame_diffs = np.diff(data[:, :, :2], axis=0)  # (T-1, 33, 2)
        frame_motion = np.sqrt(np.sum(frame_diffs ** 2, axis=(1, 2)))  # (T-1,)

        # 上半身關鍵點的運動量（更能代表動作）
        upper_body_idx = list(range(11, 23))  # 肩膀、手肘、手腕等
        upper_motion = np.sqrt(np.sum(frame_diffs[:, upper_body_idx, :] ** 2, axis=(1, 2)))

        avg_motion = frame_motion.mean()
        max_motion = frame_motion.max()
        upper_avg = upper_motion.mean()

        # 判斷是否為動態動作
        is_dynamic = upper_avg > 0.05  # 閾值可調整

        status = "動態 🏃" if is_dynamic else "靜態 🧍"

        print(f"\n{name}:")
        print(f"  總運動量: avg={avg_motion:.4f}, max={max_motion:.4f}")
        print(f"  上半身運動量: avg={upper_avg:.4f}")
        print(f"  類型判定: {status}")


if __name__ == '__main__':
    main()
