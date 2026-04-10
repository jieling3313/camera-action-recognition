#!/usr/bin/env python3
"""
診斷 feature norm 不一致的問題

目標：找出為什麼 support features 的 norm (~50) 與 query features (~104) 不同
"""

import numpy as np
import torch
import sys
import os

# 添加路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe

def load_model(checkpoint_path, device='cpu'):
    """載入模型（與 model_manager.py 和 recognition_display_node_v2.py 一致）"""
    model = OneShotActionRecognitionMediaPipe(in_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model, checkpoint

def extract_feature(model, skeleton_data, device='cpu'):
    """提取特徵向量"""
    tensor = torch.FloatTensor(skeleton_data).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.embedding(tensor)  # (1, 256)
        feature = feature.cpu().numpy()[0]
    return feature

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, 'checkpoints_mediapipe33')
    actionset_dir = os.path.join(base_dir, '..', '..', '..', 'actionset')

    # 載入 v5 動作
    actions = {
        'forwardv5': os.path.join(actionset_dir, 'forwardv5', 'forwardv5_skeleton_sequence.npy'),
        'rightv5': os.path.join(actionset_dir, 'rightv5', 'rightv5_skeleton_sequence.npy'),
        'stopv5': os.path.join(actionset_dir, 'stopv5', 'stopv5_skeleton_sequence.npy'),
    }

    print("=" * 70)
    print("Feature Norm 診斷")
    print("=" * 70)

    # 測試 1: 比較 best.pth 和 latest.pth
    print("\n[1] Checkpoint 比較")
    print("-" * 70)

    for ckpt_name in ['best.pth', 'latest.pth']:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  {ckpt_name}: 找不到")
            continue

        model, checkpoint = load_model(ckpt_path)
        print(f"\n  {ckpt_name}:")
        print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"    Acc: {checkpoint.get('best_acc', 'N/A')}%")

        # 測試用相同輸入
        test_input = np.random.randn(64, 33, 3).astype(np.float32) * 0.1 + 0.5
        feature = extract_feature(model, test_input)
        print(f"    Random input feature norm: {np.linalg.norm(feature):.4f}")

    # 測試 2: 同一模型，不同輸入數據（模擬 support vs query）
    print("\n" + "=" * 70)
    print("[2] 同模型不同輸入的 feature norm")
    print("-" * 70)

    model, _ = load_model(os.path.join(checkpoint_dir, 'best.pth'))

    for name, path in actions.items():
        if not os.path.exists(path):
            continue

        data = np.load(path)
        print(f"\n  {name}: shape={data.shape}")
        print(f"    Data range: x=[{data[...,0].min():.3f}, {data[...,0].max():.3f}], "
              f"y=[{data[...,1].min():.3f}, {data[...,1].max():.3f}]")

        # 方式 1: 完整序列（取前 64 幀或補齊）- 模擬 model_manager.py
        if data.shape[0] >= 64:
            sample1 = data[:64]
        else:
            pad_len = 64 - data.shape[0]
            sample1 = np.concatenate([data, np.repeat(data[-1:], pad_len, axis=0)], axis=0)

        feature1 = extract_feature(model, sample1)
        print(f"    方式1 (前64幀/補齊): norm={np.linalg.norm(feature1):.4f}")

        # 方式 2: 使用最後 64 幀 - 模擬 recognition_display_node_v2.py 的 sliding window
        if data.shape[0] >= 64:
            sample2 = data[-64:]
        else:
            sample2 = sample1  # 不夠 64 幀就用同樣的

        feature2 = extract_feature(model, sample2)
        print(f"    方式2 (後64幀): norm={np.linalg.norm(feature2):.4f}")

        # 方式 3: 均勻取樣到 64 幀
        T = data.shape[0]
        indices = np.linspace(0, T - 1, 64, dtype=int)
        sample3 = data[indices]
        feature3 = extract_feature(model, sample3)
        print(f"    方式3 (均勻取樣): norm={np.linalg.norm(feature3):.4f}")

    # 測試 3: 數據正規化方式
    print("\n" + "=" * 70)
    print("[3] 數據正規化對 feature norm 的影響")
    print("-" * 70)

    # 使用 forwardv5 測試
    test_path = actions.get('forwardv5')
    if test_path and os.path.exists(test_path):
        data = np.load(test_path)
        sample = data[:64] if data.shape[0] >= 64 else np.concatenate([data, np.repeat(data[-1:], 64 - data.shape[0], axis=0)], axis=0)

        # 原始數據
        feature_orig = extract_feature(model, sample)
        print(f"  原始數據: norm={np.linalg.norm(feature_orig):.4f}")

        # 中心化（以髖部中心為原點）
        sample_centered = sample.copy()
        center = (sample_centered[:, 23, :2] + sample_centered[:, 24, :2]) / 2
        sample_centered[:, :, :2] -= center[:, np.newaxis, :]
        feature_centered = extract_feature(model, sample_centered)
        print(f"  中心化後: norm={np.linalg.norm(feature_centered):.4f}")

        # 縮放到 0-1
        sample_scaled = sample.copy()
        sample_scaled[..., :2] = (sample_scaled[..., :2] - sample_scaled[..., :2].min()) / (sample_scaled[..., :2].max() - sample_scaled[..., :2].min() + 1e-8)
        feature_scaled = extract_feature(model, sample_scaled)
        print(f"  縮放到0-1: norm={np.linalg.norm(feature_scaled):.4f}")

    # 測試 4: BatchNorm 的影響
    print("\n" + "=" * 70)
    print("[4] BatchNorm 行為檢查")
    print("-" * 70)

    # 檢查 model.training 狀態
    print(f"  model.training = {model.training}")
    print(f"  model.embedding.data_bn.training = {model.embedding.data_bn.training}")

    # 檢查 running stats
    data_bn = model.embedding.data_bn
    print(f"  data_bn.running_mean: shape={data_bn.running_mean.shape}, "
          f"min={data_bn.running_mean.min().item():.4f}, max={data_bn.running_mean.max().item():.4f}")
    print(f"  data_bn.running_var: shape={data_bn.running_var.shape}, "
          f"min={data_bn.running_var.min().item():.4f}, max={data_bn.running_var.max().item():.4f}")

    # 測試 5: 使用 train() 模式的結果（模擬可能的錯誤情況）
    print("\n" + "=" * 70)
    print("[5] train() vs eval() 模式")
    print("-" * 70)

    test_sample = sample.copy()

    # eval() 模式
    model.eval()
    feature_eval = extract_feature(model, test_sample)
    print(f"  eval() 模式: norm={np.linalg.norm(feature_eval):.4f}")

    # train() 模式（不應該用，但測試看看）
    model.train()
    feature_train = extract_feature(model, test_sample)
    print(f"  train() 模式: norm={np.linalg.norm(feature_train):.4f}")

    # 差異
    diff = np.linalg.norm(feature_eval - feature_train)
    print(f"  eval vs train 差異: {diff:.4f}")

    print("\n" + "=" * 70)
    print("診斷完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
