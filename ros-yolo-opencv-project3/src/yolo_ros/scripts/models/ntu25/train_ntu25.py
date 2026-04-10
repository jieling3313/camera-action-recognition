#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTU RGB+D 25 點預訓練腳本

使用 NTU RGB+D 60/120 數據集預訓練 NTUSkeletonEmbedding 模型
直接使用原生 25 關節格式，無需映射，無資訊損失

支援：
- NTU RGB+D 60 (60 個動作類別)
- NTU RGB+D 120 (120 個動作類別)
- Cross-subject (xsub) 和 Cross-view (xview) benchmark

使用方式：
    python train_ntu25.py --data_path /path/to/ntu_skeletons --benchmark xsub --num_classes 60

    # 使用 GPU
    python train_ntu25.py --data_path /path/to/ntu_skeletons --device cuda

    # 從 checkpoint 恢復訓練
    python train_ntu25.py --data_path /path/to/ntu_skeletons --resume ./checkpoints_ntu25/latest.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 添加父目錄到路徑以便導入
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton_model_ntu25 import NTUSkeletonEmbedding, NTUJoint


# =============================================================================
# NTU RGB+D 數據集
# =============================================================================

class NTU25Dataset(Dataset):
    """
    NTU RGB+D 數據集加載器 - 原生 25 點格式

    直接使用 NTU RGB+D 的 25 關節格式，無需轉換。
    """

    # Cross-subject benchmark 訓練集的 performer ID
    XSUB_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

    # Cross-view benchmark 訓練集的 camera ID
    XVIEW_TRAIN_CAMERAS = [2, 3]

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_frames: int = 300,
        benchmark: str = 'xsub',
        num_classes: int = 60,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        參數：
            data_path: NTU RGB+D 數據集根目錄（包含 .skeleton 檔案）
            split: 'train' 或 'val'
            max_frames: 最大幀數（用於填充/截斷）
            benchmark: 'xsub' (cross-subject) 或 'xview' (cross-view)
            num_classes: 動作類別數量 (60 或 120)
            normalize: 是否正規化骨架座標
            augment: 是否進行資料增強
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_frames = max_frames
        self.benchmark = benchmark
        self.num_classes = num_classes
        self.normalize = normalize
        self.augment = augment and (split == 'train')

        # 載入樣本列表
        self.samples = self._load_samples()

        print(f"[NTU25Dataset] Loaded {len(self.samples)} samples for {split} ({benchmark})")

    def _load_samples(self):
        """載入樣本路徑和標籤"""
        samples = []

        # 尋找 .skeleton 檔案
        skeleton_files = list(self.data_path.glob("**/*.skeleton"))

        if len(skeleton_files) == 0:
            print(f"[Warning] No .skeleton files found in {self.data_path}")
            return samples

        for skeleton_file in skeleton_files:
            # 從檔名解析資訊
            # 格式：SsssCcccPpppRrrrAaaa.skeleton
            filename = skeleton_file.stem

            try:
                setup_id = int(filename[1:4])
                camera_id = int(filename[5:8])
                performer_id = int(filename[9:12])
                replication_id = int(filename[13:16])
                action_class = int(filename[17:20])
            except (ValueError, IndexError):
                print(f"[Warning] Cannot parse filename {filename}, skipping")
                continue

            # 過濾超出類別範圍的樣本
            if action_class > self.num_classes:
                continue

            # 判斷是否為訓練集或驗證集
            if self.benchmark == 'xsub':
                is_train = performer_id in self.XSUB_TRAIN_SUBJECTS
            elif self.benchmark == 'xview':
                is_train = camera_id in self.XVIEW_TRAIN_CAMERAS
            else:
                raise ValueError(f"Unknown benchmark: {self.benchmark}")

            if (self.split == 'train' and is_train) or (self.split == 'val' and not is_train):
                samples.append({
                    'path': skeleton_file,
                    'label': action_class - 1,  # 轉為 0-based 索引
                    'setup': setup_id,
                    'camera': camera_id,
                    'performer': performer_id,
                    'replication': replication_id,
                    'filename': filename
                })

        return samples

    def _read_skeleton_file(self, filepath):
        """
        讀取 .skeleton 檔案

        NTU RGB+D skeleton 檔案格式：
        - 第一行：幀數
        - 每幀：
            - 人數
            - 每人：
                - 身體資訊（10 個數值）
                - 關節數
                - 每關節：11 個數值 (x, y, z, depthX, depthY, colorX, colorY,
                          orientationW, orientationX, orientationY, orientationZ)
        """
        try:
            with open(filepath, 'r') as f:
                frame_count = int(f.readline().strip())

                frames = []
                for _ in range(frame_count):
                    body_count = int(f.readline().strip())

                    if body_count == 0:
                        # 無人幀，跳過
                        continue

                    # 處理每個人
                    frame_joints = None
                    for body_id in range(body_count):
                        # 讀取身體資訊
                        body_info = f.readline().strip()

                        # 讀取關節數
                        joint_count = int(f.readline().strip())

                        # 讀取關節資料
                        joints = []
                        for _ in range(joint_count):
                            joint_data = f.readline().strip().split()
                            if len(joint_data) >= 3:
                                x = float(joint_data[0])
                                y = float(joint_data[1])
                                z = float(joint_data[2])
                                joints.append([x, y, z])

                        # 只使用第一個有效的人
                        if body_id == 0 and len(joints) == 25:
                            frame_joints = joints

                    if frame_joints is not None:
                        frames.append(frame_joints)

                # 確保至少有一幀資料
                if len(frames) == 0:
                    return np.zeros((1, 25, 3), dtype=np.float32)

                result = np.array(frames, dtype=np.float32)

                # 驗證形狀
                if result.ndim != 3 or result.shape[1] != 25 or result.shape[2] != 3:
                    return np.zeros((1, 25, 3), dtype=np.float32)

                return result

        except Exception as e:
            print(f"[Warning] Failed to read {filepath}: {e}")
            return np.zeros((1, 25, 3), dtype=np.float32)

    def _normalize_skeleton(self, skeleton):
        """
        正規化骨架座標

        以 Spine Base (關節 0) 為中心，並根據肩寬進行縮放
        """
        # (T, 25, 3)
        T = skeleton.shape[0]

        # 以 Spine Base 為中心
        spine_base = skeleton[:, NTUJoint.SPINE_BASE, :].copy()  # (T, 3)
        skeleton = skeleton - spine_base[:, np.newaxis, :]

        # 計算肩寬用於縮放
        left_shoulder = skeleton[:, NTUJoint.SHOULDER_LEFT, :]
        right_shoulder = skeleton[:, NTUJoint.SHOULDER_RIGHT, :]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)  # (T,)

        # 避免除以零
        shoulder_width = np.maximum(shoulder_width, 0.1)

        # 縮放
        skeleton = skeleton / shoulder_width[:, np.newaxis, np.newaxis]

        return skeleton

    def _augment_skeleton(self, skeleton):
        """
        資料增強

        包含：
        - 隨機旋轉 (繞 Y 軸)
        - 隨機縮放
        - 隨機平移
        - 隨機時間裁剪
        """
        T, V, C = skeleton.shape

        # 隨機旋轉 (繞 Y 軸，±30 度)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
            skeleton = skeleton @ rotation_matrix.T

        # 隨機縮放 (0.9 ~ 1.1)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            skeleton = skeleton * scale

        # 隨機平移
        if np.random.rand() > 0.5:
            translation = np.random.uniform(-0.1, 0.1, size=(1, 1, 3))
            skeleton = skeleton + translation

        return skeleton

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 讀取骨架資料
        skeleton = self._read_skeleton_file(sample['path'])  # (T, 25, 3)

        # 正規化
        if self.normalize:
            skeleton = self._normalize_skeleton(skeleton)

        # 資料增強
        if self.augment:
            skeleton = self._augment_skeleton(skeleton)

        # 填充或截斷到固定長度
        T = skeleton.shape[0]
        if T > self.max_frames:
            # 均勻採樣
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            skeleton = skeleton[indices]
        elif T < self.max_frames:
            # 零填充
            padding = np.zeros((self.max_frames - T, 25, 3), dtype=np.float32)
            skeleton = np.concatenate([skeleton, padding], axis=0)

        # 轉為 torch tensor
        skeleton = torch.from_numpy(skeleton).float()  # (T, 25, 3)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return skeleton, label


# =============================================================================
# 訓練函數
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for skeletons, labels in pbar:
        skeletons = skeletons.to(device)  # (N, T, 25, 3)
        labels = labels.to(device)

        # 前向傳播
        optimizer.zero_grad()
        outputs = model(skeletons)  # (N, num_classes)

        loss = criterion(outputs, labels)

        # 反向傳播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 統計
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """驗證"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for skeletons, labels in tqdm(dataloader, desc="Validating"):
            skeletons = skeletons.to(device)
            labels = labels.to(device)

            outputs = model(skeletons)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# =============================================================================
# 主程式
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NTU RGB+D 25-Point Pretraining',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 數據集參數
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NTU RGB+D skeleton dataset')
    parser.add_argument('--benchmark', type=str, default='xsub',
                        choices=['xsub', 'xview'],
                        help='Benchmark protocol')
    parser.add_argument('--num_classes', type=int, default=60,
                        choices=[60, 120],
                        help='Number of action classes')

    # 模型參數
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel number in AGCN')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum number of frames')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 資料增強
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')

    # 其他參數
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ntu25',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # 建立儲存目錄
    os.makedirs(args.save_dir, exist_ok=True)

    # 儲存訓練配置
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 列印訓練資訊
    print("=" * 80)
    print("NTU RGB+D 25-Point Pretraining for One-Shot Action Recognition")
    print("=" * 80)
    print(f"Skeleton Format: NTU 25 joints (native, no mapping)")
    print(f"Dataset: {args.data_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Data augmentation: {args.augment}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # 建立數據集
    print("\nLoading datasets...")
    train_dataset = NTU25Dataset(
        args.data_path,
        split='train',
        max_frames=args.max_frames,
        benchmark=args.benchmark,
        num_classes=args.num_classes,
        normalize=True,
        augment=args.augment
    )

    val_dataset = NTU25Dataset(
        args.data_path,
        split='val',
        max_frames=args.max_frames,
        benchmark=args.benchmark,
        num_classes=args.num_classes,
        normalize=True,
        augment=False
    )

    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        print("Please check your dataset path and make sure .skeleton files are present.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 建立模型
    print("\nInitializing model...")
    model = NTUSkeletonEmbedding(
        in_channels=3,
        base_channels=args.base_channels,
        num_classes=args.num_classes
    )
    model = model.to(args.device)

    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 學習率調度器 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 恢復訓練
    start_epoch = 0
    best_acc = 0
    training_history = []

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
        training_history = checkpoint.get('history', [])
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")

    # 訓練循環
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'=' * 80}")

        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch + 1
        )

        # 驗證
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)

        # 更新學習率
        scheduler.step()

        # 記錄歷史
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(history_entry)

        # 輸出結果
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")

        # 儲存 checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_acc': best_acc,
            'history': training_history,
            'args': vars(args),
            'model_config': {
                'num_joints': 25,
                'in_channels': 3,
                'base_channels': args.base_channels,
                'num_classes': args.num_classes
            }
        }

        # 儲存最新的 checkpoint
        latest_path = os.path.join(args.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        print(f"  Saved checkpoint: {latest_path}")

        # 儲存最佳 checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint['best_acc'] = best_acc
            best_path = os.path.join(args.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  New best accuracy! Saved: {best_path}")

        # 每 10 個 epoch 儲存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(args.save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"  Saved epoch checkpoint: {epoch_path}")

        # 儲存訓練歷史
        history_path = os.path.join(args.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

    # 訓練完成
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved in: {args.save_dir}")
    print("=" * 80 + "\n")

    # 輸出最終摘要
    print("Training Summary:")
    print(f"  - Skeleton format: NTU 25 joints")
    print(f"  - Total epochs: {args.epochs}")
    print(f"  - Best validation accuracy: {best_acc:.2f}%")
    print(f"  - Model saved at: {os.path.join(args.save_dir, 'best.pth')}")
    print("\nTo use the trained model for One-Shot Action Recognition:")
    print("  from models.ntu25 import OneShotActionRecognitionNTU25")
    print(f"  model = OneShotActionRecognitionNTU25()")
    print(f"  model.embedding.load_state_dict(torch.load('{os.path.join(args.save_dir, 'best.pth')}')['model_state_dict'])")


if __name__ == '__main__':
    main()
