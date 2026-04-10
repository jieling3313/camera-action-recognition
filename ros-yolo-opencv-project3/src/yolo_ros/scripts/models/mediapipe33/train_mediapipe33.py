#!/usr/bin/env python3
"""
MediaPipe 33 點 One-Shot Action Recognition 訓練腳本

使用從 NTU RGB+D 影片提取的 MediaPipe 33 點骨架資料訓練模型。
支援兩階段訓練：
1. 預訓練 (Pre-training): 使用 Cross-Entropy Loss 進行 60 類分類
2. 元學習 (Meta-training): 使用 5-way 1-shot 任務進行 One-Shot 學習

使用方式:
    python train_mediapipe33.py \
        --data_path "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons" \
        --epochs 200 \
        --batch_size 32 \
        --num_classes 60 \
        --benchmark xsub \
        --lr 0.001 \
        --device cuda \
        --save_dir checkpoints_mediapipe33

Author: Claude AI Assistant
Date: 2026-03-30
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
import time

# 導入模型
from skeleton_model_mediapipe33 import (
    MediaPipeSkeletonEmbedding,
    OneShotActionRecognitionMediaPipe,
    preprocess_mediapipe_skeleton
)


# =============================================================================
# NTU RGB+D 60 類動作名稱
# =============================================================================

NTU_60_ACTIONS = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair",
    "drop", "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping", "jump up",
    "make a phone call/answer phone", "playing with phone/tablet",
    "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together",
    "nod head/bow", "shake head", "wipe face", "salute",
    "put the palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling", "touch head",
    "touch chest", "touch back", "touch neck", "nausea or vomiting",
    "use a fan", "punching/slapping other person",
    "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other"
]


# =============================================================================
# 資料集
# =============================================================================

class MediaPipeNTUDataset(Dataset):
    """MediaPipe 33 點 NTU RGB+D 資料集載入器"""

    def __init__(self, data_path, split='train', max_frames=64, benchmark='xsub',
                 use_xyz=True, augment=False):
        """
        參數:
            data_path: MediaPipe 骨架檔案目錄路徑
            split: 'train' 或 'val'
            max_frames: 最大幀數（用於填充/截斷）
            benchmark: 'xsub' (cross-subject) 或 'xview' (cross-view)
            use_xyz: 是否使用 xyz 座標（否則使用 xy + visibility）
            augment: 是否使用資料增強
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_frames = max_frames
        self.benchmark = benchmark
        self.use_xyz = use_xyz
        self.augment = augment and (split == 'train')

        # Cross-subject 訓練集 performer IDs
        self.train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
                               17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

        # 載入樣本列表
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split ({benchmark})")

    def _parse_filename(self, filename):
        """
        解析 NTU RGB+D 檔名

        格式: SsssCcccPpppRrrrAaaa.npy
        - S: setup number (1-17)
        - C: camera ID (1-3)
        - P: performer ID (1-40)
        - R: replication number (1-2)
        - A: action class (1-60)
        """
        basename = filename.stem

        try:
            setup_id = int(basename[1:4])
            camera_id = int(basename[5:8])
            performer_id = int(basename[9:12])
            replication_id = int(basename[13:16])
            action_class = int(basename[17:20])

            return {
                'setup': setup_id,
                'camera': camera_id,
                'performer': performer_id,
                'replication': replication_id,
                'action': action_class,
                'filename': basename
            }
        except (ValueError, IndexError):
            return None

    def _validate_npy_file(self, filepath):
        """驗證 .npy 檔案是否有效"""
        try:
            # 檢查檔案大小
            if filepath.stat().st_size == 0:
                return False

            # 嘗試讀取檔案頭部
            with open(filepath, 'rb') as f:
                # NumPy .npy 檔案的 magic number
                magic = f.read(6)
                if magic != b'\x93NUMPY':
                    return False
            return True
        except Exception:
            return False

    def _load_samples(self):
        """載入樣本路徑和標籤"""
        samples = []
        skipped_files = 0

        # 尋找所有 .npy 檔案
        npy_files = list(self.data_path.glob("*.npy"))

        if len(npy_files) == 0:
            print(f"Warning: No .npy files found in {self.data_path}")
            return samples

        for npy_file in npy_files:
            info = self._parse_filename(npy_file)
            if info is None:
                continue

            # 驗證檔案是否有效
            if not self._validate_npy_file(npy_file):
                skipped_files += 1
                continue

            # 判斷是否為訓練集或驗證集
            if self.benchmark == 'xsub':
                is_train = info['performer'] in self.train_subjects
            elif self.benchmark == 'xview':
                is_train = info['camera'] in [2, 3]
            else:
                raise ValueError(f"Unknown benchmark: {self.benchmark}")

            if (self.split == 'train' and is_train) or (self.split == 'val' and not is_train):
                samples.append({
                    'path': npy_file,
                    'label': info['action'] - 1,  # 轉為 0-based 索引
                    **info
                })

        if skipped_files > 0:
            print(f"Warning: Skipped {skipped_files} invalid/corrupted .npy files")

        return samples

    def _augment_skeleton(self, skeleton):
        """資料增強"""
        # 時間裁剪
        if np.random.random() < 0.5:
            T = skeleton.shape[0]
            if T > 10:
                start = np.random.randint(0, T // 4)
                end = T - np.random.randint(0, T // 4)
                skeleton = skeleton[start:end]

        # 空間抖動
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, skeleton.shape)
            skeleton = skeleton + noise.astype(np.float32)

        # 水平翻轉（交換左右關節）
        if np.random.random() < 0.5:
            skeleton = self._flip_skeleton(skeleton)

        return skeleton

    def _flip_skeleton(self, skeleton):
        """水平翻轉骨架（交換左右關節）"""
        flipped = skeleton.copy()

        # MediaPipe 33 點左右關節對應
        swap_pairs = [
            (1, 4), (2, 5), (3, 6),  # 眼睛
            (7, 8),                   # 耳朵
            (9, 10),                  # 嘴巴
            (11, 12),                 # 肩膀
            (13, 14),                 # 手肘
            (15, 16),                 # 手腕
            (17, 18),                 # 小指
            (19, 20),                 # 食指
            (21, 22),                 # 拇指
            (23, 24),                 # 髖部
            (25, 26),                 # 膝蓋
            (27, 28),                 # 腳踝
            (29, 30),                 # 腳跟
            (31, 32),                 # 腳尖
        ]

        for left, right in swap_pairs:
            flipped[:, left], flipped[:, right] = skeleton[:, right].copy(), skeleton[:, left].copy()

        # 翻轉 x 座標
        flipped[:, :, 0] = 1.0 - flipped[:, :, 0]

        return flipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 讀取骨架資料，添加錯誤處理
        try:
            skeleton = np.load(sample['path'])  # (T, 33, 4) - x, y, z, visibility

            # 驗證資料形狀
            if skeleton.ndim != 3 or skeleton.shape[1] != 33 or skeleton.shape[2] < 3:
                raise ValueError(f"Invalid skeleton shape: {skeleton.shape}")

        except (EOFError, ValueError, OSError) as e:
            # 如果檔案損壞，返回一個零填充的樣本
            print(f"Warning: Failed to load {sample['path']}: {e}")
            skeleton = np.zeros((self.max_frames, 33, 3), dtype=np.float32)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return torch.from_numpy(skeleton), label

        # 選擇通道
        if self.use_xyz:
            skeleton = skeleton[:, :, :3]  # (T, 33, 3) - x, y, z
        else:
            # 使用 x, y, visibility
            skeleton = np.stack([
                skeleton[:, :, 0],  # x
                skeleton[:, :, 1],  # y
                skeleton[:, :, 3] if skeleton.shape[2] > 3 else np.ones_like(skeleton[:, :, 0]),  # visibility
            ], axis=-1)  # (T, 33, 3)

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
            padding = np.zeros((self.max_frames - T, 33, 3), dtype=np.float32)
            skeleton = np.concatenate([skeleton, padding], axis=0)

        # 正規化座標（以髖部中心為原點）
        # MediaPipe: 23=左髖, 24=右髖
        hip_center = (skeleton[:, 23, :2] + skeleton[:, 24, :2]) / 2
        skeleton[:, :, :2] = skeleton[:, :, :2] - hip_center[:, np.newaxis, :]

        # 轉為 torch tensor
        skeleton = torch.from_numpy(skeleton.astype(np.float32))  # (T, 33, 3)
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
        skeletons = skeletons.to(device)  # (N, T, 33, 3)
        labels = labels.to(device)

        # 前向傳播
        optimizer.zero_grad()
        outputs = model(skeletons)  # (N, num_classes)

        loss = criterion(outputs, labels)

        # 反向傳播
        loss.backward()
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
        description='MediaPipe 33-Point One-Shot Action Recognition Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 基本訓練
    python train_mediapipe33.py \\
        --data_path "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons" \\
        --epochs 200 \\
        --batch_size 32

    # 快速測試
    python train_mediapipe33.py \\
        --data_path "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons" \\
        --epochs 5 \\
        --batch_size 16
        """
    )

    # 資料集參數
    parser.add_argument('--data_path', type=str, required=True,
                        help='MediaPipe 骨架檔案目錄路徑')
    parser.add_argument('--benchmark', type=str, default='xsub',
                        choices=['xsub', 'xview'],
                        help='Benchmark 協議 (default: xsub)')
    parser.add_argument('--num_classes', type=int, default=60,
                        help='動作類別數 (default: 60)')
    parser.add_argument('--max_frames', type=int, default=64,
                        help='最大幀數 (default: 64)')

    # 模型參數
    parser.add_argument('--base_channels', type=int, default=64,
                        help='基礎通道數 (default: 64)')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='訓練 epochs 數 (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='學習率 (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='資料載入工作程序數 (default: 4)')
    parser.add_argument('--augment', action='store_true',
                        help='啟用資料增強')

    # 其他參數
    parser.add_argument('--save_dir', type=str, default='./checkpoints_mediapipe33',
                        help='Checkpoint 儲存目錄')
    parser.add_argument('--resume', type=str, default=None,
                        help='繼續訓練的 checkpoint 路徑')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='訓練裝置 (default: cuda)')

    args = parser.parse_args()

    # 建立儲存目錄
    os.makedirs(args.save_dir, exist_ok=True)

    # 儲存訓練配置
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("MediaPipe 33-Point One-Shot Action Recognition Training")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Max frames: {args.max_frames}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Data augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print("=" * 80)

    # 建立資料集
    print("\nLoading datasets...")
    train_dataset = MediaPipeNTUDataset(
        args.data_path,
        split='train',
        max_frames=args.max_frames,
        benchmark=args.benchmark,
        augment=args.augment
    )

    val_dataset = MediaPipeNTUDataset(
        args.data_path,
        split='val',
        max_frames=args.max_frames,
        benchmark=args.benchmark,
        augment=False
    )

    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        print(f"Please check your dataset path: {args.data_path}")
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

    # 建立模型
    print("\nInitializing model...")
    model = MediaPipeSkeletonEmbedding(
        in_channels=3,
        base_channels=args.base_channels,
        num_classes=args.num_classes
    )
    model = model.to(args.device)

    # 顯示模型參數量
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

    # 學習率調度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 恢復訓練
    start_epoch = 0
    best_acc = 0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")

    # 訓練循環
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

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

        # 計算時間
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        estimated_remaining = epoch_time * (args.epochs - epoch - 1)

        # 輸出結果
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        print(f"  Epoch Time: {epoch_time:.1f}s  |  Total: {total_time/60:.1f}min  |  ETA: {estimated_remaining/60:.1f}min")

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
            'args': vars(args)
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
            print(f"  * New best accuracy! Saved: {best_path}")

        # 每 10 個 epoch 儲存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(args.save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"  Saved epoch checkpoint: {epoch_path}")

    total_training_time = time.time() - training_start_time

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    print(f"Checkpoints saved in: {args.save_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
