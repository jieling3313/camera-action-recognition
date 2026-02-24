#!/usr/bin/env python3
"""
NTU RGB+D 預訓練腳本

使用 NTU RGB+D 60/120 數據集預訓練 SkeletonEmbedding 模型
支援動作分類任務，訓練完成後可用於 One-Shot Action Recognition

數據集格式：
- NTU RGB+D 原始格式：每個 .skeleton 檔案包含 25 個關節
- 本腳本會自動將 25 關節映射到 COCO 17 關節格式
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pickle

# 導入模型
from skeleton_model import SkeletonEmbedding


# =============================================================================
# NTU RGB+D 關節映射到 COCO 17 格式
# =============================================================================

# NTU RGB+D 25 關節索引（0-based）：
# 0: 脊椎基部, 1: 脊椎中部, 2: 頸部, 3: 頭部,
# 4: 左肩, 5: 左肘, 6: 左腕, 7: 左手,
# 8: 右肩, 9: 右肘, 10: 右腕, 11: 右手,
# 12: 左髖, 13: 左膝, 14: 左踝, 15: 左腳,
# 16: 右髖, 17: 右膝, 18: 右踝, 19: 右腳,
# 20: 脊椎, 21: 左手尖, 22: 左拇指, 23: 右手尖, 24: 右拇指

# COCO 17 關節索引：
# 0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳,
# 5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘,
# 9: 左腕, 10: 右腕, 11: 左髖, 12: 右髖,
# 13: 左膝, 14: 右膝, 15: 左踝, 16: 右踝

NTU_TO_COCO_MAP = {
    0: 3,   # 頭部 -> 鼻子（近似）
    1: None, 2: None,  # 左眼、右眼（NTU 沒有）
    3: None, 4: None,  # 左耳、右耳（NTU 沒有）
    5: 4,   # 左肩
    6: 8,   # 右肩
    7: 5,   # 左肘
    8: 9,   # 右肘
    9: 6,   # 左腕
    10: 10, # 右腕
    11: 12, # 左髖
    12: 16, # 右髖
    13: 13, # 左膝
    14: 17, # 右膝
    15: 14, # 左踝
    16: 18, # 右踝
}


def ntu_skeleton_to_coco(ntu_skeleton):
    """
    將 NTU RGB+D 25 關節骨架轉換為 COCO 17 關節格式。

    參數：
        ntu_skeleton: (T, 25, 3) 或 (M, T, 25, 3)，其中 M 是人數，T 是幀數

    回傳：
        coco_skeleton: (T, 17, 3) 或 (M, T, 17, 3)
    """
    # 檢查輸入形狀是否有效
    if ntu_skeleton.ndim < 3:
        print(f"Warning: Invalid skeleton shape {ntu_skeleton.shape}, returning zeros")
        return np.zeros((1, 17, 3), dtype=np.float32)

    if ntu_skeleton.ndim == 3:
        # 單人情況 (T, 25, 3)
        T, V, C = ntu_skeleton.shape

        # 檢查關節數是否正確
        if V != 25:
            print(f"Warning: Expected 25 joints but got {V}, returning zeros")
            return np.zeros((T, 17, 3), dtype=np.float32)

        coco_skeleton = np.zeros((T, 17, C), dtype=np.float32)

        for coco_idx, ntu_idx in NTU_TO_COCO_MAP.items():
            if ntu_idx is not None:
                coco_skeleton[:, coco_idx, :] = ntu_skeleton[:, ntu_idx, :]

        # 處理 NTU 沒有的關節（設定低置信度）
        # 眼睛和耳朵用頭部位置近似
        head_pos = ntu_skeleton[:, 3, :]  # 頭部
        coco_skeleton[:, 1, :2] = head_pos[:, :2]  # 左眼
        coco_skeleton[:, 2, :2] = head_pos[:, :2]  # 右眼
        coco_skeleton[:, 3, :2] = head_pos[:, :2]  # 左耳
        coco_skeleton[:, 4, :2] = head_pos[:, :2]  # 右耳
        coco_skeleton[:, 1:5, 2] = 0.5  # 設定較低置信度

    elif ntu_skeleton.ndim == 4:
        # 多人情況 (M, T, 25, 3)
        M, T, V, C = ntu_skeleton.shape
        coco_skeleton = np.zeros((M, T, 17, C), dtype=np.float32)

        for m in range(M):
            coco_skeleton[m] = ntu_skeleton_to_coco(ntu_skeleton[m])
    else:
        # 異常情況
        print(f"Warning: Unexpected skeleton ndim {ntu_skeleton.ndim}, returning zeros")
        return np.zeros((1, 17, 3), dtype=np.float32)

    return coco_skeleton


# =============================================================================
# 數據集
# =============================================================================

class NTUDataset(Dataset):
    """NTU RGB+D 數據集加載器"""

    def __init__(self, data_path, split='train', max_frames=300, benchmark='xsub'):
        """
        參數：
            data_path: NTU RGB+D 數據集根目錄
            split: 'train' 或 'val'
            max_frames: 最大幀數（用於填充/截斷）
            benchmark: 'xsub' (cross-subject) 或 'xview' (cross-view)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_frames = max_frames
        self.benchmark = benchmark

        # 載入樣本列表
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split ({benchmark})")

    def _load_samples(self):
        """載入樣本路徑和標籤"""
        samples = []

        # 尋找 .skeleton 檔案
        skeleton_files = list(self.data_path.glob("**/*.skeleton"))

        if len(skeleton_files) == 0:
            print(f"Warning: No .skeleton files found in {self.data_path}")
            print("Please extract the NTU RGB+D dataset first.")
            return samples

        for skeleton_file in skeleton_files:
            # 從檔名解析資訊
            # 格式：SsssCcccPpppRrrrAaaa.skeleton
            # S: setup number, C: camera ID, P: performer ID,
            # R: replication number, A: action class
            filename = skeleton_file.stem

            try:
                setup_id = int(filename[1:4])
                camera_id = int(filename[5:8])
                performer_id = int(filename[9:12])
                action_class = int(filename[17:20])
            except:
                print(f"Warning: Cannot parse filename {filename}, skipping")
                continue

            # 判斷是否為訓練集或測試集
            if self.benchmark == 'xsub':
                # Cross-subject: 訓練集是 performer 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38
                train_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
                is_train = performer_id in train_subjects
            elif self.benchmark == 'xview':
                # Cross-view: 訓練集是 camera 2, 3
                train_cameras = [2, 3]
                is_train = camera_id in train_cameras
            else:
                raise ValueError(f"Unknown benchmark: {self.benchmark}")

            if (self.split == 'train' and is_train) or (self.split == 'val' and not is_train):
                samples.append({
                    'path': skeleton_file,
                    'label': action_class - 1,  # 轉為 0-based 索引
                    'setup': setup_id,
                    'camera': camera_id,
                    'performer': performer_id
                })

        return samples

    def _read_skeleton_file(self, filepath):
        """讀取 .skeleton 檔案"""
        try:
            with open(filepath, 'r') as f:
                frame_count = int(f.readline())

                frames = []
                for _ in range(frame_count):
                    # 讀取人數
                    body_count = int(f.readline())

                    # 暫時只處理第一個人
                    frame_joints = None
                    for body_id in range(body_count):
                        # 讀取身體資訊
                        body_info = f.readline()  # bodyID, clipedEdges, ...

                        # 讀取關節數
                        joint_count = int(f.readline())

                        # 讀取關節資料
                        joints = []
                        for _ in range(joint_count):
                            joint_data = f.readline().strip().split()
                            if len(joint_data) >= 3:
                                x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                                joints.append([x, y, z])

                        if body_id == 0 and len(joints) == 25:  # 只使用第一個人且關節數正確
                            frame_joints = joints

                    # 如果此幀有有效的身體資料，添加到 frames
                    if frame_joints is not None:
                        frames.append(frame_joints)

                # 確保至少有一幀資料
                if len(frames) == 0:
                    # 返回一個零填充的幀
                    frames = [[[0.0, 0.0, 0.0] for _ in range(25)]]

                result = np.array(frames, dtype=np.float32)  # (T, 25, 3)

                # 確保形狀正確
                if result.ndim != 3 or result.shape[1] != 25 or result.shape[2] != 3:
                    # 如果形狀不對，返回單幀零數據
                    result = np.zeros((1, 25, 3), dtype=np.float32)

                return result

        except Exception as e:
            # 如果讀取失敗，返回單幀零數據
            print(f"Warning: Failed to read {filepath}: {e}")
            return np.zeros((1, 25, 3), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 讀取骨架資料
        skeleton = self._read_skeleton_file(sample['path'])  # (T, 25, 3)

        # 轉換為 COCO 格式
        skeleton = ntu_skeleton_to_coco(skeleton)  # (T, 17, 3)

        # 填充或截斷到固定長度
        T = skeleton.shape[0]
        if T > self.max_frames:
            # 均勻採樣
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            skeleton = skeleton[indices]
        else:
            # 零填充
            padding = np.zeros((self.max_frames - T, 17, 3), dtype=np.float32)
            skeleton = np.concatenate([skeleton, padding], axis=0)

        # 轉為 torch tensor
        skeleton = torch.from_numpy(skeleton)  # (T, 17, 3)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return skeleton, label


# =============================================================================
# 訓練
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for skeletons, labels in pbar:
        skeletons = skeletons.to(device)  # (N, T, 17, 3)
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


def main():
    parser = argparse.ArgumentParser(description='NTU RGB+D Pretraining')

    # 數據集參數
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NTU RGB+D dataset')
    parser.add_argument('--benchmark', type=str, default='xsub', choices=['xsub', 'xview'],
                        help='Benchmark protocol (default: xsub)')
    parser.add_argument('--num_classes', type=int, default=60,
                        help='Number of action classes (60 or 120)')

    # 模型參數
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel number in AGCN (default: 64)')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum number of frames (default: 300)')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # 其他參數
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available)')

    args = parser.parse_args()

    # 建立儲存目錄
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("NTU RGB+D Pretraining for One-Shot Action Recognition")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 80)

    # 建立數據集
    print("\nLoading datasets...")
    train_dataset = NTUDataset(
        args.data_path,
        split='train',
        max_frames=args.max_frames,
        benchmark=args.benchmark
    )

    val_dataset = NTUDataset(
        args.data_path,
        split='val',
        max_frames=args.max_frames,
        benchmark=args.benchmark
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
        pin_memory=True
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
    model = SkeletonEmbedding(
        in_channels=3,
        base_channels=args.base_channels,
        num_classes=args.num_classes
    )
    model = model.to(args.device)

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 學習率調度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 恢復訓練
    start_epoch = 0
    best_acc = 0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
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
            print(f"  ⭐ New best accuracy! Saved: {best_path}")

        # 每 10 個 epoch 儲存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(args.save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"  Saved epoch checkpoint: {epoch_path}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved in: {args.save_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
