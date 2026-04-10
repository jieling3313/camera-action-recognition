#!/usr/bin/env python3
"""
測試 NTU RGB+D 數據集載入
快速驗證 train_ntu_rgbd.py 是否能正確讀取資料
"""

import sys
import os
from pathlib import Path

# 設定路徑
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from train_ntu_rgbd import NTUDataset, ntu_skeleton_to_coco
import torch

def test_dataset_loading(data_path):
    """測試數據集載入"""
    print("=" * 80)
    print("NTU RGB+D Dataset Loading Test")
    print("=" * 80)
    print(f"Dataset path: {data_path}\n")

    # 測試訓練集 (Cross-Subject)
    print("Testing training set (xsub)...")
    try:
        train_dataset = NTUDataset(
            data_path=data_path,
            split='train',
            max_frames=64,  # 使用較小的 frame 數以加快測試
            benchmark='xsub'
        )
        print(f"✓ Training set loaded: {len(train_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load training set: {e}")
        return False

    # 測試驗證集
    print("\nTesting validation set (xsub)...")
    try:
        val_dataset = NTUDataset(
            data_path=data_path,
            split='val',
            max_frames=64,
            benchmark='xsub'
        )
        print(f"✓ Validation set loaded: {len(val_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load validation set: {e}")
        return False

    if len(train_dataset) == 0:
        print("\n✗ Error: No samples in training set!")
        return False

    # 測試讀取單個樣本
    print("\n" + "-" * 80)
    print("Testing sample loading...")
    try:
        skeleton, label = train_dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - Skeleton shape: {skeleton.shape}")  # 應該是 (T, 17, 3)
        print(f"  - Label: {label.item()}")
        print(f"  - Skeleton dtype: {skeleton.dtype}")
        print(f"  - Label dtype: {label.dtype}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 測試 DataLoader
    print("\n" + "-" * 80)
    print("Testing DataLoader...")
    try:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 使用單線程避免複雜問題
        )

        # 載入一個 batch
        for batch_skeletons, batch_labels in train_loader:
            print(f"✓ DataLoader works")
            print(f"  - Batch skeleton shape: {batch_skeletons.shape}")  # (N, T, 17, 3)
            print(f"  - Batch labels shape: {batch_labels.shape}")  # (N,)
            print(f"  - Sample labels in batch: {batch_labels.tolist()}")
            break
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 測試模型推論
    print("\n" + "-" * 80)
    print("Testing model inference...")
    try:
        from skeleton_model import SkeletonEmbedding

        model = SkeletonEmbedding(
            in_channels=3,
            base_channels=64,
            num_classes=60
        )
        model.eval()

        with torch.no_grad():
            outputs = model(batch_skeletons)  # (N, num_classes)

        print(f"✓ Model inference successful")
        print(f"  - Output shape: {outputs.shape}")  # (N, 60)
        print(f"  - Predicted classes: {outputs.argmax(dim=1).tolist()}")
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nDataset Summary:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Total: {len(train_dataset) + len(val_dataset)}")
    print(f"  - Benchmark: xsub (Cross-Subject)")
    print(f"  - Action classes: 60")
    print("\n✓ Ready to start training!")
    print("\nTo start training, run:")
    print(f"  python3 train_ntu_rgbd.py --data_path {data_path} --epochs 5 --batch_size 8")
    print("=" * 80)

    return True


if __name__ == '__main__':
    # 設定數據集路徑
    data_path = "/root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"

    # 如果從命令行提供路徑
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    # 檢查路徑是否存在
    if not os.path.exists(data_path):
        print(f"Error: Dataset path does not exist: {data_path}")
        print("\nPlease provide the correct path:")
        print(f"  python3 {sys.argv[0]} /path/to/nturgbd_skeletons")
        sys.exit(1)

    # 執行測試
    success = test_dataset_loading(data_path)

    sys.exit(0 if success else 1)
