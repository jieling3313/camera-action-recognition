#!/usr/bin/env python3.10
"""
檢查 checkpoints 目錄中模型的資訊

互動式腳本，讓使用者選擇要查看哪個模型的詳細資訊
"""

import torch
import os
import glob
from skeleton_model import SkeletonEmbedding

def format_size(size_bytes):
    """格式化檔案大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_parameters(model):
    """計算模型參數數量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def list_checkpoints(checkpoint_dir):
    """列出所有可用的 checkpoints"""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return []

    print(f"\n{'='*80}")
    print("Available Checkpoints")
    print(f"{'='*80}")
    print(f"\n{'#':<5} {'Filename':<20} {'Size':<12}")
    print(f"{'-'*5} {'-'*20} {'-'*12}")

    for i, ckpt_path in enumerate(checkpoint_files, 1):
        filename = os.path.basename(ckpt_path)
        size = os.path.getsize(ckpt_path)
        print(f"{i:<5} {filename:<20} {format_size(size):<12}")

    return checkpoint_files

def inspect_checkpoint(checkpoint_path):
    """檢查單一 checkpoint 檔案"""
    print(f"\n{'='*80}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")

    # 檔案資訊
    file_size = os.path.getsize(checkpoint_path)
    print(f"\n--- File Information ---")
    print(f"Path: {checkpoint_path}")
    print(f"Size: {format_size(file_size)}")

    try:
        # 載入 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 訓練資訊
        print(f"\n--- Training Information ---")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint:
            print(f"Best Accuracy: {checkpoint['best_acc']:.4f}%")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.6f}")

        # 載入模型以檢查參數
        if 'model_state_dict' in checkpoint:
            print(f"\n--- Model Architecture ---")

            # 檢查是否有 classifier（用於判斷訓練時的類別數）
            state_dict = checkpoint['model_state_dict']
            num_classes = None
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
                print(f"Trained with classifier: {num_classes} classes")

            # 創建對應的模型
            model = SkeletonEmbedding(in_channels=3, base_channels=64, num_classes=num_classes)

            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                total_params, trainable_params = count_parameters(model)
                print(f"Total Parameters: {total_params:,}")
                print(f"Trainable Parameters: {trainable_params:,}")
                print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
                print(f"Model Memory (float32): {format_size(total_params * 4)}")
            except Exception as e:
                print(f"Warning: Could not load model state_dict: {e}")
        else:
            print(f"\n--- Model Architecture ---")
            print("No model_state_dict found in checkpoint")

        # 其他資訊
        print(f"\n--- Checkpoint Contents ---")
        for key in checkpoint.keys():
            value_type = type(checkpoint[key]).__name__
            if key == 'model_state_dict':
                print(f"  - {key}: {value_type} (state dict with {len(checkpoint[key])} entries)")
            elif key == 'optimizer_state_dict':
                print(f"  - {key}: {value_type} (optimizer state)")
            elif key == 'scheduler_state_dict':
                print(f"  - {key}: {value_type} (scheduler state)")
            else:
                print(f"  - {key}: {checkpoint[key]}")

        print(f"\n{'='*80}")
        return checkpoint

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_quick_summary(checkpoint_dir):
    """顯示所有 checkpoints 的簡要摘要"""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))

    if not checkpoint_files:
        return

    print(f"\n{'='*80}")
    print("Quick Summary (All Checkpoints)")
    print(f"{'='*80}")
    print(f"\n{'Filename':<20} {'Epoch':<8} {'Accuracy (%)':<15} {'Loss':<12}")
    print(f"{'-'*20} {'-'*8} {'-'*15} {'-'*12}")

    best_acc = 0
    best_file = None

    for ckpt_path in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            filename = os.path.basename(ckpt_path)
            epoch = checkpoint.get('epoch', 'N/A')
            accuracy = checkpoint.get('best_acc', 0.0)
            loss = checkpoint.get('loss', 0.0)

            print(f"{filename:<20} {str(epoch):<8} {accuracy:<15.4f} {loss:<12.6f}")

            if accuracy > best_acc:
                best_acc = accuracy
                best_file = filename

        except Exception as e:
            print(f"{os.path.basename(ckpt_path):<20} Error: {str(e)[:30]}")

    if best_file:
        print(f"\nBest Model: {best_file} ({best_acc:.4f}%)")

def main():
    """主函數"""
    checkpoint_dir = '/root/catkin_ws/src/yolo_ros/scripts/checkpoints'

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return

    while True:
        # 列出所有 checkpoints
        checkpoint_files = list_checkpoints(checkpoint_dir)

        if not checkpoint_files:
            break

        # 顯示選項
        print(f"\n{'='*80}")
        print("Options:")
        print(f"  1-{len(checkpoint_files)}: View detailed information for a specific checkpoint")
        print(f"  s: Show quick summary of all checkpoints")
        print(f"  q: Quit")
        print(f"{'='*80}")

        # 取得使用者輸入
        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q':
            print("Exiting...")
            break
        elif choice == 's':
            show_quick_summary(checkpoint_dir)
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoint_files):
                inspect_checkpoint(checkpoint_files[idx])
                input("\nPress Enter to continue...")
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(checkpoint_files)}")
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
