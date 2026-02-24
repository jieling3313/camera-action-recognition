#!/usr/bin/env python3
"""
測試腳本：從靜態圖片提取骨架
使用 YOLOv8-Pose 模型從圖片中提取人體骨架
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from skeleton_extractor import SkeletonExtractor

def test_image(image_path, output_dir="output"):
    """
    從單張圖片提取並視覺化骨架

    Args:
        image_path: 輸入圖片路徑
        output_dir: 輸出目錄
    """
    print(f"\n{'='*60}")
    print(f"Testing image: {image_path}")
    print(f"{'='*60}")

    # 檢查圖片是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return False

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return False

    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # 初始化骨架提取器（使用 YOLOv8-Pose）
    print("\n[1/4] Initializing YOLOv8-Pose model...")
    extractor = SkeletonExtractor(model_path='yolov8m-pose.pt', conf_threshold=0.5)
    print("✓ Model loaded successfully")

    # 提取骨架
    print("\n[2/4] Extracting skeleton keypoints...")
    persons = extractor.extract_all_persons(image, normalize=False)

    if len(persons) == 0:
        print("✗ No person detected in the image")
        return False

    print(f"✓ Detected {len(persons)} person(s)")

    # 顯示每個人的骨架資訊
    keypoint_names = [
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
        'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
        'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip',
        'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
    ]

    for i, (skeleton, bbox) in enumerate(persons):
        print(f"\n  Person {i+1}:")
        print(f"    - Valid keypoints: {np.sum(skeleton[:, 2] > 0.3)}/17")
        if bbox is not None:
            print(f"    - Bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")

        print(f"    - Keypoints:")
        for j, name in enumerate(keypoint_names):
            x, y, c = skeleton[j]
            if c > 0.3:  # 只顯示可信度高的關鍵點
                print(f"      {name:12s}: ({x:6.1f}, {y:6.1f}) conf={c:.3f}")

    # 視覺化骨架
    print("\n[3/4] Visualizing skeleton...")
    vis_image = image.copy()
    for skeleton, bbox in persons:
        vis_image = extractor.draw_skeleton(vis_image, skeleton, bbox=bbox, normalized=False)
    print("✓ Visualization complete")

    # 儲存結果
    print("\n[4/4] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    # 生成輸出檔名
    input_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_skeleton.jpg")

    cv2.imwrite(output_path, vis_image)
    print(f"✓ Saved to: {output_path}")

    # 也儲存骨架資料（所有偵測到的人）
    if len(persons) > 0:
        skeleton_data_path = os.path.join(output_dir, f"{input_name}_skeleton.npy")
        # 只儲存關鍵點，不包含 bbox
        skeletons_only = [skeleton for skeleton, bbox in persons]
        np.save(skeleton_data_path, np.array(skeletons_only))
        print(f"✓ Skeleton data saved to: {skeleton_data_path} ({len(persons)} person(s))")

    print(f"\n{'='*60}")
    print("✓✓✓ SUCCESS! Skeleton extracted and saved ✓✓✓")
    print(f"{'='*60}\n")

    return True

def batch_test(image_dir, output_dir="output"):
    """
    批次處理目錄中的所有圖片

    Args:
        image_dir: 包含圖片的目錄
        output_dir: 輸出目錄
    """
    # 支援的圖片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 尋找所有圖片
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    if len(image_files) == 0:
        print(f"Error: No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    success_count = 0
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n\n{'#'*60}")
        print(f"# Processing image {i}/{len(image_files)}")
        print(f"{'#'*60}")

        if test_image(str(image_path), output_dir):
            success_count += 1

    print(f"\n\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(image_files)} images")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}\n")

def main():
    """主函數"""
    print("\n" + "="*60)
    print("YOLOv8-Pose Skeleton Extraction Test")
    print("="*60 + "\n")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python3 test_skeleton_from_images.py <image_path>")
        print("  Batch mode:   python3 test_skeleton_from_images.py <image_directory>")
        print("\nExample:")
        print("  python3 test_skeleton_from_images.py test_images/person.jpg")
        print("  python3 test_skeleton_from_images.py test_images/")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = "skeleton_output" if len(sys.argv) < 3 else sys.argv[2]

    # 判斷是檔案還是目錄
    if os.path.isfile(input_path):
        # 單張圖片模式
        test_image(input_path, output_dir)
    elif os.path.isdir(input_path):
        # 批次處理模式
        batch_test(input_path, output_dir)
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
