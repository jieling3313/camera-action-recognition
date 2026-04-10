#!/usr/bin/env python3
"""
NTU RGB+D 影片 MediaPipe 骨架提取腳本

此腳本從 NTU RGB+D 資料集的原始 RGB 影片中提取 MediaPipe 33 點骨架序列，
用於訓練 MediaPipe 33 點版本的 One-Shot Action Recognition 模型。

使用方式:
    python extract_mediapipe_from_ntu.py --video_dir /path/to/ntu_rgb_videos \
                                          --output_dir /path/to/output \
                                          --num_workers 8

NTU RGB+D 資料集結構:
    ntu_rgb_videos/
    ├── nturgb+d_rgb/
    │   ├── S001C001P001R001A001_rgb.avi
    │   ├── S001C001P001R001A002_rgb.avi
    │   └── ...

輸出結構:
    output/
    ├── mediapipe_skeletons/
    │   ├── S001C001P001R001A001.npy  # shape: (T, 33, 4) - x, y, z, visibility
    │   ├── S001C001P001R001A001.json # JSON 格式骨架資料
    │   ├── S001C001P001R001A002.npy
    │   ├── S001C001P001R001A002.json
    │   └── ...
    └── extraction_log.json

Author: Claude AI Assistant
Date: 2026-03-23
Updated: 2026-03-28 - 加入 JSON 輸出格式支援
"""

import os
import sys
import argparse
import json
import time
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
import traceback
import signal

import numpy as np
import cv2

# 嘗試匯入 MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("錯誤：請安裝 mediapipe: pip install mediapipe")


# =============================================================================
# MediaPipe 33 關節名稱
# =============================================================================
MEDIAPIPE_JOINT_NAMES = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]


# =============================================================================
# MediaPipe 骨架提取器
# =============================================================================

class MediaPipePoseExtractor:
    """
    使用 MediaPipe Pose 從影片中提取 33 點骨架。
    """

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        """
        初始化 MediaPipe Pose 提取器。

        參數:
            model_path: MediaPipe pose_landmarker.task 模型路徑
            use_gpu: 是否使用 GPU 加速
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe 未安裝")

        # 預設模型路徑
        if model_path is None:
            # 嘗試多個可能的路徑
            possible_paths = [
                '/root/.mediapipe/models/pose_landmarker.task',
                os.path.expanduser('~/.mediapipe/models/pose_landmarker.task'),
                './pose_landmarker.task',
                '/tmp/pose_landmarker.task',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                print("警告：找不到 MediaPipe 模型，將嘗試下載...")
                model_path = self._download_model()

        self.model_path = model_path
        self.use_gpu = use_gpu
        self.detector = None

    def _download_model(self) -> str:
        """下載 MediaPipe Pose Landmarker 模型。"""
        import urllib.request

        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        save_path = "/tmp/pose_landmarker.task"

        print(f"正在下載 MediaPipe 模型...")
        urllib.request.urlretrieve(model_url, save_path)
        print(f"模型已下載至: {save_path}")

        return save_path

    def _create_detector(self):
        """創建 MediaPipe Pose Landmarker。"""
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1,  # 只追蹤一個人
            output_segmentation_masks=False
        )

        return vision.PoseLandmarker.create_from_options(options)

    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        從影片中提取骨架序列。

        參數:
            video_path: 影片檔案路徑

        回傳:
            骨架序列 (T, 33, 4) 其中 4 = (x, y, z, visibility)
            若提取失敗則回傳 None
        """
        if not os.path.exists(video_path):
            print(f"錯誤：影片不存在: {video_path}")
            return None

        # 創建檢測器（每個影片創建新的實例以避免狀態問題）
        detector = self._create_detector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"錯誤：無法開啟影片: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 預設 FPS

        frame_idx = 0
        skeletons = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 轉換 BGR 到 RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 創建 MediaPipe Image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # 計算時間戳（毫秒）
                timestamp_ms = int(frame_idx * 1000 / fps)

                # 執行姿勢檢測
                result = detector.detect_for_video(mp_image, timestamp_ms)

                # 提取骨架
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    skeleton = np.zeros((33, 4), dtype=np.float32)

                    for i, lm in enumerate(landmarks):
                        skeleton[i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]

                    skeletons.append(skeleton)
                else:
                    # 若此幀未檢測到人，使用零值
                    skeletons.append(np.zeros((33, 4), dtype=np.float32))

                frame_idx += 1

        except Exception as e:
            print(f"處理影片時發生錯誤 {video_path}: {e}")
            traceback.print_exc()
            return None

        finally:
            cap.release()
            detector.close()

        if len(skeletons) == 0:
            return None

        return np.array(skeletons, dtype=np.float32)


# =============================================================================
# 輸出格式轉換
# =============================================================================

def skeleton_to_json(skeleton: np.ndarray, video_info: Dict) -> Dict:
    """
    將 numpy 骨架資料轉換為 JSON 格式。

    參數:
        skeleton: (T, 33, 4) 骨架序列
        video_info: 影片資訊字典 (setup, camera, performer, replication, action)

    回傳:
        JSON 可序列化的字典
    """
    num_frames = skeleton.shape[0]

    json_data = {
        "format": "mediapipe_33",
        "num_joints": 33,
        "joint_names": MEDIAPIPE_JOINT_NAMES,
        "channels": ["x", "y", "z", "visibility"],
        "video_info": video_info,
        "num_frames": num_frames,
        "frames": []
    }

    for frame_idx in range(num_frames):
        frame_data = {
            "frame_index": frame_idx,
            "num_bodies": 1,
            "bodies": [
                {
                    "body_id": 0,
                    "joints": []
                }
            ]
        }

        # 檢查此幀是否有有效的骨架（非全零）
        frame_skeleton = skeleton[frame_idx]
        has_valid_skeleton = np.any(frame_skeleton != 0)

        if has_valid_skeleton:
            for joint_idx in range(33):
                joint = frame_skeleton[joint_idx]
                frame_data["bodies"][0]["joints"].append({
                    "index": joint_idx,
                    "name": MEDIAPIPE_JOINT_NAMES[joint_idx],
                    "x": float(joint[0]),
                    "y": float(joint[1]),
                    "z": float(joint[2]),
                    "visibility": float(joint[3])
                })
        else:
            # 空幀
            frame_data["bodies"] = []

        json_data["frames"].append(frame_data)

    return json_data


# =============================================================================
# 批量處理函數
# =============================================================================

def process_single_video(args: Tuple[str, str, str, bool, bool]) -> dict:
    """
    處理單一影片（用於多進程）。

    參數:
        args: (video_path, output_base_path, model_path, save_npy, save_json)

    回傳:
        處理結果字典
    """
    video_path, output_base_path, model_path, save_npy, save_json = args

    result = {
        'video': os.path.basename(video_path),
        'success': False,
        'frames': 0,
        'error': None,
        'time': 0
    }

    start_time = time.time()

    try:
        # 解析影片資訊
        video_info = extract_action_info(video_path)
        if video_info is None:
            result['error'] = "Cannot parse video filename"
            result['time'] = time.time() - start_time
            return result

        extractor = MediaPipePoseExtractor(model_path=model_path)
        skeleton = extractor.extract_from_video(video_path)

        if skeleton is not None and len(skeleton) > 0:
            # 儲存骨架
            if save_npy:
                npy_path = output_base_path + ".npy"
                np.save(npy_path, skeleton)

            if save_json:
                json_path = output_base_path + ".json"
                json_data = skeleton_to_json(skeleton, video_info)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False)

            result['success'] = True
            result['frames'] = len(skeleton)
        else:
            result['error'] = "No skeleton extracted"

    except Exception as e:
        result['error'] = str(e)

    result['time'] = time.time() - start_time
    return result


def extract_action_info(filename: str) -> Optional[Dict]:
    """
    從 NTU RGB+D 檔名解析動作資訊。

    檔名格式: SsssCcccPpppRrrrAaaa_rgb.avi
    - S: setup number (1-17 for NTU 60, 1-32 for NTU 120)
    - C: camera ID (1-3)
    - P: performer ID (1-40 for NTU 60, 1-106 for NTU 120)
    - R: replication number (1-2)
    - A: action class (1-60 for NTU 60, 1-120 for NTU 120)
    """
    basename = os.path.basename(filename).replace('_rgb.avi', '').replace('.avi', '').replace('.npy', '').replace('.json', '')

    try:
        return {
            'setup': int(basename[1:4]),
            'camera': int(basename[5:8]),
            'performer': int(basename[9:12]),
            'replication': int(basename[13:16]),
            'action': int(basename[17:20]),
            'filename': basename
        }
    except (ValueError, IndexError):
        return None


def get_split_info(info: dict, benchmark: str = 'xsub') -> str:
    """
    判斷樣本屬於訓練集還是測試集。

    參數:
        info: 從 extract_action_info 取得的資訊字典
        benchmark: 'xsub' (cross-subject) 或 'xview' (cross-view)

    回傳:
        'train' 或 'val'
    """
    if benchmark == 'xsub':
        # Cross-subject: 訓練集是特定的 performer IDs
        train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        return 'train' if info['performer'] in train_subjects else 'val'
    elif benchmark == 'xview':
        # Cross-view: 訓練集是 camera 2, 3
        return 'train' if info['camera'] in [2, 3] else 'val'
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# =============================================================================
# 主程式
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='從 NTU RGB+D 影片提取 MediaPipe 33 點骨架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 基本使用（輸出 .npy 和 .json）
    python extract_mediapipe_from_ntu.py \\
        --video_dir "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb" \\
        --output_dir "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe" \\
        --num_workers 8

    # 只輸出 .npy
    python extract_mediapipe_from_ntu.py \\
        --video_dir /data/ntu_rgb_videos \\
        --output_dir /data/mediapipe_skeletons \\
        --output_format npy \\
        --num_workers 8

    # 只處理特定動作類別
    python extract_mediapipe_from_ntu.py \\
        --video_dir /data/ntu_rgb_videos \\
        --output_dir /data/mediapipe_skeletons \\
        --action_range 1 60

    # 繼續中斷的任務（跳過已存在的檔案）
    python extract_mediapipe_from_ntu.py \\
        --video_dir /data/ntu_rgb_videos \\
        --output_dir /data/mediapipe_skeletons \\
        --skip_existing \\
        --num_workers 8
        """
    )

    parser.add_argument('--video_dir', type=str, required=True,
                        help='NTU RGB+D 影片目錄路徑')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出目錄路徑')
    parser.add_argument('--model_path', type=str, default=None,
                        help='MediaPipe pose_landmarker.task 模型路徑')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='並行處理的工作程序數量 (default: 1)')
    parser.add_argument('--action_range', type=int, nargs=2, default=None,
                        help='只處理指定範圍的動作類別 (e.g., 1 60)')
    parser.add_argument('--output_format', type=str, default='both',
                        choices=['npy', 'json', 'both'],
                        help='輸出格式：npy, json, 或 both (default: both)')
    parser.add_argument('--benchmark', type=str, default='xsub',
                        choices=['xsub', 'xview'],
                        help='Benchmark 協議用於分割訓練/測試集')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳過已存在的輸出檔案')
    parser.add_argument('--dry_run', action='store_true',
                        help='只列出要處理的檔案，不實際處理')

    args = parser.parse_args()

    # 確定輸出格式
    save_npy = args.output_format in ['npy', 'both']
    save_json = args.output_format in ['json', 'both']

    # 建立輸出目錄
    output_skeleton_dir = os.path.join(args.output_dir, 'mediapipe_skeletons')
    os.makedirs(output_skeleton_dir, exist_ok=True)

    # 尋找所有影片檔案
    print("=" * 70)
    print("NTU RGB+D MediaPipe 骨架提取工具")
    print("=" * 70)
    print(f"\n影片目錄: {args.video_dir}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"輸出格式: {args.output_format}")
    print(f"工作程序: {args.num_workers}")

    video_patterns = [
        os.path.join(args.video_dir, '**', '*_rgb.avi'),
        os.path.join(args.video_dir, '**', '*.avi'),
    ]

    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(pattern, recursive=True))

    # 去重複
    video_files = list(set(video_files))
    # 排序以確保一致性
    video_files.sort()

    print(f"\n找到 {len(video_files)} 個影片檔案")

    if len(video_files) == 0:
        print("錯誤：找不到任何影片檔案")
        print(f"請確認影片目錄路徑正確: {args.video_dir}")
        sys.exit(1)

    # 過濾影片
    tasks = []
    skipped = 0
    filtered = 0

    for video_path in video_files:
        info = extract_action_info(video_path)

        if info is None:
            print(f"警告：無法解析檔名: {video_path}")
            continue

        # 檢查動作範圍
        if args.action_range:
            action_min, action_max = args.action_range
            if not (action_min <= info['action'] <= action_max):
                filtered += 1
                continue

        # 輸出路徑（不含副檔名）
        output_base_path = os.path.join(output_skeleton_dir, info['filename'])

        # 檢查是否已存在
        if args.skip_existing:
            npy_exists = os.path.exists(output_base_path + ".npy")
            json_exists = os.path.exists(output_base_path + ".json")

            if save_npy and save_json:
                if npy_exists and json_exists:
                    skipped += 1
                    continue
            elif save_npy and npy_exists:
                skipped += 1
                continue
            elif save_json and json_exists:
                skipped += 1
                continue

        tasks.append((video_path, output_base_path, args.model_path, save_npy, save_json))

    print(f"過濾後: {len(tasks)} 個影片待處理")
    print(f"跳過已存在: {skipped} 個")
    print(f"範圍過濾: {filtered} 個")

    if args.dry_run:
        print("\n[Dry Run] 將處理以下檔案:")
        for video_path, output_base_path, _, _, _ in tasks[:10]:
            print(f"  {os.path.basename(video_path)} -> {os.path.basename(output_base_path)}.*")
        if len(tasks) > 10:
            print(f"  ... 還有 {len(tasks) - 10} 個檔案")
        return

    if len(tasks) == 0:
        print("沒有需要處理的檔案")
        return

    # 開始處理
    print(f"\n開始處理 {len(tasks)} 個影片...")
    print("提示：按 Ctrl+C 可安全中斷，使用 --skip_existing 繼續")
    start_time = time.time()

    results = {
        'success': 0,
        'failed': 0,
        'total_frames': 0,
        'errors': [],
        'details': []
    }

    def save_progress():
        """儲存當前進度"""
        elapsed_time = time.time() - start_time
        log_path = os.path.join(args.output_dir, 'extraction_log.json')
        log_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'interrupted' if results['success'] + results['failed'] < len(tasks) else 'completed',
            'config': {
                'video_dir': args.video_dir,
                'output_dir': args.output_dir,
                'num_workers': args.num_workers,
                'action_range': args.action_range,
                'output_format': args.output_format,
                'benchmark': args.benchmark
            },
            'results': {
                'total_tasks': len(tasks),
                'processed': results['success'] + results['failed'],
                'success': results['success'],
                'failed': results['failed'],
                'total_frames': results['total_frames'],
                'elapsed_time': elapsed_time
            },
            'errors': results['errors'][:100]  # 只保存前100個錯誤
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        return log_path

    try:
        if args.num_workers > 1:
            # 多進程處理
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(process_single_video, task): task for task in tasks}

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results['details'].append(result)

                    if result['success']:
                        results['success'] += 1
                        results['total_frames'] += result['frames']
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'video': result['video'],
                            'error': result['error']
                        })

                    # 進度顯示
                    progress = (i + 1) / len(tasks) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(tasks) - i - 1)
                    print(f"\r進度: {i + 1}/{len(tasks)} ({progress:.1f}%) - "
                          f"成功: {results['success']}, 失敗: {results['failed']} - "
                          f"ETA: {eta/60:.1f} 分鐘",
                          end='', flush=True)

                    # 每 1000 個檔案儲存一次進度
                    if (i + 1) % 1000 == 0:
                        save_progress()
        else:
            # 單進程處理
            for i, task in enumerate(tasks):
                result = process_single_video(task)
                results['details'].append(result)

                if result['success']:
                    results['success'] += 1
                    results['total_frames'] += result['frames']
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'video': result['video'],
                        'error': result['error']
                    })

                # 進度顯示
                progress = (i + 1) / len(tasks) * 100
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(tasks) - i - 1)
                print(f"\r進度: {i + 1}/{len(tasks)} ({progress:.1f}%) - "
                      f"成功: {results['success']}, 失敗: {results['failed']} - "
                      f"ETA: {eta/60:.1f} 分鐘",
                      end='', flush=True)

                # 每 1000 個檔案儲存一次進度
                if (i + 1) % 1000 == 0:
                    save_progress()

    except KeyboardInterrupt:
        print("\n\n收到中斷信號，正在儲存進度...")
        log_path = save_progress()
        print(f"進度已儲存至: {log_path}")
        print(f"使用 --skip_existing 參數繼續未完成的任務")
        sys.exit(0)

    print()  # 換行

    elapsed_time = time.time() - start_time

    # 儲存日誌
    log_path = save_progress()

    # 顯示摘要
    print("\n" + "=" * 70)
    print("處理完成!")
    print("=" * 70)
    print(f"\n總計處理: {len(tasks)} 個影片")
    print(f"成功: {results['success']} ({results['success']/len(tasks)*100:.1f}%)")
    print(f"失敗: {results['failed']} ({results['failed']/len(tasks)*100:.1f}%)")
    print(f"總幀數: {results['total_frames']:,}")
    print(f"耗時: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分鐘, {elapsed_time/3600:.2f} 小時)")
    if elapsed_time > 0:
        print(f"平均速度: {len(tasks)/elapsed_time:.2f} 影片/秒")
    print(f"\n輸出目錄: {output_skeleton_dir}")
    print(f"日誌檔案: {log_path}")

    if results['failed'] > 0:
        print(f"\n前 5 個錯誤:")
        for err in results['errors'][:5]:
            print(f"  - {err['video']}: {err['error']}")


# =============================================================================
# 額外工具函數
# =============================================================================

def verify_output(output_dir: str):
    """驗證輸出的骨架檔案。"""
    skeleton_files = glob.glob(os.path.join(output_dir, 'mediapipe_skeletons', '*.npy'))

    print(f"\n驗證 {len(skeleton_files)} 個骨架檔案...")

    valid = 0
    invalid = []

    for f in skeleton_files:
        try:
            data = np.load(f)
            if data.shape[1:] == (33, 4):
                valid += 1
            else:
                invalid.append((f, f"Wrong shape: {data.shape}"))
        except Exception as e:
            invalid.append((f, str(e)))

    print(f"有效: {valid}")
    print(f"無效: {len(invalid)}")

    if invalid:
        print("\n無效檔案:")
        for f, err in invalid[:5]:
            print(f"  - {os.path.basename(f)}: {err}")


def create_label_file(skeleton_dir: str, output_path: str, benchmark: str = 'xsub'):
    """
    建立標籤檔案，用於訓練。

    輸出格式: JSON
    {
        "train": [
            {"filename": "S001C001P001R001A001.npy", "action": 0, "setup": 1, ...},
            ...
        ],
        "val": [...]
    }
    """
    skeleton_files = glob.glob(os.path.join(skeleton_dir, '*.npy'))

    data = {'train': [], 'val': []}

    for f in skeleton_files:
        info = extract_action_info(f)
        if info is None:
            continue

        split = get_split_info(info, benchmark)
        info['label'] = info['action'] - 1  # 轉為 0-based index

        data[split].append(info)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"標籤檔案已建立: {output_path}")
    print(f"  訓練集: {len(data['train'])} 樣本")
    print(f"  驗證集: {len(data['val'])} 樣本")


if __name__ == '__main__':
    main()
