#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Azure Kinect DK 骨架提取器

從 Azure Kinect DK 深度感測器提取 32 個關節點，
並支援轉換為 NTU RGB+D 25 點格式用於動作辨識。

Azure Kinect Body Tracking SDK 關節定義：
https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, NamedTuple
from enum import IntEnum


class AzureKinectJoint(IntEnum):
    """Azure Kinect DK 32 個關節索引"""
    PELVIS = 0
    SPINE_NAVEL = 1
    SPINE_CHEST = 2
    NECK = 3
    CLAVICLE_LEFT = 4
    SHOULDER_LEFT = 5
    ELBOW_LEFT = 6
    WRIST_LEFT = 7
    HAND_LEFT = 8
    HANDTIP_LEFT = 9
    THUMB_LEFT = 10
    CLAVICLE_RIGHT = 11
    SHOULDER_RIGHT = 12
    ELBOW_RIGHT = 13
    WRIST_RIGHT = 14
    HAND_RIGHT = 15
    HANDTIP_RIGHT = 16
    THUMB_RIGHT = 17
    HIP_LEFT = 18
    KNEE_LEFT = 19
    ANKLE_LEFT = 20
    FOOT_LEFT = 21
    HIP_RIGHT = 22
    KNEE_RIGHT = 23
    ANKLE_RIGHT = 24
    FOOT_RIGHT = 25
    HEAD = 26
    NOSE = 27
    EYE_LEFT = 28
    EAR_LEFT = 29
    EYE_RIGHT = 30
    EAR_RIGHT = 31


class NTUJoint(IntEnum):
    """NTU RGB+D (Kinect v2) 25 個關節索引"""
    SPINE_BASE = 0
    SPINE_MID = 1
    NECK = 2
    HEAD = 3
    SHOULDER_LEFT = 4
    ELBOW_LEFT = 5
    WRIST_LEFT = 6
    HAND_LEFT = 7
    SHOULDER_RIGHT = 8
    ELBOW_RIGHT = 9
    WRIST_RIGHT = 10
    HAND_RIGHT = 11
    HIP_LEFT = 12
    KNEE_LEFT = 13
    ANKLE_LEFT = 14
    FOOT_LEFT = 15
    HIP_RIGHT = 16
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18
    FOOT_RIGHT = 19
    SPINE = 20
    HAND_TIP_LEFT = 21
    THUMB_LEFT = 22
    HAND_TIP_RIGHT = 23
    THUMB_RIGHT = 24


# Azure Kinect 32 點 → NTU 25 點 映射表
AZURE_TO_NTU_MAP = {
    # NTU idx: Azure idx
    NTUJoint.SPINE_BASE: AzureKinectJoint.PELVIS,
    NTUJoint.SPINE_MID: AzureKinectJoint.SPINE_NAVEL,
    NTUJoint.NECK: AzureKinectJoint.NECK,
    NTUJoint.HEAD: AzureKinectJoint.HEAD,
    NTUJoint.SHOULDER_LEFT: AzureKinectJoint.SHOULDER_LEFT,
    NTUJoint.ELBOW_LEFT: AzureKinectJoint.ELBOW_LEFT,
    NTUJoint.WRIST_LEFT: AzureKinectJoint.WRIST_LEFT,
    NTUJoint.HAND_LEFT: AzureKinectJoint.HAND_LEFT,
    NTUJoint.SHOULDER_RIGHT: AzureKinectJoint.SHOULDER_RIGHT,
    NTUJoint.ELBOW_RIGHT: AzureKinectJoint.ELBOW_RIGHT,
    NTUJoint.WRIST_RIGHT: AzureKinectJoint.WRIST_RIGHT,
    NTUJoint.HAND_RIGHT: AzureKinectJoint.HAND_RIGHT,
    NTUJoint.HIP_LEFT: AzureKinectJoint.HIP_LEFT,
    NTUJoint.KNEE_LEFT: AzureKinectJoint.KNEE_LEFT,
    NTUJoint.ANKLE_LEFT: AzureKinectJoint.ANKLE_LEFT,
    NTUJoint.FOOT_LEFT: AzureKinectJoint.FOOT_LEFT,
    NTUJoint.HIP_RIGHT: AzureKinectJoint.HIP_RIGHT,
    NTUJoint.KNEE_RIGHT: AzureKinectJoint.KNEE_RIGHT,
    NTUJoint.ANKLE_RIGHT: AzureKinectJoint.ANKLE_RIGHT,
    NTUJoint.FOOT_RIGHT: AzureKinectJoint.FOOT_RIGHT,
    NTUJoint.SPINE: AzureKinectJoint.SPINE_CHEST,
    NTUJoint.HAND_TIP_LEFT: AzureKinectJoint.HANDTIP_LEFT,
    NTUJoint.THUMB_LEFT: AzureKinectJoint.THUMB_LEFT,
    NTUJoint.HAND_TIP_RIGHT: AzureKinectJoint.HANDTIP_RIGHT,
    NTUJoint.THUMB_RIGHT: AzureKinectJoint.THUMB_RIGHT,
}


class AzureKinectSkeletonExtractor:
    """
    Azure Kinect DK 骨架提取器

    使用 Azure Kinect Body Tracking SDK 提取 32 個關節點，
    並可轉換為 NTU RGB+D 相容的 25 點格式。
    """

    # Azure Kinect 關節名稱
    JOINT_NAMES = [
        'pelvis', 'spine_navel', 'spine_chest', 'neck',
        'clavicle_left', 'shoulder_left', 'elbow_left', 'wrist_left',
        'hand_left', 'handtip_left', 'thumb_left',
        'clavicle_right', 'shoulder_right', 'elbow_right', 'wrist_right',
        'hand_right', 'handtip_right', 'thumb_right',
        'hip_left', 'knee_left', 'ankle_left', 'foot_left',
        'hip_right', 'knee_right', 'ankle_right', 'foot_right',
        'head', 'nose', 'eye_left', 'ear_left', 'eye_right', 'ear_right'
    ]

    # NTU 關節名稱
    NTU_JOINT_NAMES = [
        'spine_base', 'spine_mid', 'neck', 'head',
        'shoulder_left', 'elbow_left', 'wrist_left', 'hand_left',
        'shoulder_right', 'elbow_right', 'wrist_right', 'hand_right',
        'hip_left', 'knee_left', 'ankle_left', 'foot_left',
        'hip_right', 'knee_right', 'ankle_right', 'foot_right',
        'spine', 'hand_tip_left', 'thumb_left', 'hand_tip_right', 'thumb_right'
    ]

    # Azure Kinect 骨架連接（用於視覺化）
    SKELETON_CONNECTIONS_32 = [
        # 脊椎
        (0, 1), (1, 2), (2, 3), (3, 26),
        # 左臂
        (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
        # 右臂
        (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),
        # 左腿
        (0, 18), (18, 19), (19, 20), (20, 21),
        # 右腿
        (0, 22), (22, 23), (23, 24), (24, 25),
        # 臉部
        (26, 27), (26, 28), (28, 29), (26, 30), (30, 31),
    ]

    # NTU 25 點骨架連接
    SKELETON_CONNECTIONS_25 = [
        # 脊椎
        (0, 1), (1, 20), (20, 2), (2, 3),
        # 左臂
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        # 右臂
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        # 左腿
        (0, 12), (12, 13), (13, 14), (14, 15),
        # 右腿
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]

    def __init__(self, device_index: int = 0, output_format: str = 'ntu25'):
        """
        初始化 Azure Kinect 骨架提取器

        參數：
            device_index: Azure Kinect 設備索引
            output_format: 輸出格式 'azure32' 或 'ntu25'
        """
        self.device_index = device_index
        self.output_format = output_format
        self.device = None
        self.tracker = None
        self._initialized = False

        # 根據輸出格式設定關節數量
        if output_format == 'ntu25':
            self.num_joints = 25
        else:
            self.num_joints = 32

    def initialize(self) -> bool:
        """
        初始化 Azure Kinect 設備和 Body Tracking

        回傳：
            是否成功初始化
        """
        try:
            import pyk4a
            from pyk4a import PyK4A, Config, ColorResolution, DepthMode

            # 設定設備配置
            config = Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )

            # 開啟設備
            self.device = PyK4A(config, device_id=self.device_index)
            self.device.start()

            # 初始化 Body Tracking
            from pyk4a import PyK4ABodyTracker
            self.tracker = PyK4ABodyTracker()

            self._initialized = True
            print(f"[AzureKinectExtractor] 設備初始化成功，輸出格式: {self.output_format}")
            return True

        except ImportError as e:
            print(f"[AzureKinectExtractor] 缺少必要的套件: {e}")
            print("請安裝: pip install pyk4a")
            return False
        except Exception as e:
            print(f"[AzureKinectExtractor] 初始化失敗: {e}")
            return False

    def extract(self, capture=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        提取骨架關節點

        參數：
            capture: 可選的 pyk4a capture 物件，若為 None 則自動擷取

        回傳：
            (joints, color_image, depth_image)
            joints: 形狀為 (N, 4) 的陣列，N=25 或 32，包含 (x, y, z, confidence)
            color_image: BGR 彩色影像
            depth_image: 深度影像
        """
        if not self._initialized:
            if not self.initialize():
                return None, None, None

        try:
            # 擷取影像
            if capture is None:
                capture = self.device.get_capture()

            if capture is None:
                return None, None, None

            # 取得彩色和深度影像
            color_image = capture.color
            depth_image = capture.transformed_depth

            # 執行 Body Tracking
            self.tracker.enqueue_capture(capture)
            body_frame = self.tracker.pop_result()

            if body_frame is None or body_frame.num_bodies == 0:
                return None, color_image, depth_image

            # 取得第一個人的骨架（可擴展為多人）
            skeleton = body_frame.get_body(0).numpy()

            # skeleton 形狀: (32, 4) - (x, y, z, confidence)
            joints_32 = skeleton

            # 根據輸出格式轉換
            if self.output_format == 'ntu25':
                joints = self._convert_to_ntu25(joints_32)
            else:
                joints = joints_32

            return joints, color_image, depth_image

        except Exception as e:
            print(f"[AzureKinectExtractor] 提取失敗: {e}")
            return None, None, None

    def _convert_to_ntu25(self, joints_32: np.ndarray) -> np.ndarray:
        """
        將 Azure Kinect 32 點轉換為 NTU 25 點格式

        參數：
            joints_32: 形狀為 (32, 4) 的陣列

        回傳：
            形狀為 (25, 4) 的陣列
        """
        joints_25 = np.zeros((25, 4), dtype=np.float32)

        for ntu_idx, azure_idx in AZURE_TO_NTU_MAP.items():
            joints_25[ntu_idx] = joints_32[azure_idx]

        return joints_25

    def extract_normalized(self, capture=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        提取正規化的 2D 骨架座標（用於與其他提取器相容）

        回傳：
            (keypoints, bbox)
            keypoints: 形狀為 (N, 3) 的陣列，包含 (x, y, confidence)，座標正規化至 [0, 1]
            bbox: 邊界框 [x1, y1, x2, y2]
        """
        joints, color_image, _ = self.extract(capture)

        if joints is None or color_image is None:
            return None, None

        h, w = color_image.shape[:2]

        # 將 3D 座標投影到 2D 並正規化
        keypoints = np.zeros((self.num_joints, 3), dtype=np.float32)

        for i, joint in enumerate(joints):
            x, y, z, conf = joint
            # 正規化座標
            keypoints[i, 0] = x / w if w > 0 else 0
            keypoints[i, 1] = y / h if h > 0 else 0
            keypoints[i, 2] = conf

        # 計算邊界框
        valid_points = keypoints[keypoints[:, 2] > 0.5]
        if len(valid_points) > 0:
            x_min = valid_points[:, 0].min() * w
            x_max = valid_points[:, 0].max() * w
            y_min = valid_points[:, 1].min() * h
            y_max = valid_points[:, 1].max() * h
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            bbox = None

        return keypoints, bbox

    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray,
                      bbox: Optional[np.ndarray] = None,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2,
                      normalized: bool = True) -> np.ndarray:
        """
        在影像上繪製骨架

        參數：
            image: BGR 影像
            keypoints: 形狀為 (N, 3) 的陣列
            bbox: 可選的邊界框
            color: 繪製顏色
            thickness: 線條粗細
            normalized: 座標是否已正規化

        回傳：
            標註後的影像
        """
        img = image.copy()
        h, w = img.shape[:2]

        kpts = keypoints.copy()
        if normalized:
            kpts[:, 0] *= w
            kpts[:, 1] *= h

        # 選擇對應的骨架連接
        if self.num_joints == 25:
            connections = self.SKELETON_CONNECTIONS_25
        else:
            connections = self.SKELETON_CONNECTIONS_32

        # 繪製關節點
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.3:
                cv2.circle(img, (int(x), int(y)), 4, color, -1)
                # 標註關節編號（可選）
                # cv2.putText(img, str(i), (int(x)+5, int(y)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 繪製骨架連接
        for start, end in connections:
            if start < len(kpts) and end < len(kpts):
                if kpts[start, 2] > 0.3 and kpts[end, 2] > 0.3:
                    pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
                    pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
                    cv2.line(img, pt1, pt2, color, thickness)

        # 繪製邊界框
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        return img

    def close(self):
        """關閉設備"""
        if self.tracker is not None:
            self.tracker = None
        if self.device is not None:
            self.device.stop()
            self.device = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AzureKinectSkeletonBuffer:
    """
    用於累積時間序列骨架資料的緩衝區（Azure Kinect 版本）
    """

    def __init__(self, buffer_size: int = 64, num_joints: int = 25):
        """
        初始化骨架緩衝區

        參數：
            buffer_size: 緩衝的幀數
            num_joints: 每幀的關節數量（25 或 32）
        """
        self.buffer_size = buffer_size
        self.num_joints = num_joints
        self.buffer = []

    def add(self, keypoints: Optional[np.ndarray]):
        """將關節點加入緩衝區"""
        if keypoints is not None:
            self.buffer.append(keypoints.copy())
        else:
            # 未偵測到時加入零值
            self.buffer.append(np.zeros((self.num_joints, 3), dtype=np.float32))

        # 維持緩衝區大小
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def is_full(self) -> bool:
        """檢查緩衝區是否已滿"""
        return len(self.buffer) >= self.buffer_size

    def get_sequence(self) -> Optional[np.ndarray]:
        """取得緩衝的序列，形狀為 (T, N, C)"""
        if len(self.buffer) == 0:
            return None
        return np.array(self.buffer, dtype=np.float32)

    def get_padded_sequence(self) -> np.ndarray:
        """取得填充至 buffer_size 的序列"""
        if len(self.buffer) == 0:
            return np.zeros((self.buffer_size, self.num_joints, 3), dtype=np.float32)

        seq = np.array(self.buffer, dtype=np.float32)

        if len(seq) < self.buffer_size:
            padding = np.zeros((self.buffer_size - len(seq), self.num_joints, 3),
                              dtype=np.float32)
            seq = np.concatenate([seq, padding], axis=0)

        return seq

    def clear(self):
        """清空緩衝區"""
        self.buffer = []


def azure_to_ntu_skeleton(skeleton_32: np.ndarray) -> np.ndarray:
    """
    將 Azure Kinect 32 點骨架轉換為 NTU 25 點格式

    參數：
        skeleton_32: 形狀為 (T, 32, C) 或 (32, C) 的陣列

    回傳：
        形狀為 (T, 25, C) 或 (25, C) 的陣列
    """
    if skeleton_32.ndim == 2:
        # 單幀
        skeleton_25 = np.zeros((25, skeleton_32.shape[1]), dtype=skeleton_32.dtype)
        for ntu_idx, azure_idx in AZURE_TO_NTU_MAP.items():
            skeleton_25[ntu_idx] = skeleton_32[azure_idx]
        return skeleton_25
    elif skeleton_32.ndim == 3:
        # 序列
        T = skeleton_32.shape[0]
        C = skeleton_32.shape[2]
        skeleton_25 = np.zeros((T, 25, C), dtype=skeleton_32.dtype)
        for ntu_idx, azure_idx in AZURE_TO_NTU_MAP.items():
            skeleton_25[:, ntu_idx, :] = skeleton_32[:, azure_idx, :]
        return skeleton_25
    else:
        raise ValueError(f"不支援的骨架維度: {skeleton_32.ndim}")


if __name__ == '__main__':
    """測試 Azure Kinect 骨架提取器"""

    print("=" * 60)
    print("Azure Kinect DK 骨架提取器測試")
    print("=" * 60)

    # 檢查是否有安裝 pyk4a
    try:
        import pyk4a
        print("[OK] pyk4a 已安裝")
    except ImportError:
        print("[ERROR] 未安裝 pyk4a")
        print("請執行: pip install pyk4a")
        print("\n如果需要從原始碼安裝:")
        print("  git clone https://github.com/etiennedub/pyk4a.git")
        print("  cd pyk4a")
        print("  pip install .")
        exit(1)

    # 初始化提取器
    extractor = AzureKinectSkeletonExtractor(output_format='ntu25')
    buffer = AzureKinectSkeletonBuffer(buffer_size=64, num_joints=25)

    print("\n正在初始化 Azure Kinect...")

    if not extractor.initialize():
        print("無法初始化 Azure Kinect，請確認：")
        print("1. Azure Kinect DK 已連接")
        print("2. Azure Kinect SDK 已安裝")
        print("3. Azure Kinect Body Tracking SDK 已安裝")
        exit(1)

    print("初始化成功！按 'q' 退出, 's' 儲存序列")

    try:
        while True:
            # 提取骨架
            keypoints, bbox = extractor.extract_normalized()

            # 取得彩色影像
            joints, color_image, _ = extractor.extract()

            if color_image is None:
                continue

            # 加入緩衝區
            buffer.add(keypoints)

            # 繪製骨架
            if keypoints is not None:
                color_image = extractor.draw_skeleton(color_image, keypoints, bbox)

            # 顯示緩衝區狀態
            status = f"Buffer: {len(buffer.buffer)}/{buffer.buffer_size}"
            format_info = f"Format: NTU 25 joints"
            cv2.putText(color_image, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, format_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Azure Kinect Skeleton', color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and buffer.is_full():
                seq = buffer.get_sequence()
                np.save('azure_kinect_skeleton_sequence.npy', seq)
                print(f"已儲存序列，形狀: {seq.shape}")

    finally:
        extractor.close()
        cv2.destroyAllWindows()
