#!/usr/bin/env python3
"""
骨架提取器 - 使用 YOLOv8-Pose

從 RGB 影像中提取 17 個 COCO 關鍵點用於動作辨識。
COCO 關鍵點：鼻子、眼睛、耳朵、肩膀、手肘、手腕、髖部、膝蓋、腳踝
"""

import numpy as np
from ultralytics import YOLO
import cv2


class SkeletonExtractor:
    """
    使用 YOLOv8-Pose 模型提取骨架關鍵點。

    COCO 17 關鍵點：
    0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳,
    5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘,
    9: 左腕, 10: 右腕, 11: 左髖, 12: 右髖,
    13: 左膝, 14: 右膝, 15: 左踝, 16: 右踝
    """

    # COCO 關鍵點名稱
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # 骨架連接（用於視覺化）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 臉部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
        (5, 11), (6, 12), (11, 12),  # 軀幹
        (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
    ]

    def __init__(self, model_path='yolov8x-pose.pt', device='cpu', conf_threshold=0.5):
        """
        初始化骨架提取器。

        參數：
            model_path: YOLOv8-Pose 模型路徑或模型名稱
            device: 'cpu' 或 'cuda'
            conf_threshold: 偵測信心度閾值
        """
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.num_keypoints = 17

    def extract(self, image, normalize=True):
        """
        從影像中提取骨架關鍵點。

        參數：
            image: BGR 影像（numpy 陣列）
            normalize: 若為 True，將座標正規化至 [0, 1]

        回傳：
            keypoints: 形狀為 (N, C) 的陣列，N=17, C=3 (x, y, conf)
                      若未偵測到人則回傳 None
            bbox: 偵測到的人的邊界框 [x1, y1, x2, y2]
        """
        # 執行推論
        results = self.model(image, device=self.device, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None

        # 取得關鍵點資料
        keypoints_data = results[0].keypoints

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return None, None

        # 選擇信心度最高的人（若偵測到多人）
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)

            # 檢查信心度是否高於閾值
            if confidences[best_idx] < self.conf_threshold:
                return None, None
        else:
            best_idx = 0

        # 提取最佳人選的關鍵點
        kpts_xy = keypoints_data.xy[best_idx].cpu().numpy()  # (17, 2)

        # 取得信心度分數（如果有的話）
        if keypoints_data.conf is not None:
            kpts_conf = keypoints_data.conf[best_idx].cpu().numpy()  # (17,)
        else:
            kpts_conf = np.ones(self.num_keypoints)

        # 組合成 (17, 3) 陣列
        keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        keypoints[:, :2] = kpts_xy
        keypoints[:, 2] = kpts_conf

        # 正規化座標
        if normalize:
            h, w = image.shape[:2]
            keypoints[:, 0] /= w  # x
            keypoints[:, 1] /= h  # y

        # 取得邊界框
        bbox = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bbox = results[0].boxes.xyxy[best_idx].cpu().numpy()

        return keypoints, bbox

    def extract_all_persons(self, image, normalize=True):
        """
        提取所有偵測到的人的骨架關鍵點。

        參數：
            image: BGR 影像（numpy 陣列）
            normalize: 若為 True，將座標正規化至 [0, 1]

        回傳：
            每個人的 (keypoints, bbox) 元組列表
        """
        results = self.model(image, device=self.device, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return []

        keypoints_data = results[0].keypoints

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return []

        persons = []
        h, w = image.shape[:2]

        for idx in range(len(keypoints_data.xy)):
            # 檢查信心度
            if results[0].boxes is not None:
                conf = results[0].boxes.conf[idx].cpu().item()
                if conf < self.conf_threshold:
                    continue

            kpts_xy = keypoints_data.xy[idx].cpu().numpy()

            if keypoints_data.conf is not None:
                kpts_conf = keypoints_data.conf[idx].cpu().numpy()
            else:
                kpts_conf = np.ones(self.num_keypoints)

            keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            keypoints[:, :2] = kpts_xy
            keypoints[:, 2] = kpts_conf

            if normalize:
                keypoints[:, 0] /= w
                keypoints[:, 1] /= h

            bbox = None
            if results[0].boxes is not None:
                bbox = results[0].boxes.xyxy[idx].cpu().numpy()

            persons.append((keypoints, bbox))

        return persons

    def draw_skeleton(self, image, keypoints, bbox=None, color=(0, 255, 0),
                      thickness=2, normalized=True):
        """
        在影像上繪製骨架。

        參數：
            image: BGR 影像（會被原地修改）
            keypoints: 形狀為 (17, 3) 的陣列
            bbox: 可選的邊界框 [x1, y1, x2, y2]
            color: 繪製顏色
            thickness: 線條粗細
            normalized: 若為 True，關鍵點已正規化

        回傳：
            標註後的影像
        """
        img = image.copy()
        h, w = img.shape[:2]

        # 如需要則反正規化
        kpts = keypoints.copy()
        if normalized:
            kpts[:, 0] *= w
            kpts[:, 1] *= h

        # 繪製關鍵點
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.3:  # 只繪製高信心度的關鍵點
                cv2.circle(img, (int(x), int(y)), 4, color, -1)

        # 繪製骨架連接
        for start, end in self.SKELETON_CONNECTIONS:
            if kpts[start, 2] > 0.3 and kpts[end, 2] > 0.3:
                pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
                pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
                cv2.line(img, pt1, pt2, color, thickness)

        # 繪製邊界框
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        return img


class SkeletonBuffer:
    """
    用於累積時間序列骨架資料的緩衝區。
    """

    def __init__(self, buffer_size=64, num_keypoints=17):
        """
        初始化骨架緩衝區。

        參數：
            buffer_size: 緩衝的幀數
            num_keypoints: 每幀的關鍵點數量
        """
        self.buffer_size = buffer_size
        self.num_keypoints = num_keypoints
        self.buffer = []

    def add(self, keypoints):
        """
        將關鍵點加入緩衝區。

        參數：
            keypoints: 形狀為 (N, C) 的陣列
        """
        if keypoints is not None:
            self.buffer.append(keypoints.copy())
        else:
            # 未偵測到時加入零值
            self.buffer.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))

        # 維持緩衝區大小
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def is_full(self):
        """檢查緩衝區是否已滿。"""
        return len(self.buffer) >= self.buffer_size

    def get_sequence(self):
        """
        取得緩衝的序列。

        回傳：
            形狀為 (T, N, C) 的陣列
        """
        if len(self.buffer) == 0:
            return None

        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        """清空緩衝區。"""
        self.buffer = []

    def get_padded_sequence(self):
        """
        取得填充至 buffer_size 的序列。

        回傳：
            形狀為 (buffer_size, N, C) 的陣列
        """
        if len(self.buffer) == 0:
            return np.zeros((self.buffer_size, self.num_keypoints, 3), dtype=np.float32)

        seq = np.array(self.buffer, dtype=np.float32)

        if len(seq) < self.buffer_size:
            # 用零填充
            padding = np.zeros((self.buffer_size - len(seq), self.num_keypoints, 3),
                              dtype=np.float32)
            seq = np.concatenate([seq, padding], axis=0)

        return seq


if __name__ == '__main__':
    # 測試骨架提取器
    import sys

    # 初始化提取器
    extractor = SkeletonExtractor(model_path='yolov8m-pose.pt')

    # 使用網路攝影機測試
    cap = cv2.VideoCapture(0)
    buffer = SkeletonBuffer(buffer_size=64)

    print("Press 'q' to quit, 's' to save current sequence")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 提取骨架
        keypoints, bbox = extractor.extract(frame, normalize=True)

        # 加入緩衝區
        buffer.add(keypoints)

        # 繪製骨架
        if keypoints is not None:
            frame = extractor.draw_skeleton(frame, keypoints, bbox, normalized=True)

        # 顯示緩衝區狀態
        status = f"Buffer: {len(buffer.buffer)}/{buffer.buffer_size}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow('Skeleton Extraction', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and buffer.is_full():
            seq = buffer.get_sequence()
            np.save('skeleton_sequence.npy', seq)
            print(f"Saved sequence, shape: {seq.shape}")

    cap.release()
    cv2.destroyAllWindows()
