#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
即時辨識顯示節點 - 使用 ROS 原生視窗顯示辨識結果
"""

import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
import mediapipe as mp
from collections import deque
import sys
import os

# 添加 scripts 目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from skeleton_model import SkeletonEmbedding
except ImportError:
    rospy.logerr("Failed to import skeleton_model. Make sure skeleton_model.py exists.")

class RecognitionDisplayNode:
    """即時辨識顯示節點類別"""

    def __init__(self):
        """初始化辨識顯示節點"""
        rospy.init_node('recognition_display_node', anonymous=False)

        # CvBridge 用於轉換 ROS 影像訊息
        self.bridge = CvBridge()
        self.current_frame = None

        # MediaPipe 姿態估計初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,      # 影片模式（非靜態圖片）
            model_complexity=1,            # 模型複雜度（0=輕量, 1=標準, 2=精準）
            min_detection_confidence=0.5,  # 最小偵測信心閾值
            min_tracking_confidence=0.5    # 最小追蹤信心閾值
        )

        # 骨架序列緩衝區（儲存最近 64 幀）
        self.skeleton_buffer = deque(maxlen=64)
        self.frame_count = 0
        self.recognition_interval = 30  # 每 30 幀執行一次辨識

        # 當前辨識結果
        self.current_action = "No recognition"
        self.current_confidence = 0.0
        self.all_scores = {}  # 所有動作的信心分數

        # 深度學習模型相關變數
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.support_features = {}  # 儲存每個動作的支援特徵向量
        self.action_names = []      # 已載入的動作名稱列表

        # 載入預訓練模型（如果存在）
        self.load_model()

        # 訂閱相機影像主題
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        # 提供載入動作集的服務
        self.load_actions_srv = rospy.Service(
            '/recognition_display/load_actions',
            Trigger,
            self.handle_load_actions
        )

        # 發布辨識結果到主題（格式：動作名稱,信心分數）
        self.result_pub = rospy.Publisher(
            '/recognition_display/result',
            String,
            queue_size=1
        )

        # 發布辨識結果影像（供 image_view 顯示）
        self.output_image_pub = rospy.Publisher(
            '/recognition_display/output_image',
            Image,
            queue_size=1
        )

        rospy.loginfo("Recognition Display Node initialized")
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo("Services available:")
        rospy.loginfo("  - /recognition_display/load_actions")

    def load_model(self):
        """載入預訓練的骨架辨識模型"""
        checkpoint_path = "/root/catkin_ws/src/yolo_ros/scripts/checkpoints/best.pth"

        # 檢查檢查點檔案是否存在
        if not os.path.exists(checkpoint_path):
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            rospy.logwarn("Model will not be loaded. Recognition will not work.")
            return

        try:
            # 建立模型實例
            self.model = SkeletonEmbedding(in_channels=3, base_channels=64)

            # 載入檢查點
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 載入模型權重（支援兩種檢查點格式）
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            # 將模型移到指定裝置並設為評估模式
            self.model.to(self.device)
            self.model.eval()

            rospy.loginfo("Model loaded successfully")
            rospy.loginfo(f"Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, "
                         f"Acc {checkpoint.get('best_acc', 'N/A')}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            self.model = None

    def handle_load_actions(self, req):
        """處理載入動作集的服務請求

        從 actionset 目錄載入所有動作的骨架序列，
        並使用模型提取特徵向量作為支援集
        """
        if self.model is None:
            return TriggerResponse(success=False, message="Model not loaded")

        actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"

        # 檢查動作集目錄是否存在
        if not os.path.exists(actionset_dir):
            return TriggerResponse(success=False, message=f"Actionset dir not found: {actionset_dir}")

        # 掃描所有動作子目錄（例如：stop, left, right）
        action_dirs = [d for d in os.listdir(actionset_dir)
                      if os.path.isdir(os.path.join(actionset_dir, d))]

        if not action_dirs:
            return TriggerResponse(success=False, message="No actions found in actionset")

        # 清空現有的動作資料
        self.support_features = {}
        self.action_names = []

        # 逐一處理每個動作類別
        for action_name in action_dirs:
            action_path = os.path.join(actionset_dir, action_name)

            # 尋找骨架序列檔案（格式：*_skeleton_sequence.npy）
            skeleton_files = [f for f in os.listdir(action_path)
                            if f.endswith('_skeleton_sequence.npy')]

            if not skeleton_files:
                rospy.logwarn(f"No skeleton sequence found for action: {action_name}")
                continue

            # 載入該動作的所有樣本
            samples = []
            for skeleton_file in skeleton_files:
                skeleton_path = os.path.join(action_path, skeleton_file)
                try:
                    skeleton_data = np.load(skeleton_path)

                    # 轉換 MediaPipe (33點) 到 COCO (17點) 格式
                    # 簡化版：取 MediaPipe 前 17 個關鍵點
                    if skeleton_data.shape[1] == 33:
                        skeleton_data = skeleton_data[:, :17, :]

                    # 調整序列長度到 64 幀
                    if len(skeleton_data) < 64:
                        # 序列過短：重複填充至 64 幀
                        skeleton_data = np.tile(skeleton_data, (64 // len(skeleton_data) + 1, 1, 1))[:64]
                    elif len(skeleton_data) > 64:
                        # 序列過長：均勻採樣取 64 幀
                        indices = np.linspace(0, len(skeleton_data) - 1, 64).astype(int)
                        skeleton_data = skeleton_data[indices]

                    samples.append(skeleton_data)
                except Exception as e:
                    rospy.logwarn(f"Failed to load {skeleton_file}: {e}")

            if not samples:
                rospy.logwarn(f"No valid samples for action: {action_name}")
                continue

            # 使用深度學習模型提取特徵向量
            try:
                with torch.no_grad():
                    features = []
                    for sample in samples:
                        # 將 numpy 陣列轉換為 PyTorch 張量並送到裝置
                        tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                        # 通過模型提取特徵
                        feat = self.model(tensor)
                        features.append(feat)

                    # 計算所有樣本的平均特徵向量作為該動作的代表特徵
                    avg_feature = torch.stack(features).mean(0)
                    self.support_features[action_name] = avg_feature
                    self.action_names.append(action_name)

                rospy.loginfo(f"Loaded action: {action_name} ({len(samples)} samples)")
            except Exception as e:
                rospy.logerr(f"Failed to extract features for {action_name}: {e}")

        if self.action_names:
            return TriggerResponse(
                success=True,
                message=f"Loaded {len(self.action_names)} actions: {', '.join(self.action_names)}"
            )
        else:
            return TriggerResponse(success=False, message="No valid actions loaded")

    def image_callback(self, msg):
        """處理相機影像並進行即時骨架提取與動作辨識

        每幀影像都會進行 MediaPipe 骨架提取，並將關鍵點加入緩衝區。
        當緩衝區滿 64 幀且達到辨識間隔時，執行動作辨識。
        """
        try:
            # 將 ROS 影像訊息轉換為 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_frame = cv_image.copy()

            # 轉換為 RGB 供 MediaPipe 處理
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # 建立顯示影像副本
            display_img = cv_image.copy()

            if results.pose_landmarks:
                # 在影像上繪製骨架關鍵點和連線
                self.mp_drawing.draw_landmarks(
                    display_img,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 提取關鍵點座標（正規化到影像尺寸）
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                h, w = cv_image.shape[:2]
                for lm in landmarks[:17]:  # 只取前 17 個關鍵點（COCO 格式）
                    keypoints.append([lm.x * w, lm.y * h, lm.visibility])

                # 將當前幀的關鍵點加入緩衝區
                self.skeleton_buffer.append(keypoints)

                # 檢查是否達到辨識條件
                self.frame_count += 1
                if self.frame_count >= self.recognition_interval and len(self.skeleton_buffer) == 64:
                    self.recognize_action()
                    self.frame_count = 0

            # 在影像上疊加辨識結果資訊
            self.draw_recognition_result(display_img)

            # 發布辨識結果影像到 ROS topic（供 image_view 顯示）
            try:
                output_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
                self.output_image_pub.publish(output_msg)
            except Exception as e:
                rospy.logerr(f"Failed to publish output image: {e}")

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def recognize_action(self):
        """執行動作辨識"""
        if self.model is None or not self.support_features:
            return

        try:
            # 準備查詢序列
            query_sequence = np.array(list(self.skeleton_buffer))  # (64, 17, 3)
            query_tensor = torch.FloatTensor(query_sequence).unsqueeze(0).to(self.device)

            # 提取特徵
            with torch.no_grad():
                query_feature = self.model(query_tensor)  # (1, 256)

                # 計算與所有動作的相似度（余弦相似度）
                similarities = {}
                for action_name, support_feature in self.support_features.items():
                    # 余弦相似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        query_feature, support_feature, dim=1
                    )
                    similarities[action_name] = cos_sim.item()

                # 正規化到 0-1
                if similarities:
                    max_sim = max(similarities.values())
                    min_sim = min(similarities.values())
                    if max_sim > min_sim:
                        similarities = {
                            k: (v - min_sim) / (max_sim - min_sim)
                            for k, v in similarities.items()
                        }

                    # 找到最高分
                    self.all_scores = similarities
                    self.current_action = max(similarities, key=similarities.get)
                    self.current_confidence = similarities[self.current_action]

                    # 發布結果
                    result_msg = f"{self.current_action},{self.current_confidence:.2f}"
                    self.result_pub.publish(result_msg)

                    rospy.loginfo(f"Recognition: {self.current_action} ({self.current_confidence:.2%})")

        except Exception as e:
            rospy.logerr(f"Error in recognize_action: {e}")

    def draw_recognition_result(self, img):
        """在影像上繪製辨識結果"""
        h, w = img.shape[:2]

        # 背景框
        cv2.rectangle(img, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (w - 10, 120), (255, 255, 255), 2)

        # 當前辨識結果
        text = f"Current: {self.current_action}"
        cv2.putText(img, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        confidence_text = f"Confidence: {self.current_confidence:.1%}"
        cv2.putText(img, confidence_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 緩衝區狀態
        buffer_text = f"Buffer: {len(self.skeleton_buffer)}/64 frames"
        cv2.putText(img, buffer_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 顯示所有動作的分數（右側）
        if self.all_scores:
            y_offset = 150
            cv2.rectangle(img, (w - 300, 130), (w - 10, 130 + len(self.all_scores) * 35 + 20),
                         (0, 0, 0), -1)
            cv2.putText(img, "All Scores:", (w - 290, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += 30
            sorted_scores = sorted(self.all_scores.items(), key=lambda x: x[1], reverse=True)
            for action, score in sorted_scores:
                color = (0, 255, 0) if action == self.current_action else (200, 200, 200)
                text = f"{action}: {score:.1%}"
                cv2.putText(img, text, (w - 290, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 30

    def run(self):
        """主循環"""
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = RecognitionDisplayNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Recognition Display Node")
        cv2.destroyAllWindows()
