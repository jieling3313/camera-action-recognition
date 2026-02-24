#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
即時辨識顯示節點 v2 - 雙模式辨識

支援兩種辨識模式：
1. One-Shot Learning: 使用 Page 2 設定的自訂動作 (從 ROS parameter 讀取)
2. NTU RGB+D Classification: 使用預訓練模型的 60 個類別

Author: Claude AI Assistant
Date: 2025-11-26
"""

import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
import mediapipe as mp
import mediapipe_compat  # 啟用兼容層以支援新版 MediaPipe
from collections import deque
import sys
import os

# 添加 scripts 目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from skeleton_model import SkeletonEmbedding
    from ntu_rgbd_classes import get_class_name, get_all_classes
    from convert_mediapipe_to_coco import mediapipe_to_coco_single_frame
except ImportError as e:
    rospy.logerr(f"Failed to import required modules: {e}")
    # Fallback: 如果轉換函數不可用，使用簡單映射
    def mediapipe_to_coco_single_frame(mediapipe_keypoints):
        """簡化版：直接取前 17 個關鍵點"""
        return np.array(mediapipe_keypoints[:17])

class RecognitionDisplayNodeV2:
    """即時辨識顯示節點 v2（雙模式辨識）"""

    def __init__(self):
        """初始化辨識顯示節點"""
        rospy.init_node('recognition_display_node_v2', anonymous=False)

        # CvBridge 用於轉換 ROS 影像訊息
        self.bridge = CvBridge()
        self.current_frame = None

        # MediaPipe 姿態估計初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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

        # One-Shot Learning 變數（自訂動作）
        self.custom_actions = []        # 自訂動作名稱列表
        self.support_features = {}      # 自訂動作的特徵向量

        # NTU RGB+D Classification 變數
        self.ntu_classes = get_all_classes()  # 60 個類別名稱
        self.use_ntu_classification = False   # 暫時停用 NTU 分類（模型未充分訓練）

        # 載入預訓練模型
        self.load_model()

        # 自動載入自訂動作（從 ROS parameter）
        self._auto_load_custom_actions()

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

        # 發布辨識結果到主題
        self.result_pub = rospy.Publisher(
            '/recognition_display/result',
            String,
            queue_size=1
        )

        # 發布辨識結果影像
        self.output_image_pub = rospy.Publisher(
            '/recognition_display/output_image',
            Image,
            queue_size=1
        )

        rospy.loginfo("Recognition Display Node V2 initialized")
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo(f"NTU RGB+D Classification: {'Enabled' if self.use_ntu_classification else 'Disabled'}")
        rospy.loginfo("Services available:")
        rospy.loginfo("  - /recognition_display/load_actions")

    def load_model(self):
        """載入預訓練的骨架辨識模型"""
        checkpoint_path = "/root/catkin_ws/src/yolo_ros/scripts/checkpoints/best.pth"

        if not os.path.exists(checkpoint_path):
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            rospy.logwarn("Model will not be loaded. Recognition will not work.")
            return

        try:
            # 檢查 checkpoint 中的類別數
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            num_classes = None

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'classifier.weight' in state_dict:
                    num_classes = state_dict['classifier.weight'].shape[0]
                    rospy.loginfo(f"Model trained with {num_classes} classes (NTU RGB+D)")

            # 建立模型實例（包含 classifier）
            self.model = SkeletonEmbedding(
                in_channels=3,
                base_channels=64,
                num_classes=num_classes  # 60 for NTU RGB+D
            )

            # 載入模型權重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            # 將模型移到指定裝置並設為評估模式
            self.model.to(self.device)
            self.model.eval()

            rospy.loginfo("Model loaded successfully")
            rospy.loginfo(f"Checkpoint: Epoch {checkpoint.get('epoch', 'N/A')}, "
                         f"Acc {checkpoint.get('best_acc', 'N/A')}%")

        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _auto_load_custom_actions(self):
        """初始化時自動載入自訂動作（從 ROS parameter）"""
        try:
            # 嘗試從 ROS parameter 讀取自訂動作
            if rospy.has_param('/model_manager/custom_actions'):
                custom_actions = rospy.get_param('/model_manager/custom_actions', [])
                support_features_dict = rospy.get_param('/model_manager/support_features', {})

                if custom_actions and support_features_dict:
                    self.custom_actions = custom_actions
                    self.support_features = {
                        name: np.array(features)
                        for name, features in support_features_dict.items()
                    }
                    rospy.loginfo(f"Auto-loaded {len(self.custom_actions)} custom actions: {', '.join(self.custom_actions)}")
                else:
                    rospy.loginfo("No custom actions found in ROS parameters (this is normal on first startup)")
            else:
                rospy.loginfo("No custom actions parameter found (this is normal on first startup)")

        except Exception as e:
            rospy.logwarn(f"Failed to auto-load custom actions: {e}")
            rospy.logwarn("You can manually load actions from Page 2 or Page 3")

    def handle_load_actions(self, req):
        """處理載入動作集的服務請求

        從 ROS parameter server 讀取 Page 2 設定的自訂動作列表
        """
        if self.model is None:
            return TriggerResponse(success=False, message="Model not loaded")

        try:
            # 從 ROS parameter 讀取自訂動作列表
            if rospy.has_param('/model_manager/custom_actions'):
                self.custom_actions = rospy.get_param('/model_manager/custom_actions')
                rospy.loginfo(f"Loaded custom actions from ROS param: {self.custom_actions}")
            else:
                self.custom_actions = []
                rospy.logwarn("No custom actions found in ROS parameters")

            # 從 ROS parameter 讀取特徵向量
            if rospy.has_param('/model_manager/support_features'):
                features_dict = rospy.get_param('/model_manager/support_features')
                self.support_features = {}
                for action_name, feature_list in features_dict.items():
                    self.support_features[action_name] = np.array(feature_list, dtype=np.float32)
                rospy.loginfo(f"Loaded {len(self.support_features)} support features")
            else:
                self.support_features = {}
                rospy.logwarn("No support features found in ROS parameters")

            if self.custom_actions:
                return TriggerResponse(
                    success=True,
                    message=f"Loaded {len(self.custom_actions)} custom actions: {', '.join(self.custom_actions)}"
                )
            else:
                return TriggerResponse(
                    success=True,
                    message="No custom actions loaded. Using NTU RGB+D classification only."
                )

        except Exception as e:
            rospy.logerr(f"Failed to load actions from ROS params: {e}")
            import traceback
            traceback.print_exc()
            return TriggerResponse(success=False, message=f"Error: {str(e)}")

    def recognize_one_shot(self, query_feature):
        """使用 One-Shot Learning 辨識自訂動作

        Args:
            query_feature: 查詢特徵向量 (256,)

        Returns:
            dict: {action_name: confidence_score}
        """
        if not self.custom_actions or not self.support_features:
            return {}

        scores = {}
        for action_name, support_feature in self.support_features.items():
            # 計算餘弦相似度 (範圍: -1 到 1)
            similarity = np.dot(query_feature, support_feature) / (
                np.linalg.norm(query_feature) * np.linalg.norm(support_feature) + 1e-8
            )
            # 將相似度從 [-1, 1] 映射到 [0, 100]
            # similarity = -1 -> confidence = 0
            # similarity = 0 -> confidence = 50
            # similarity = 1 -> confidence = 100
            confidence = (similarity + 1) * 50.0
            # 確保在 [0, 100] 範圍內
            confidence = max(0.0, min(100.0, confidence))
            scores[action_name] = float(confidence)

        return scores

    def recognize_ntu_classification(self, skeleton_sequence):
        """使用 NTU RGB+D 分類器辨識動作

        Args:
            skeleton_sequence: 骨架序列 (64, 17, 3)

        Returns:
            dict: {class_name: confidence_score}
        """
        if self.model is None or self.model.classifier is None:
            return {}

        try:
            # 轉換為 tensor
            tensor = torch.FloatTensor(skeleton_sequence).unsqueeze(0).to(self.device)  # (1, 64, 17, 3)

            with torch.no_grad():
                # 使用分類器（包含 classifier 層）
                logits = self.model(tensor)  # (1, 60)
                probs = F.softmax(logits, dim=1)[0]  # (60,)

                # 除錯：顯示 top 5 logits 和 probs
                top5_indices = torch.topk(logits[0], 5).indices.cpu().numpy()
                top5_logits = logits[0][top5_indices].cpu().numpy()
                top5_probs = probs[top5_indices].cpu().numpy()

                rospy.loginfo_throttle(5, f"NTU Top 5 logits: {top5_logits}")
                rospy.loginfo_throttle(5, f"NTU Top 5 probs: {top5_probs * 100}")

                # 轉換為字典 {class_name: confidence%}
                scores = {}
                for i, prob in enumerate(probs.cpu().numpy()):
                    class_name = get_class_name(i)
                    scores[class_name] = float(prob * 100)

            return scores

        except Exception as e:
            rospy.logerr(f"NTU classification failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def image_callback(self, msg):
        """處理相機影像並進行即時骨架提取與動作辨識"""
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
                # 繪製骨架
                self.mp_drawing.draw_landmarks(
                    display_img,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 提取 MediaPipe 33 關鍵點
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                h, w = cv_image.shape[:2]
                for lm in landmarks:
                    keypoints.append([lm.x * w, lm.y * h, lm.visibility])

                # 轉換為 COCO 17 關鍵點（使用正確的映射函數）
                keypoints_array = mediapipe_to_coco_single_frame(np.array(keypoints))  # (17, 3)

                # 加入緩衝區
                self.skeleton_buffer.append(keypoints_array)

                # 執行辨識
                self.frame_count += 1
                if len(self.skeleton_buffer) >= 64 and self.frame_count % self.recognition_interval == 0:
                    skeleton_sequence = np.array(list(self.skeleton_buffer))  # (64, 17, 3)

                    # 執行雙模式辨識
                    all_scores = {}

                    # 1. One-Shot Learning（自訂動作）
                    if self.custom_actions and self.support_features:
                        # 提取特徵向量（繞過分類器）
                        tensor = torch.FloatTensor(skeleton_sequence).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            # 臨時移除分類器以獲取特徵向量
                            has_classifier = self.model.classifier is not None
                            if has_classifier:
                                original_classifier = self.model.classifier
                                self.model.classifier = None

                            query_feature = self.model.forward(tensor, return_multi_scale=False)  # (1, 256)
                            query_feature = query_feature.cpu().numpy()[0]  # (256,)

                            # 恢復分類器
                            if has_classifier:
                                self.model.classifier = original_classifier

                        custom_scores = self.recognize_one_shot(query_feature)
                        all_scores.update(custom_scores)

                    # 2. NTU RGB+D Classification
                    if self.use_ntu_classification:
                        ntu_scores = self.recognize_ntu_classification(skeleton_sequence)
                        all_scores.update(ntu_scores)

                    # 更新結果
                    if all_scores:
                        self.all_scores = all_scores
                        # 找出最高分數的動作
                        best_action = max(all_scores.items(), key=lambda x: x[1])
                        self.current_action = best_action[0]
                        self.current_confidence = best_action[1]

                        # 發布結果
                        result_msg = f"{self.current_action},{self.current_confidence:.2f}"
                        self.result_pub.publish(result_msg)

            # 在影像上顯示辨識結果
            self._draw_recognition_results(display_img)

            # 發布辨識結果影像
            output_msg = self.bridge.cv2_to_imgmsg(display_img, "bgr8")
            self.output_image_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def _draw_recognition_results(self, image):
        """在影像上繪製辨識結果（右下角）"""
        h, w = image.shape[:2]

        # 計算右下角位置
        box_width = 400
        box_height = 240
        x_start = w - box_width - 10
        y_start = h - box_height - 10
        x_end = w - 10
        y_end = h - 10

        # 背景矩形（右下角）
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

        # 當前辨識結果（調整位置）
        text_x = x_start + 10
        text_y = y_start + 40

        cv2.putText(
            image,
            f"Action: {self.current_action}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            image,
            f"Confidence: {self.current_confidence:.2f}%",
            (text_x, text_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # 顯示 Top 5 分數
        if self.all_scores:
            sorted_scores = sorted(self.all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            cv2.putText(image, "Top 5:", (text_x, text_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            for i, (action, score) in enumerate(sorted_scores):
                y_pos = text_y + 95 + i * 22
                color = (0, 255, 0) if i == 0 else (180, 180, 180)
                text = f"{i+1}. {action[:30]}: {score:.1f}%"
                cv2.putText(image, text, (text_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def run(self):
        """主循環"""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RecognitionDisplayNodeV2()
        node.run()
    except rospy.ROSInterruptException:
        pass
