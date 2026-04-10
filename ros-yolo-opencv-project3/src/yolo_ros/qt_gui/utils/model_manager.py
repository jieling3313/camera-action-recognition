#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
用於載入、管理訓練模型和執行動作辨識

Author: Claude AI Assistant
Date: 2025-11-23
"""

import os
import glob
import torch
import numpy as np
import rospy
import sys

# 新增 scripts/models 目錄到 Python 路徑
scripts_path = "/root/catkin_ws/src/yolo_ros/scripts"
models_path = os.path.join(scripts_path, "models")
mediapipe33_path = os.path.join(models_path, "mediapipe33")
common_path = os.path.join(models_path, "common")

for path in [scripts_path, models_path, mediapipe33_path, common_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 使用 MediaPipe 33-point 模型
from skeleton_model_mediapipe33 import OneShotActionRecognitionMediaPipe as OneShotActionRecognition


class ModelManager:
    """模型管理器"""

    def __init__(self):
        # 模型檔案目錄 (使用 MediaPipe 33-point 模型的 checkpoints)
        self.checkpoint_dir = "/root/catkin_ws/src/yolo_ros/scripts/models/mediapipe33/checkpoints_mediapipe33"

        # 當前載入的模型
        self.model = None
        self.current_model_name = None

        # 當前支援的動作列表
        self.action_list = []
        self.support_features = {}  # 動作名稱 -> 特徵向量

        # 裝置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rospy.loginfo(f"Model Manager initialized (device: {self.device})")

    def _normalize_skeleton_data(self, skeleton_data):
        """正規化骨架數據，確保座標在合理範圍內

        MediaPipe 的正規化座標通常在 0-1 範圍，但當身體部分超出畫面時
        可以超過 1 或小於 0（例如 y 可能到 2-3）。

        真正的像素座標會是 0-640 和 0-480 的範圍。

        Args:
            skeleton_data: (T, V, C) 骨架序列

        Returns:
            骨架序列（不做任何正規化，因為數據應該已經是正規化座標）
        """
        skeleton_data = np.array(skeleton_data, dtype=np.float32)

        # 檢查 x, y 座標的範圍
        x_coords = skeleton_data[..., 0]
        y_coords = skeleton_data[..., 1]

        x_max = np.max(x_coords)
        y_max = np.max(y_coords)
        x_min = np.min(x_coords)
        y_min = np.min(y_coords)

        rospy.loginfo(f"Skeleton data range: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

        # 只有當 x 座標明顯超過 10（不可能是正規化座標）時才認為是像素座標
        # MediaPipe 的正規化座標即使超出畫面也不會超過 3-4
        if x_max > 10:
            rospy.logwarn(f"Detected pixel coordinates (max x={x_max:.1f}). "
                         f"Auto-normalizing to 0-1 range.")

            # 使用 640x480 作為參考解析度
            skeleton_data[..., 0] = skeleton_data[..., 0] / 640.0
            skeleton_data[..., 1] = skeleton_data[..., 1] / 480.0

            rospy.loginfo(f"Normalized: x/=640, y/=480")
        else:
            rospy.loginfo("Data appears to be normalized coordinates (no conversion needed)")

        return skeleton_data

    def get_available_models(self):
        """取得可用的模型列表

        Returns:
            dict: {model_name: {epoch: int, accuracy: float, ...}}
        """
        models = {}

        if not os.path.exists(self.checkpoint_dir):
            rospy.logwarn(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return models

        # 尋找所有 .pth 檔案
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))

        for ckpt_path in checkpoint_files:
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')

                model_name = os.path.basename(ckpt_path).replace('.pth', '')

                models[model_name] = {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'accuracy': checkpoint.get('best_acc', 'N/A'),
                    'num_classes': checkpoint.get('num_classes', 'N/A'),
                    'path': ckpt_path
                }

            except Exception as e:
                rospy.logwarn(f"Failed to load checkpoint {ckpt_path}: {e}")

        return models

    def load_model(self, model_name):
        """載入指定模型

        Args:
            model_name: 模型名稱 (例如 'best', 'epoch_50')

        Returns:
            bool: 是否成功載入
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.pth")

        if not os.path.exists(checkpoint_path):
            rospy.logerr(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # 建立模型
            self.model = OneShotActionRecognition(
                in_channels=3,
                base_channels=64
            )

            # 載入 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 載入模型權重
            if 'model_state_dict' in checkpoint:
                self.model.embedding.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=False
                )
            else:
                rospy.logwarn("No model_state_dict found in checkpoint")

            self.model.to(self.device)
            self.model.eval()

            self.current_model_name = model_name

            rospy.loginfo(f"Successfully loaded model: {model_name}")

            # 清空當前動作列表
            self.action_list = []
            self.support_features = {}

            return True

        except Exception as e:
            rospy.logerr(f"Failed to load model {model_name}: {e}")
            return False

    def is_model_loaded(self):
        """檢查是否已載入模型"""
        return self.model is not None

    def get_model_info(self, model_name):
        """取得模型資訊

        Args:
            model_name: 模型名稱

        Returns:
            dict: 模型資訊
        """
        models = self.get_available_models()
        return models.get(model_name, {})

    def get_current_actions(self):
        """取得目前支援的動作列表"""
        return self.action_list.copy()

    def _sample_to_fixed_length(self, sample, target_length=32):
        """將骨架序列調整為固定長度

        這確保與 recognition_display_node_v2.py 的輸入格式一致。

        Args:
            sample: (T, V, C) 骨架序列
            target_length: 目標長度（預設 32，與即時辨識一致）

        Returns:
            調整後的骨架序列 (target_length, V, C)
        """
        T = sample.shape[0]

        if T == target_length:
            return sample
        elif T > target_length:
            # 如果序列太長，均勻取樣
            indices = np.linspace(0, T - 1, target_length, dtype=int)
            return sample[indices]
        else:
            # 如果序列太短，重複最後一幀來填充
            pad_length = target_length - T
            padding = np.repeat(sample[-1:], pad_length, axis=0)
            return np.concatenate([sample, padding], axis=0)

    def add_custom_action(self, action_name, skeleton_samples):
        """新增自訂動作

        Args:
            action_name: 動作名稱
            skeleton_samples: 骨骼樣本列表 [array1, array2, ...]
                             每個 array shape: (T, V, C)

        Returns:
            bool: 是否成功新增
        """
        if self.model is None:
            rospy.logerr("No model loaded")
            return False

        try:
            # 將樣本轉換為特徵向量
            features = []

            for sample in skeleton_samples:
                # 檢查並正規化數據（確保座標在 0-1 範圍內）
                sample = self._normalize_skeleton_data(sample)

                # 調整為固定長度（64 幀，與即時辨識一致）
                sample = self._sample_to_fixed_length(sample, target_length=32)
                rospy.loginfo(f"Sample adjusted to shape: {sample.shape}")

                # 關鍵步驟：髖部中心化（與訓練時的預處理一致）
                # MediaPipe: 23=左髖, 24=右髖
                hip_center = (sample[:, 23, :2] + sample[:, 24, :2]) / 2
                sample[:, :, :2] = sample[:, :, :2] - hip_center[:, np.newaxis, :]
                rospy.loginfo(f"Applied hip-centering normalization")

                # 轉換為 tensor
                sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    feature = self.model.embedding(sample_tensor)  # (1, 256)
                    features.append(feature.cpu().numpy()[0])

                # 除錯：顯示特徵統計
                rospy.loginfo(f"Feature stats: mean={feature.cpu().numpy()[0].mean():.4f}, "
                             f"std={feature.cpu().numpy()[0].std():.4f}, "
                             f"norm={np.linalg.norm(feature.cpu().numpy()[0]):.4f}")

            # 計算平均特徵
            mean_feature = np.mean(features, axis=0)

            # 儲存
            self.support_features[action_name] = mean_feature
            self.action_list.append(action_name)

            # 更新 ROS parameter server（讓 recognition_display_node 可以讀取）
            self._update_ros_params()

            rospy.loginfo(f"Added custom action: {action_name} ({len(skeleton_samples)} samples)")

            return True

        except Exception as e:
            rospy.logerr(f"Failed to add custom action {action_name}: {e}")
            return False

    def remove_action(self, action_name):
        """移除動作

        Args:
            action_name: 動作名稱

        Returns:
            bool: 是否成功移除
        """
        if action_name in self.action_list:
            self.action_list.remove(action_name)
            del self.support_features[action_name]

            # 更新 ROS parameter server
            self._update_ros_params()

            rospy.loginfo(f"Removed action: {action_name}")
            return True

        return False

    def _update_ros_params(self):
        """更新 ROS parameter server with current action list and features"""
        try:
            # 儲存動作列表
            rospy.set_param('/model_manager/custom_actions', self.action_list)

            # 儲存特徵向量（轉換為 list 以便 ROS param 儲存）
            features_dict = {}
            for action_name, feature in self.support_features.items():
                features_dict[action_name] = feature.tolist()

            rospy.set_param('/model_manager/support_features', features_dict)

            # 儲存目前使用的模型名稱（讓 recognition_display_node 使用相同模型）
            if self.current_model_name:
                rospy.set_param('/model_manager/current_model', self.current_model_name)
                rospy.loginfo(f"Updated ROS params: {len(self.action_list)} custom actions, model={self.current_model_name}")
            else:
                rospy.loginfo(f"Updated ROS params: {len(self.action_list)} custom actions")

        except Exception as e:
            rospy.logerr(f"Failed to update ROS params: {e}")

    def get_action_data(self, action_name):
        """取得動作資料（用於預覽）

        Args:
            action_name: 動作名稱

        Returns:
            numpy array: 骨骼序列，若無則返回 None
        """
        # 這裡返回 None，因為我們只儲存特徵向量
        # 實際預覽會從 actionset 資料夾讀取
        return None

    def recognize_action(self, skeleton_sequence):
        """辨識動作

        Args:
            skeleton_sequence: 骨骼序列 (T, V, C)

        Returns:
            prediction: 預測的動作名稱
            confidences: 所有動作的信心分數 dict {action: score}
        """
        if self.model is None or len(self.action_list) == 0:
            rospy.logwarn("Model not loaded or no actions defined")
            return None, {}

        try:
            # 轉換為 tensor
            query_tensor = torch.FloatTensor(skeleton_sequence).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 提取查詢特徵
                query_feature = self.model.embedding(query_tensor)  # (1, 256)
                query_feature = query_feature.cpu().numpy()[0]

                # 計算與所有支援動作的相似度
                similarities = {}

                for action_name, support_feature in self.support_features.items():
                    # 計算余弦相似度
                    similarity = np.dot(query_feature, support_feature) / (
                        np.linalg.norm(query_feature) * np.linalg.norm(support_feature)
                    )

                    # 轉換為百分比
                    confidence = (similarity + 1) * 50  # [-1, 1] -> [0, 100]
                    similarities[action_name] = confidence

                # 找出最高分數
                if len(similarities) > 0:
                    predicted_action = max(similarities.keys(), key=lambda k: similarities[k])
                    return predicted_action, similarities
                else:
                    return None, {}

        except Exception as e:
            rospy.logerr(f"Error during recognition: {e}")
            return None, {}
