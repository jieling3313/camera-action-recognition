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

# 新增 scripts 目錄到 Python 路徑
scripts_path = "/root/catkin_ws/src/yolo_ros/scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from skeleton_model import OneShotActionRecognition


class ModelManager:
    """模型管理器"""

    def __init__(self):
        # 模型檔案目錄
        self.checkpoint_dir = "/root/catkin_ws/src/yolo_ros/scripts/checkpoints"

        # 當前載入的模型
        self.model = None
        self.current_model_name = None

        # 當前支援的動作列表
        self.action_list = []
        self.support_features = {}  # 動作名稱 -> 特徵向量

        # 裝置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rospy.loginfo(f"Model Manager initialized (device: {self.device})")

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
                # 轉換為 tensor
                sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    feature = self.model.embedding(sample_tensor)  # (1, 256)
                    features.append(feature.cpu().numpy()[0])

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
