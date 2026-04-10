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
from std_msgs.msg import String, Bool, Float32MultiArray
from std_srvs.srv import Trigger, TriggerResponse
# 注意: 不使用 cv_bridge，改用 CvBridgeSimple 避免 NumPy 版本衝突
import mediapipe as mp
from collections import deque
import sys
import os

# 添加 scripts/models 目錄到 Python 路徑
scripts_path = "/root/catkin_ws/src/yolo_ros/scripts"
models_path = os.path.join(scripts_path, "models")
mediapipe33_path = os.path.join(models_path, "mediapipe33")
common_path = os.path.join(models_path, "common")
legacy_path = os.path.join(models_path, "legacy")

for path in [scripts_path, models_path, mediapipe33_path, common_path, legacy_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 啟用 MediaPipe 兼容層（為新版 MediaPipe 0.10.x+ 提供 mp.solutions API）
# 需要 /root/.mediapipe/models/pose_landmarker.task 模型文件
try:
    import mediapipe_compat
except ImportError as e:
    print(f"Warning: mediapipe_compat not found: {e}")


class CvBridgeSimple:
    """Simple replacement for cv_bridge to avoid NumPy version conflicts"""

    def imgmsg_to_cv2(self, img_msg, desired_encoding="bgr8"):
        """Convert ROS Image message to OpenCV image"""
        dtype = np.uint8
        if img_msg.encoding == "32FC1":
            dtype = np.float32
        elif img_msg.encoding == "16UC1":
            dtype = np.uint16

        img = np.frombuffer(img_msg.data, dtype=dtype)

        if img_msg.encoding in ["rgb8", "bgr8"]:
            img = img.reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding == "rgba8" or img_msg.encoding == "bgra8":
            img = img.reshape((img_msg.height, img_msg.width, 4))
        elif img_msg.encoding in ["mono8", "8UC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        elif img_msg.encoding in ["mono16", "16UC1", "32FC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        else:
            try:
                img = img.reshape((img_msg.height, img_msg.width, 3))
            except:
                img = img.reshape((img_msg.height, img_msg.width))

        if img_msg.encoding == "rgb8" and desired_encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "bgr8" and desired_encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
        """Convert OpenCV image to ROS Image message"""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding

        if len(cv_image.shape) == 3:
            img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        else:
            img_msg.step = cv_image.shape[1]

        img_msg.data = cv_image.tobytes()
        return img_msg

try:
    # 優先使用 MediaPipe 33-point 模型
    # 注意：要用 OneShotActionRecognitionMediaPipe，與 model_manager.py 一致
    from skeleton_model_mediapipe33 import (
        MediaPipeSkeletonEmbedding as SkeletonEmbedding,
        OneShotActionRecognitionMediaPipe
    )
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

        # CvBridgeSimple 用於轉換 ROS 影像訊息（避免 NumPy 版本衝突）
        self.bridge = CvBridgeSimple()
        self.current_frame = None

        # MediaPipe 姿態估計初始化（備用模式，當沒有收到共享骨架數據時使用）
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 共享骨架數據模式（優先使用 camera_display_node 發布的骨架數據）
        # 這樣可以避免重複執行 MediaPipe，減少 CPU 負載
        self.use_shared_skeleton = True  # 是否使用共享骨架數據
        self.shared_keypoints = None     # 從 camera_display_node 接收的骨架數據
        self.shared_keypoints_timestamp = None  # 接收時間戳

        # 骨架序列緩衝區（儲存最近 32 幀）
        # 使用較小窗口 (32) 而非 64，更精確捕捉動作變化
        self.skeleton_buffer = deque(maxlen=32)
        self.frame_count = 0
        self.recognition_interval = 16  # 每 16 幀執行一次辨識

        # 運動檢測參數
        self.motion_threshold = 0.015  # 運動量閾值（低於此值視為靜止）
        self.last_keypoints = None
        self.motion_history = deque(maxlen=30)  # 最近 30 幀的運動量
        self.is_moving = False  # 是否正在移動
        self.min_motion_frames = 10  # 至少需要連續 N 幀有運動才開始辨識

        # 靜態動作關鍵字（這些動作會在靜止時被選中）
        self.static_action_keywords = ['stop', 'idle', 'stand', 'wait', 'pause']

        # Similarity 閾值：低於此值視為「未知動作」
        # 當最高 raw similarity 低於此閾值時，不會辨識為任何已知動作
        self.similarity_threshold = 0.88  # 提高閾值以避免誤判

        # 相對差距閾值：最高與第二高相似度的差距必須大於此值
        # 如果差距太小，表示模型不確定，應辨識為 Unknown
        self.similarity_margin = 0.05  # 恢復原值

        # ========== 手臂優先辨識模式 ==========
        # MediaPipe 33 關節點中的手臂相關索引
        # 11,12=肩膀, 13,14=手肘, 15,16=手腕, 17-22=手指
        self.arm_joints = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        self.use_arm_only_recognition = True  # 啟用只用手臂辨識

        # ========== 觸發手勢模式 (Trigger Gesture Mode) ==========
        # 新流程: 偵測到 trigger -> 錄製固定時間 -> 自動結束並辨識
        # IDLE: 等待觸發手勢
        # RECORDING: 正在錄製動作序列（固定時間後自動結束）
        self.trigger_mode_enabled = False  # 是否啟用觸發手勢模式
        self.trigger_action_name = None    # 觸發手勢的動作名稱（例如 "circle"）
        self.trigger_state = "IDLE"        # 當前狀態: IDLE, RECORDING
        self.trigger_threshold = 0.83      # 觸發手勢的相似度閾值（降低以便偵測靜態動作）
        self.recording_actions = []        # 錄製期間識別到的動作列表（舊方式，保留兼容）
        self.recording_skeletons = []      # 錄製期間收集的骨架幀（新方式：均勻取樣）
        self.recording_start_time = None   # 錄製開始時間
        self.recording_duration = 5.0      # 錄製持續時間（秒）- 固定 5 秒
        self.last_trigger_time = 0         # 上次觸發的時間（防抖動）
        self.trigger_cooldown = 1.0        # 觸發冷卻時間（秒）- 錄製結束後的冷卻
        self.final_command = None          # 最終輸出的指令
        self.final_command_time = None     # 指令輸出時間（用於顯示）

        # 當前辨識結果
        self.current_action = "No recognition"
        self.current_confidence = 0.0
        self.all_scores = {}  # 所有動作的信心分數
        self.max_raw_similarity = 0.0  # 最高原始相似度（用於閾值判斷）

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

        # 訂閱共享骨架數據（從 camera_display_node 發布）
        # 這樣可以避免重複執行 MediaPipe，減少 CPU 負載約 15-25%
        self.keypoints_sub = rospy.Subscriber(
            "/camera/skeleton_keypoints",
            Float32MultiArray,
            self.keypoints_callback,
            queue_size=1
        )

        # 提供載入動作集的服務
        self.load_actions_srv = rospy.Service(
            '/recognition_display/load_actions',
            Trigger,
            self.handle_load_actions
        )

        # 訂閱觸發動作設定（使用簡單的 topic 方式）
        self.trigger_sub = rospy.Subscriber(
            '/recognition_display/set_trigger',
            String,
            self.handle_set_trigger,
            queue_size=1
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
        rospy.loginfo(f"Shared skeleton mode: {'Enabled' if self.use_shared_skeleton else 'Disabled'}")
        rospy.loginfo("Services available:")
        rospy.loginfo("  - /recognition_display/load_actions")
        rospy.loginfo("Subscribing to:")
        rospy.loginfo("  - /camera/skeleton_keypoints (shared skeleton data from camera_display_node)")

    def load_model(self, model_name=None):
        """載入預訓練的骨架辨識模型

        重要：載入方式必須與 model_manager.py 一致，
        才能確保 One-Shot Learning 的特徵提取正確。

        Args:
            model_name: 模型名稱（例如 'best', 'latest'），若為 None 則嘗試從 ROS param 讀取
        """
        # checkpoint 目錄
        checkpoint_dir = "/root/catkin_ws/src/yolo_ros/scripts/models/mediapipe33/checkpoints_mediapipe33"

        # 決定使用哪個模型
        if model_name is None:
            # 嘗試從 ROS parameter 讀取（與 model_manager.py 同步）
            model_name = rospy.get_param('/model_manager/current_model', 'best')
            rospy.loginfo(f"Using model from ROS param: {model_name}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")

        # 如果指定的模型不存在，退回使用 best.pth
        if not os.path.exists(checkpoint_path):
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
            if not os.path.exists(checkpoint_path):
                rospy.logwarn(f"Fallback checkpoint also not found: {checkpoint_path}")
                rospy.logwarn("Model will not be loaded. Recognition will not work.")
                return
            rospy.loginfo(f"Using fallback checkpoint: best.pth")

        self.current_model_path = checkpoint_path

        try:
            # 載入 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 使用 OneShotActionRecognitionMediaPipe（與 model_manager.py 一致）
            # 這個類別包含 self.embedding 和 self.matcher
            self.model = OneShotActionRecognitionMediaPipe(
                in_channels=3,
                base_channels=64
            )

            # 載入權重到 embedding 部分（與 model_manager.py 一致）
            if 'model_state_dict' in checkpoint:
                # 顯示 checkpoint 中的鍵
                state_dict = checkpoint['model_state_dict']
                rospy.loginfo(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")

                # 載入權重，並檢查哪些鍵被載入/忽略
                missing_keys, unexpected_keys = self.model.embedding.load_state_dict(
                    state_dict,
                    strict=False
                )
                if missing_keys:
                    rospy.logwarn(f"Missing keys (not loaded): {missing_keys[:5]}...")
                if unexpected_keys:
                    rospy.loginfo(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")

                rospy.loginfo("Loaded weights to model.embedding (consistent with model_manager)")
            else:
                rospy.logwarn("No model_state_dict found in checkpoint")

            # 將模型移到指定裝置並設為評估模式
            self.model.to(self.device)
            self.model.eval()

            rospy.loginfo("Model loaded successfully (OneShotActionRecognitionMediaPipe)")
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

                    # 顯示每個動作的特徵統計（用於除錯）
                    for name, feat in self.support_features.items():
                        rospy.loginfo(f"  {name}: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}, norm={np.linalg.norm(feat):.4f}")

                    # 計算 support features 之間的相似度（診斷用）
                    action_names = list(self.support_features.keys())
                    if len(action_names) >= 2:
                        rospy.loginfo("Support feature similarities (should be LOW for different actions):")
                        for i, a1 in enumerate(action_names):
                            for a2 in action_names[i+1:]:
                                f1 = self.support_features[a1]
                                f2 = self.support_features[a2]
                                sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
                                rospy.loginfo(f"  {a1} vs {a2}: similarity = {sim:.4f}")
                else:
                    rospy.loginfo("No custom actions found in ROS parameters (this is normal on first startup)")
            else:
                rospy.loginfo("No custom actions parameter found (this is normal on first startup)")

        except Exception as e:
            rospy.logwarn(f"Failed to auto-load custom actions: {e}")
            rospy.logwarn("You can manually load actions from Page 2 or Page 3")

    def keypoints_callback(self, msg):
        """接收來自 camera_display_node 的共享骨架數據

        這樣可以避免在本節點重複執行 MediaPipe，減少 CPU 負載約 15-25%。

        Args:
            msg: Float32MultiArray，包含 33*3=99 個值 (x, y, visibility) * 33 關節
        """
        import time
        try:
            # 將 flat array 轉換為 (33, 3) 形狀
            data = np.array(msg.data, dtype=np.float32)
            if len(data) == 99:  # 33 joints * 3 values
                self.shared_keypoints = data.reshape(33, 3)
                self.shared_keypoints_timestamp = time.time()
            else:
                rospy.logwarn_throttle(5.0, f"Invalid keypoints data length: {len(data)}, expected 99")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error processing shared keypoints: {e}")

    def handle_set_trigger(self, msg):
        """處理設定觸發動作的請求

        訊息格式:
        - "action_name" : 啟用觸發模式，使用指定動作
        - "DISABLE" : 停用觸發模式

        Args:
            msg: String 訊息
        """
        action_name = msg.data.strip()

        if action_name.upper() == "DISABLE" or action_name == "":
            self.set_trigger_action(None, enabled=False)
            rospy.loginfo("Trigger mode disabled")
        else:
            # 檢查動作是否存在於自訂動作列表中
            if action_name in self.custom_actions:
                self.set_trigger_action(action_name, enabled=True)
                rospy.loginfo(f"Trigger mode enabled with action: {action_name}")
            else:
                rospy.logwarn(f"Trigger action '{action_name}' not found in custom actions: {self.custom_actions}")
                rospy.logwarn("Please add this action first in Page 2")

    def handle_load_actions(self, req):
        """處理載入動作集的服務請求

        從 ROS parameter server 讀取 Page 2 設定的自訂動作列表
        同時檢查模型是否需要重新載入（確保與 model_manager.py 使用相同模型）
        """
        _ = req  # 忽略未使用的參數

        # 檢查模型是否需要重新載入
        if rospy.has_param('/model_manager/current_model'):
            expected_model = rospy.get_param('/model_manager/current_model')
            current_model_path = getattr(self, 'current_model_path', None)

            if current_model_path:
                current_model_name = os.path.basename(current_model_path).replace('.pth', '')
                if current_model_name != expected_model:
                    rospy.loginfo(f"Model mismatch detected: current={current_model_name}, expected={expected_model}")
                    rospy.loginfo("Reloading model to match model_manager...")
                    self.load_model(expected_model)

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

                # 載入手臂特徵（用於手臂優先辨識）
                self._load_arm_features()
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

    def _load_arm_features(self):
        """載入手臂特徵（從原始骨架序列計算）

        從 actionset 目錄讀取原始骨架序列，
        只提取手臂關節並進行 hip-centering 和均勻取樣，
        生成用於手臂優先辨識的特徵。
        """
        import os
        self.support_arm_features = {}

        # actionset 目錄路徑
        actionset_path = "/root/catkin_ws/src/yolo_ros/actionset"

        for action_name in self.custom_actions:
            try:
                # 尋找對應的骨架序列檔案
                action_folder = None
                for folder in os.listdir(actionset_path):
                    # 動作名稱可能是 "circle"，資料夾可能是 "circlev8"
                    if folder.startswith(action_name) or action_name in folder:
                        folder_path = os.path.join(actionset_path, folder)
                        if os.path.isdir(folder_path):
                            action_folder = folder_path
                            break

                if not action_folder:
                    rospy.logwarn(f"Cannot find folder for action: {action_name}")
                    continue

                # 尋找骨架序列檔案
                npy_file = None
                for f in os.listdir(action_folder):
                    if f.endswith('skeleton_sequence.npy') and 'coco17' not in f and 'backup' not in f:
                        npy_file = os.path.join(action_folder, f)
                        break

                if not npy_file:
                    rospy.logwarn(f"Cannot find skeleton file in: {action_folder}")
                    continue

                # 載入骨架序列
                data = np.load(npy_file)
                T = data.shape[0]

                # Hip centering
                hip_center = (data[:, 23, :2] + data[:, 24, :2]) / 2
                data[:, :, :2] = data[:, :, :2] - hip_center[:, np.newaxis, :]

                # 均勻取樣到 32 幀
                if T >= 32:
                    indices = np.linspace(0, T - 1, 32, dtype=int)
                    sampled = data[indices]
                else:
                    pad_len = 32 - T
                    padding = np.repeat(data[-1:], pad_len, axis=0)
                    sampled = np.concatenate([data, padding], axis=0)

                # 只取手臂關節並展平
                arm_feature = sampled[:, self.arm_joints, :].flatten()
                self.support_arm_features[action_name] = arm_feature

                rospy.loginfo(f"Loaded arm feature for {action_name}: shape={arm_feature.shape}")

            except Exception as e:
                rospy.logwarn(f"Failed to load arm feature for {action_name}: {e}")

        # 計算手臂特徵之間的相似度（診斷用）
        if len(self.support_arm_features) >= 2:
            rospy.loginfo("Arm feature similarities (should be LOWER than full-body):")
            names = list(self.support_arm_features.keys())
            for i, n1 in enumerate(names):
                for n2 in names[i+1:]:
                    f1 = self.support_arm_features[n1]
                    f2 = self.support_arm_features[n2]
                    sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
                    rospy.loginfo(f"  {n1} vs {n2}: {sim:.4f}")

    def calculate_motion(self, current_keypoints):
        """計算當前幀與上一幀之間的運動量

        Args:
            current_keypoints: 當前幀的關鍵點 (33, 3)

        Returns:
            float: 運動量（所有關鍵點位移的加權平均值）
        """
        if self.last_keypoints is None:
            self.last_keypoints = current_keypoints.copy()
            return 0.0

        # 計算每個關鍵點的位移（只用 x, y 座標）
        diff = current_keypoints[:, :2] - self.last_keypoints[:, :2]

        # 計算每個關鍵點的歐氏距離
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # 使用加權平均：上半身關鍵點權重較高（更能代表動作）
        # MediaPipe 索引：0-10 是臉部，11-22 是上半身，23-32 是下半身
        weights = np.ones(33)
        weights[11:23] = 2.0  # 上半身（手臂、肩膀）權重加倍
        weights[0:11] = 0.5   # 臉部權重降低

        weighted_motion = np.average(distances, weights=weights)

        # 更新上一幀
        self.last_keypoints = current_keypoints.copy()

        return weighted_motion

    def update_motion_state(self, motion):
        """更新運動狀態

        Args:
            motion: 當前幀的運動量

        Returns:
            bool: 是否應該執行辨識
        """
        self.motion_history.append(motion)

        # 計算最近幾幀中有運動的幀數
        motion_frames = sum(1 for m in self.motion_history if m > self.motion_threshold)

        # 判斷是否正在移動
        self.is_moving = motion_frames >= self.min_motion_frames

        # 每 5 秒輸出運動狀態（除錯用）
        avg_motion = np.mean(list(self.motion_history)) if self.motion_history else 0
        rospy.loginfo_throttle(5, f"Motion: current={motion:.4f}, avg={avg_motion:.4f}, "
                              f"motion_frames={motion_frames}/{len(self.motion_history)}, "
                              f"is_moving={self.is_moving}")

        # 只在移動時執行辨識
        return self.is_moving

    def _is_static_action(self, action_name):
        """判斷動作是否為靜態動作（如 stop, idle 等）"""
        action_lower = action_name.lower()
        return any(keyword in action_lower for keyword in self.static_action_keywords)

    def recognize_one_shot(self, query_feature, exclude_static=False):
        """使用 One-Shot Learning 辨識自訂動作

        使用相對排名方法：最高相似度的動作獲得高信心度，
        其他動作根據與最高分的差距獲得較低信心度。

        Args:
            query_feature: 查詢特徵向量 (256,)
            exclude_static: 是否排除靜態動作（如 stop）

        Returns:
            tuple: (scores_dict, max_raw_similarity, similarity_margin)
                - scores_dict: {action_name: confidence_score}
                - max_raw_similarity: 最高的原始 cosine similarity
                - similarity_margin: 最高與第二高相似度的差距
        """
        if not self.custom_actions or not self.support_features:
            return {}, 0.0, 0.0

        # 計算所有動作的原始相似度
        raw_similarities = {}
        debug_info = []

        for action_name, support_feature in self.support_features.items():
            # 如果需要排除靜態動作
            if exclude_static and self._is_static_action(action_name):
                continue

            query_norm = np.linalg.norm(query_feature)
            support_norm = np.linalg.norm(support_feature)
            dot_product = np.dot(query_feature, support_feature)

            similarity = dot_product / (query_norm * support_norm + 1e-8)
            raw_similarities[action_name] = similarity
            debug_info.append(f"{action_name}: sim={similarity:.4f}")

        # 如果沒有可比較的動作，返回空
        if not raw_similarities:
            return {}, 0.0, 0.0

        # 計算最大原始相似度（用於閾值判斷）
        sorted_sims = sorted(raw_similarities.values(), reverse=True)
        max_raw_similarity = sorted_sims[0]

        # 計算相似度差距（最高 vs 第二高）
        if len(sorted_sims) >= 2:
            similarity_margin = sorted_sims[0] - sorted_sims[1]
        else:
            # 只有一個動作時，使用較大的 margin 表示確定
            similarity_margin = 1.0

        # 每 5 秒輸出除錯資訊
        mode = "(excl. static)" if exclude_static else "(all)"
        rospy.loginfo_throttle(5, f"One-Shot similarities {mode}: {', '.join(debug_info)} | max={max_raw_similarity:.4f}, margin={similarity_margin:.4f}")

        # 使用 softmax 風格的相對信心度計算
        # 這會放大微小的差異，讓最高分的動作獲得明顯更高的信心度
        similarities = np.array(list(raw_similarities.values()))
        action_names = list(raw_similarities.keys())

        # 溫度參數：越小差異越大
        # 0.01 讓模型更果斷地選擇最高分的動作
        temperature = 0.01

        # Softmax 計算
        exp_sims = np.exp((similarities - similarities.max()) / temperature)
        softmax_scores = exp_sims / exp_sims.sum()

        # 轉換為百分比
        scores = {}
        for name, score in zip(action_names, softmax_scores):
            scores[name] = float(score * 100.0)

        return scores, max_raw_similarity, similarity_margin

    def _process_trigger_mode(self, current_action, max_raw_sim, current_skeleton=None):
        """處理觸發手勢模式的狀態機

        新流程:
        IDLE -> (偵測到觸發手勢) -> RECORDING -> (3秒後自動結束) -> IDLE
                                       |
                                       v
                                  收集骨架幀，均勻取樣後辨識

        改進：
        - 偵測到 trigger 後開始錄製
        - 固定錄製 3 秒（不需要再做一次 trigger 來結束）
        - 3 秒後自動均勻取樣辨識，排除 trigger action

        Args:
            current_action: 當前識別的動作
            max_raw_sim: 最高原始相似度
            current_skeleton: 當前幀的骨架數據 (33, 3)，用於錄製
        """
        import time
        current_time = time.time()

        # 檢查是否是觸發手勢
        is_trigger = (current_action == self.trigger_action_name and
                      max_raw_sim >= self.trigger_threshold)

        # 防抖動：觸發後有冷卻時間
        if is_trigger and (current_time - self.last_trigger_time) < self.trigger_cooldown:
            is_trigger = False

        if self.trigger_state == "IDLE":
            if is_trigger:
                # 開始錄製
                self.trigger_state = "RECORDING"
                self.recording_actions = []
                self.recording_skeletons = []  # 清空骨架錄製
                self.recording_start_time = current_time
                self.last_trigger_time = current_time
                self.final_command = None
                rospy.loginfo(f"=== TRIGGER: Start recording for {self.recording_duration} seconds ===")

        elif self.trigger_state == "RECORDING":
            # 計算已錄製時間
            elapsed_time = current_time - self.recording_start_time
            remaining_time = self.recording_duration - elapsed_time

            # 每秒輸出一次進度（使用 loginfo_throttle）
            if int(elapsed_time) != int(elapsed_time - 0.033):  # 約每秒一次
                rospy.loginfo_throttle(1.0, f"Recording... {remaining_time:.1f}s remaining ({len(self.recording_skeletons)} frames)")

            # 檢查是否錄製完成（固定時間到了）
            if elapsed_time >= self.recording_duration:
                # 錄製時間到，自動結束
                self.trigger_state = "IDLE"
                self.last_trigger_time = current_time

                rospy.loginfo(f"=== TRIGGER: Recording complete ({self.recording_duration}s) ===")
                rospy.loginfo(f"=== Collected {len(self.recording_skeletons)} skeleton frames ===")

                # 使用均勻取樣方式辨識整個動作序列
                if len(self.recording_skeletons) >= 16:  # 至少需要 16 幀
                    final_action = self._recognize_with_uniform_sampling(self.recording_skeletons)

                    if final_action and final_action != "Unknown" and final_action != self.trigger_action_name:
                        self.final_command = final_action
                        self.final_command_time = current_time
                        rospy.loginfo(f"=== FINAL COMMAND: {self.final_command} ===")

                        # 發布最終指令
                        self.result_pub.publish(f"COMMAND:{self.final_command}")
                    else:
                        rospy.logwarn(f"=== TRIGGER: No valid action detected (got: {final_action}) ===")
                        self.final_command = None
                else:
                    rospy.logwarn(f"=== TRIGGER: Not enough frames ({len(self.recording_skeletons)} < 16) ===")
                    self.final_command = None

                self.recording_actions = []
                self.recording_skeletons = []

    def _recognize_with_uniform_sampling(self, skeleton_list):
        """對骨架序列進行均勻取樣後辨識

        這個方法會將收集到的所有骨架幀均勻取樣到 32 幀，
        然後進行一次完整辨識。這樣可以涵蓋整個動作，
        消除動作前後「準備姿勢」的影響。

        新增：只使用手臂關節進行相似度計算，提高動作區分度

        Args:
            skeleton_list: 骨架幀列表，每個元素是 (33, 3) 的 numpy array

        Returns:
            str: 辨識結果（動作名稱或 "Unknown"）
        """
        if not skeleton_list or len(skeleton_list) < 16:
            return "Unknown"

        try:
            # 轉換為 numpy array
            skeleton_array = np.array(skeleton_list)  # (T, 33, 3)
            T = skeleton_array.shape[0]

            rospy.loginfo(f"Uniform sampling: {T} frames -> 32 frames")

            # 均勻取樣到 32 幀
            if T >= 32:
                indices = np.linspace(0, T - 1, 32, dtype=int)
                sampled = skeleton_array[indices]
            else:
                # 如果不足 32 幀，重複最後一幀
                pad_len = 32 - T
                padding = np.repeat(skeleton_array[-1:], pad_len, axis=0)
                sampled = np.concatenate([skeleton_array, padding], axis=0)

            # 髖部中心化（與訓練時一致）
            hip_center = (sampled[:, 23, :2] + sampled[:, 24, :2]) / 2
            sampled[:, :, :2] = sampled[:, :, :2] - hip_center[:, np.newaxis, :]

            # ========== 使用手臂優先辨識 ==========
            if self.use_arm_only_recognition:
                return self._recognize_arms_only(sampled)

            # 以下是原本的全身特徵辨識（備用）
            # 提取特徵
            tensor = torch.FloatTensor(sampled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                query_feature = self.model.embedding(tensor).cpu().numpy()[0]

            # 計算與各動作的相似度
            if not self.support_features:
                return "Unknown"

            similarities = {}
            for action_name, support_feature in self.support_features.items():
                # 排除觸發動作
                if action_name == self.trigger_action_name:
                    continue

                query_norm = np.linalg.norm(query_feature)
                support_norm = np.linalg.norm(support_feature)
                sim = np.dot(query_feature, support_feature) / (query_norm * support_norm + 1e-8)
                similarities[action_name] = sim

            if not similarities:
                return "Unknown"

            # 找最高相似度
            best_action, best_sim = max(similarities.items(), key=lambda x: x[1])

            # 計算 margin
            sorted_sims = sorted(similarities.values(), reverse=True)
            margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0

            rospy.loginfo(f"Uniform sampling result: {best_action} (sim={best_sim:.4f}, margin={margin:.4f})")

            # 閾值判斷（均勻取樣更可靠，使用較寬鬆的閾值）
            if best_sim < 0.75:
                rospy.loginfo(f"Uniform sampling: below threshold ({best_sim:.4f} < 0.75)")
                return "Unknown"

            return best_action

        except Exception as e:
            rospy.logerr(f"Error in uniform sampling recognition: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown"

    def _recognize_arms_only(self, sampled_skeleton):
        """只使用手臂關節進行辨識

        直接比較手臂關節的骨架座標，不經過深度學習模型。
        這樣可以更準確地區分手勢動作。

        Args:
            sampled_skeleton: 均勻取樣後的骨架 (32, 33, 3)

        Returns:
            str: 辨識結果
        """
        # 提取手臂關節
        query_arms = sampled_skeleton[:, self.arm_joints, :].flatten()
        query_norm = np.linalg.norm(query_arms)

        if query_norm < 1e-8:
            return "Unknown"

        # 載入 support 動作的手臂特徵（需要從原始骨架計算）
        similarities = {}

        for action_name in self.support_features.keys():
            # 排除觸發動作
            if action_name == self.trigger_action_name:
                continue

            # 從 support_arm_features 取得手臂特徵
            if hasattr(self, 'support_arm_features') and action_name in self.support_arm_features:
                support_arms = self.support_arm_features[action_name]
                support_norm = np.linalg.norm(support_arms)

                if support_norm > 1e-8:
                    sim = np.dot(query_arms, support_arms) / (query_norm * support_norm)
                    similarities[action_name] = sim

        if not similarities:
            rospy.logwarn("No arm features available, falling back to full body")
            self.use_arm_only_recognition = False
            return "Unknown"

        # 找最高相似度
        best_action, best_sim = max(similarities.items(), key=lambda x: x[1])

        # 計算 margin
        sorted_sims = sorted(similarities.values(), reverse=True)
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0

        # 顯示所有相似度
        sim_str = ", ".join([f"{k}: {v:.3f}" for k, v in similarities.items()])
        rospy.loginfo(f"Arms-only similarities: {sim_str}")
        rospy.loginfo(f"Arms-only result: {best_action} (sim={best_sim:.4f}, margin={margin:.4f})")

        # 閾值判斷（手臂辨識用較低閾值）
        # 由於 support 動作之間相似度本身就高，我們只檢查最低相似度閾值
        # 不再檢查 margin，直接信任最高相似度的動作
        if best_sim < 0.65:
            rospy.loginfo(f"Arms-only: below threshold ({best_sim:.4f} < 0.65)")
            return "Unknown"

        # 直接返回最高相似度的動作（不檢查 margin）
        # 因為 trigger mode 已經排除了 circle，剩下的動作應該是用戶想做的
        return best_action

    def set_trigger_action(self, action_name, enabled=True):
        """設定觸發手勢

        Args:
            action_name: 觸發手勢的動作名稱（例如 "circle"）
            enabled: 是否啟用觸發模式
        """
        self.trigger_action_name = action_name
        self.trigger_mode_enabled = enabled
        self.trigger_state = "IDLE"
        self.recording_actions = []
        self.recording_skeletons = []
        self.final_command = None

        if enabled:
            rospy.loginfo(f"Trigger mode ENABLED with action: {action_name}")
        else:
            rospy.loginfo("Trigger mode DISABLED")

    def recognize_ntu_classification(self, skeleton_sequence):
        """使用 NTU RGB+D 分類器辨識動作

        Args:
            skeleton_sequence: 骨架序列 (64, 33, 3)

        Returns:
            dict: {class_name: confidence_score}
        """
        # OneShotActionRecognitionMediaPipe 沒有 classifier
        # NTU 分類功能暫時停用
        if self.model is None:
            return {}

        # 檢查 embedding 是否有 classifier
        if not hasattr(self.model.embedding, 'classifier') or self.model.embedding.classifier is None:
            return {}

        try:
            # 轉換為 tensor
            tensor = torch.FloatTensor(skeleton_sequence).unsqueeze(0).to(self.device)  # (1, 64, 33, 3)

            with torch.no_grad():
                # 使用 embedding 的分類器
                logits = self.model.embedding(tensor)  # (1, 60)
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
        import time
        try:
            # 將 ROS 影像訊息轉換為 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_frame = cv_image.copy()

            # 建立顯示影像副本
            display_img = cv_image.copy()

            # ========== 優先使用共享骨架數據（避免重複 MediaPipe 處理） ==========
            keypoints_array = None
            use_shared = False

            if self.use_shared_skeleton and self.shared_keypoints is not None:
                # 檢查共享數據是否新鮮（0.5秒內）
                if self.shared_keypoints_timestamp and (time.time() - self.shared_keypoints_timestamp) < 0.5:
                    keypoints_array = self.shared_keypoints.copy()
                    use_shared = True

            # 如果沒有共享數據或數據過期，回退到本地 MediaPipe 處理
            if keypoints_array is None:
                # 轉換為 RGB 供 MediaPipe 處理
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

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
                    for lm in landmarks:
                        # 使用正規化座標 (0-1)，與訓練數據一致
                        keypoints.append([lm.x, lm.y, lm.visibility])

                    keypoints_array = np.array(keypoints)  # (33, 3)

            # 顯示模式標記
            mode_text = "[Shared]" if use_shared else "[Local MP]"
            cv2.putText(display_img, mode_text, (10, display_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if use_shared else (0, 165, 255), 1)

            # 如果有骨架數據（無論來源），進行辨識
            if keypoints_array is not None:
                # 計算運動量並更新狀態
                motion = self.calculate_motion(keypoints_array)
                should_recognize = self.update_motion_state(motion)

                # 加入緩衝區
                self.skeleton_buffer.append(keypoints_array)

                # ========== Trigger Mode: 每幀都收集骨架 ==========
                if (self.trigger_mode_enabled and
                    self.trigger_state == "RECORDING"):
                    self.recording_skeletons.append(keypoints_array.copy())

                # 執行辨識
                self.frame_count += 1
                if len(self.skeleton_buffer) >= 32 and self.frame_count % self.recognition_interval == 0:
                    skeleton_sequence = np.array(list(self.skeleton_buffer))  # (32, 33, 3)

                    # 關鍵步驟：髖部中心化（與訓練時的預處理一致）
                    # MediaPipe: 23=左髖, 24=右髖
                    skeleton_norm = skeleton_sequence.copy()
                    hip_center = (skeleton_norm[:, 23, :2] + skeleton_norm[:, 24, :2]) / 2
                    skeleton_norm[:, :, :2] = skeleton_norm[:, :, :2] - hip_center[:, np.newaxis, :]

                    # 除錯：顯示輸入數據範圍（每 5 秒）
                    rospy.loginfo_throttle(5, f"Input (after hip-centering): x=[{skeleton_norm[...,0].min():.3f}, {skeleton_norm[...,0].max():.3f}], "
                                          f"y=[{skeleton_norm[...,1].min():.3f}, {skeleton_norm[...,1].max():.3f}]")

                    # 除錯：顯示手腕相對於肩膀的位置（判斷手是否舉起）
                    # 取最後一幀
                    last_frame = skeleton_norm[-1]
                    left_wrist_y = last_frame[15, 1]  # 左手腕
                    right_wrist_y = last_frame[16, 1]  # 右手腕
                    left_shoulder_y = last_frame[11, 1]  # 左肩
                    right_shoulder_y = last_frame[12, 1]  # 右肩
                    left_diff = left_wrist_y - left_shoulder_y  # 負值=舉起
                    right_diff = right_wrist_y - right_shoulder_y
                    rospy.loginfo_throttle(5, f"Hand position: L_wrist-shoulder={left_diff:.3f}, R_wrist-shoulder={right_diff:.3f} (negative=raised)")

                    # 執行雙模式辨識
                    all_scores = {}
                    max_raw_sim = 0.0

                    # 1. One-Shot Learning（自訂動作）
                    if self.custom_actions and self.support_features:
                        # 提取特徵向量（使用 model.embedding，與 model_manager.py 一致）
                        tensor = torch.FloatTensor(skeleton_norm).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            # 使用 embedding 提取特徵（與 model_manager.py 相同方式）
                            query_feature = self.model.embedding(tensor)  # (1, 256)
                            query_feature = query_feature.cpu().numpy()[0]  # (256,)

                        # 除錯：顯示特徵統計
                        rospy.loginfo_throttle(5, f"Query feature stats: mean={query_feature.mean():.4f}, std={query_feature.std():.4f}, norm={np.linalg.norm(query_feature):.4f}")

                        # 純模型辨識：讓所有動作公平競爭
                        # 返回 softmax 信心度、最大原始相似度、相似度差距
                        custom_scores, max_raw_sim, sim_margin = self.recognize_one_shot(query_feature, exclude_static=False)
                        all_scores.update(custom_scores)

                    # 2. NTU RGB+D Classification
                    if self.use_ntu_classification:
                        ntu_scores = self.recognize_ntu_classification(skeleton_sequence)
                        all_scores.update(ntu_scores)

                    # 更新結果
                    if all_scores:
                        self.all_scores = all_scores
                        self.max_raw_similarity = max_raw_sim

                        # 雙重閾值判斷：
                        # 1. 最高相似度必須高於閾值
                        # 2. 最高與第二高的差距必須大於 margin
                        is_below_threshold = max_raw_sim < self.similarity_threshold
                        is_margin_too_small = sim_margin < self.similarity_margin and len(self.custom_actions) > 1

                        # 儲存 margin 供顯示用
                        self.current_margin = sim_margin

                        if is_below_threshold:
                            self.current_action = "Unknown"
                            self.current_confidence = max_raw_sim * 100
                            rospy.loginfo_throttle(5, f"Below threshold ({self.similarity_threshold}): max_sim={max_raw_sim:.4f} -> Unknown")
                        elif is_margin_too_small:
                            self.current_action = "Unknown"
                            self.current_confidence = max_raw_sim * 100
                            rospy.loginfo_throttle(5, f"Margin too small ({self.similarity_margin}): margin={sim_margin:.4f} -> Unknown (ambiguous)")
                        else:
                            # 找出最高分數的動作
                            best_action = max(all_scores.items(), key=lambda x: x[1])
                            self.current_action = best_action[0]
                            self.current_confidence = best_action[1]

                        # ========== 觸發手勢模式處理 ==========
                        if self.trigger_mode_enabled and self.trigger_action_name:
                            # 傳入當前骨架幀，用於錄製期間收集
                            self._process_trigger_mode(self.current_action, max_raw_sim, keypoints_array)

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
        import time
        h, w = image.shape[:2]

        # 計算右下角位置
        box_width = 400
        box_height = 300  # 增加高度以顯示觸發模式狀態和 margin
        x_start = w - box_width - 10
        y_start = h - box_height - 10
        x_end = w - 10
        y_end = h - 10

        # 背景矩形（右下角）
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

        # 當前辨識結果（調整位置）
        text_x = x_start + 10
        text_y = y_start + 30

        # ========== 觸發模式狀態顯示 ==========
        if self.trigger_mode_enabled:
            # 顯示觸發模式狀態
            if self.trigger_state == "IDLE":
                state_text = f"[TRIGGER MODE] Waiting for '{self.trigger_action_name}'"
                state_color = (255, 255, 0)  # 黃色
                cv2.putText(image, state_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                text_y += 25
            else:  # RECORDING
                elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
                remaining = max(0, self.recording_duration - elapsed)
                num_frames = len(self.recording_skeletons)

                # 錄製中的狀態顯示（更醒目）
                state_text = f"[RECORDING] {remaining:.1f}s left"
                state_color = (0, 255, 255)  # 青色
                cv2.putText(image, state_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                text_y += 25

                # 顯示收集的幀數
                frames_text = f"Frames: {num_frames}"
                cv2.putText(image, frames_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                text_y += 20

                # 繪製進度條
                progress = elapsed / self.recording_duration
                bar_width = box_width - 40
                bar_height = 15
                bar_x = text_x
                bar_y = text_y

                # 背景
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # 進度
                progress_width = int(bar_width * min(progress, 1.0))
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 255), -1)
                # 邊框
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

                text_y += 25

            # 顯示最終指令（如果有）
            if self.final_command and self.final_command_time:
                elapsed_since_command = time.time() - self.final_command_time
                if elapsed_since_command < 5.0:  # 顯示 5 秒
                    cmd_text = f">>> COMMAND: {self.final_command} <<<"
                    cv2.putText(image, cmd_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    text_y += 30

        # 根據辨識結果選擇顏色
        action_color = (0, 0, 255) if self.current_action == "Unknown" else (0, 255, 0)

        cv2.putText(
            image,
            f"Action: {self.current_action}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            action_color,
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

        # 顯示原始相似度和閾值（幫助調試）
        sim_color = (0, 255, 0) if self.max_raw_similarity >= self.similarity_threshold else (0, 0, 255)
        cv2.putText(
            image,
            f"MaxSim: {self.max_raw_similarity:.3f} (thr: {self.similarity_threshold})",
            (text_x, text_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            sim_color,
            1
        )

        # 顯示相似度差距 (Margin)（幫助調試）
        current_margin = getattr(self, 'current_margin', 0.0)
        margin_color = (0, 255, 0) if current_margin >= self.similarity_margin else (0, 0, 255)
        cv2.putText(
            image,
            f"Margin: {current_margin:.3f} (thr: {self.similarity_margin})",
            (text_x, text_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            margin_color,
            1
        )

        # 顯示 Top 5 分數
        if self.all_scores:
            sorted_scores = sorted(self.all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            cv2.putText(image, "Top 5:", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            for i, (action, score) in enumerate(sorted_scores):
                y_pos = text_y + 125 + i * 22
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
