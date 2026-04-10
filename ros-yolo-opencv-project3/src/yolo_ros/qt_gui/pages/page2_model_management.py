#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二個頁面：Model Management - Select Model and Actions
功能:
- 左側：訓練模型選擇
- 中間：Action Management
- 右側：Skeleton Preview

Author: Claude AI Assistant
Date: 2025-11-23
"""

import os
import glob
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QListWidget, QLineEdit, QMessageBox,
                              QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from widgets.skeleton_widget import SkeletonPreviewWidget
from utils.model_manager import ModelManager


class Page2ModelManagement(QWidget):
    """第二個頁面：Model Management - Select Model and Actions"""

    def __init__(self):
        super().__init__()

        # 模型管理器
        self.model_manager = ModelManager()

        # 當前選擇的模型
        self.current_model = None

        # Current Action List
        self.current_actions = []

        # actionset 路徑
        self.actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"

        # 建立 UI
        self._create_ui()

        # 載入Available Models列表
        self._load_available_models()

        # 訂閱 actionset 更新通知
        self.actionset_update_sub = rospy.Subscriber(
            '/actionset/updated',
            String,
            self._on_actionset_updated,
            queue_size=1
        )

        rospy.loginfo("Page 2 (Model Management) initialized")

    def _create_ui(self):
        """建立使用者介面"""
        layout = QVBoxLayout(self)

        # 標題
        title = QLabel("Model Management - Select Model and Actions")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 主要內容區域（三欄布局）
        content_layout = QHBoxLayout()

        # 左側：模型選擇
        self._create_model_selection_panel(content_layout)

        # 中間：動作管理
        self._create_action_management_panel(content_layout)

        # 右側：Skeleton Preview
        self._create_action_preview_panel(content_layout)

        layout.addLayout(content_layout)

    def _create_model_selection_panel(self, parent_layout):
        """建立模型選擇面板"""
        group_box = QGroupBox("Available Models")
        group_box.setFont(QFont("Arial", 12, QFont.Bold))
        layout = QVBoxLayout()

        # 模型列表
        self.model_list = QListWidget()
        self.model_list.setFont(QFont("Arial", 10))
        self.model_list.itemClicked.connect(self._on_model_selected)
        layout.addWidget(self.model_list)

        # 模型資訊標籤
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.model_info_label)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def _create_action_management_panel(self, parent_layout):
        """建立動作管理面板"""
        group_box = QGroupBox("Action Management")
        group_box.setFont(QFont("Arial", 12, QFont.Bold))
        layout = QVBoxLayout()

        # 當前辨識動作列表
        current_label = QLabel("Current Action List:")
        current_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(current_label)

        self.current_action_list = QListWidget()
        self.current_action_list.setFont(QFont("Arial", 10))
        self.current_action_list.itemClicked.connect(self._on_action_selected)
        layout.addWidget(self.current_action_list)

        # 操作按鈕
        button_layout = QHBoxLayout()

        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all_actions)

        self.btn_remove_action = QPushButton("Remove Action")
        self.btn_remove_action.clicked.connect(self._remove_selected_action)

        button_layout.addWidget(self.btn_select_all)
        button_layout.addWidget(self.btn_remove_action)
        layout.addLayout(button_layout)

        # 新增動作區域
        add_label = QLabel("Add actions from actionset:")
        add_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(add_label)

        self.actionset_list = QListWidget()
        self.actionset_list.setFont(QFont("Arial", 10))
        self.actionset_list.setSelectionMode(QListWidget.MultiSelection)
        self.actionset_list.itemClicked.connect(self._on_actionset_item_clicked)
        layout.addWidget(self.actionset_list)

        # 新增動作名稱輸入
        name_layout = QHBoxLayout()
        name_label = QLabel("Action Name:")
        self.action_name_input = QLineEdit()
        self.action_name_input.setPlaceholderText("Enter new action name")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.action_name_input)
        layout.addLayout(name_layout)

        # 新增按鈕
        self.btn_add_action = QPushButton("Add Action")
        self.btn_add_action.clicked.connect(self._add_custom_action)
        layout.addWidget(self.btn_add_action)

        # ========== 觸發手勢設定 ==========
        trigger_label = QLabel("Trigger Gesture Mode:")
        trigger_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(trigger_label)

        # 觸發模式開關
        self.trigger_checkbox = QCheckBox("Enable Trigger Mode")
        self.trigger_checkbox.setChecked(False)
        self.trigger_checkbox.stateChanged.connect(self._on_trigger_mode_changed)
        layout.addWidget(self.trigger_checkbox)

        # 觸發動作選擇提示
        self.trigger_info_label = QLabel("Select an action from 'Current Action List' as trigger")
        self.trigger_info_label.setFont(QFont("Arial", 9))
        self.trigger_info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.trigger_info_label)

        # 設為觸發動作按鈕
        self.btn_set_trigger = QPushButton("Set as Trigger Action")
        self.btn_set_trigger.setEnabled(False)
        self.btn_set_trigger.clicked.connect(self._set_trigger_action)
        layout.addWidget(self.btn_set_trigger)

        # 當前觸發動作顯示
        self.current_trigger_label = QLabel("Current Trigger: None")
        self.current_trigger_label.setFont(QFont("Arial", 9))
        self.current_trigger_label.setStyleSheet("color: blue;")
        layout.addWidget(self.current_trigger_label)

        # 觸發動作發布者
        self.trigger_pub = rospy.Publisher(
            '/recognition_display/set_trigger',
            String,
            queue_size=1
        )

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def _create_action_preview_panel(self, parent_layout):
        """建立Skeleton Preview面板"""
        group_box = QGroupBox("Skeleton Preview")
        group_box.setFont(QFont("Arial", 12, QFont.Bold))
        layout = QVBoxLayout()

        # 預覽元件
        self.skeleton_preview = SkeletonPreviewWidget()
        layout.addWidget(self.skeleton_preview)

        # 播放控制
        control_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Pause")

        self.btn_play.clicked.connect(self.skeleton_preview.play)
        self.btn_stop.clicked.connect(self.skeleton_preview.stop)

        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_stop)
        layout.addLayout(control_layout)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def _load_available_models(self):
        """載入可用的模型列表"""
        self.model_list.clear()

        models = self.model_manager.get_available_models()

        for model_name, model_info in models.items():
            self.model_list.addItem(f"{model_name} ({model_info['epoch']} epochs)")

        # 載入 actionset 列表
        self._load_actionset_list()

    def _load_actionset_list(self):
        """載入 actionset 資料夾列表"""
        self.actionset_list.clear()

        if not os.path.exists(self.actionset_dir):
            return

        # 列出所有子資料夾
        action_dirs = [d for d in os.listdir(self.actionset_dir)
                       if os.path.isdir(os.path.join(self.actionset_dir, d))]

        for action_dir in sorted(action_dirs):
            self.actionset_list.addItem(action_dir)

    def _on_model_selected(self, item):
        """模型選擇事件"""
        model_name = item.text().split(" (")[0]

        # 載入模型
        success = self.model_manager.load_model(model_name)

        if success:
            self.current_model = model_name
            model_info = self.model_manager.get_model_info(model_name)

            # 更新模型資訊
            info_text = f"Model: {model_name}\n"
            info_text += f"Epochs: {model_info['epoch']}\n"
            info_text += f"Accuracy: {model_info.get('accuracy', 'N/A')}\n"
            info_text += f"Classes: {model_info.get('num_classes', 'N/A')}"

            self.model_info_label.setText(info_text)

            # 更新動作列表
            self._update_current_actions()

            rospy.loginfo(f"Loaded model: {model_name}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to load model: {model_name}")

    def _update_current_actions(self):
        """更新目前辨識的動作列表"""
        self.current_action_list.clear()
        self.current_actions = self.model_manager.get_current_actions()

        for action in self.current_actions:
            self.current_action_list.addItem(action)

    def _select_all_actions(self):
        """全選所有動作"""
        for i in range(self.current_action_list.count()):
            self.current_action_list.item(i).setSelected(True)

    def _remove_selected_action(self):
        """移除選定的動作"""
        selected_items = self.current_action_list.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select actions to remove")
            return

        for item in selected_items:
            action_name = item.text()
            self.model_manager.remove_action(action_name)
            rospy.loginfo(f"Removed action: {action_name}")

        self._update_current_actions()

    def _on_action_selected(self, item):
        """動作選擇事件（預覽）"""
        action_name = item.text()

        # 載入並預覽該動作的骨架資料
        action_data = self.model_manager.get_action_data(action_name)

        if action_data is not None:
            self.skeleton_preview.load_skeleton_sequence(action_data)
            rospy.loginfo(f"Previewing action: {action_name}")
        else:
            # 嘗試從 actionset 載入
            action_path = os.path.join(self.actionset_dir, action_name)

            # MediaPipe 33 格式骨架序列
            skeleton_file_mp33 = os.path.join(action_path, f"{action_name}_skeleton_sequence.npy")
            # 單幀骨架檔案
            skeleton_files_single = glob.glob(os.path.join(action_path, f"{action_name}_*_skeleton.npy"))

            if os.path.exists(skeleton_file_mp33):
                skeleton_seq = np.load(skeleton_file_mp33)
                # MediaPipe 33 點格式
                if skeleton_seq.shape[-2] == 33 and skeleton_seq.shape[-1] >= 3:
                    self.skeleton_preview.load_skeleton_sequence(skeleton_seq)
                    rospy.loginfo(f"Loaded MediaPipe33 skeleton: {skeleton_file_mp33} - Shape: {skeleton_seq.shape}")
                else:
                    rospy.logwarn(f"Cannot preview {action_name}: Invalid format {skeleton_seq.shape}. Expected (..., 33, 3+)")
            elif skeleton_files_single:
                # 載入單幀骨架並堆疊
                frames = []
                for f in sorted(skeleton_files_single):
                    try:
                        frame = np.load(f)
                        if frame.shape[-2] == 33:
                            frames.append(frame)
                        else:
                            rospy.logwarn(f"Skipping {f}: Not MediaPipe33 format")
                    except Exception as e:
                        rospy.logwarn(f"Failed to load {f}: {e}")
                if frames:
                    skeleton_seq = np.stack(frames, axis=0)
                    self.skeleton_preview.load_skeleton_sequence(skeleton_seq)
                    rospy.loginfo(f"Loaded {len(frames)} single frames for preview - Shape: {skeleton_seq.shape}")
                else:
                    rospy.logwarn(f"Cannot preview {action_name}: No valid MediaPipe33 skeleton frames found")
            else:
                rospy.logwarn(f"Cannot preview {action_name}: No skeleton sequence file found")

    def _add_custom_action(self):
        """新增自訂動作"""
        selected_items = self.actionset_list.selectedItems()
        custom_name = self.action_name_input.text().strip()

        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select at least one action from actionset")
            return

        if not custom_name:
            QMessageBox.warning(self, "Error", "Please enter action name")
            return

        # 收集選定的動作資料
        action_samples = []
        for item in selected_items:
            action_dir = item.text()
            action_path = os.path.join(self.actionset_dir, action_dir)

            # MediaPipe 33 格式的骨架序列
            skeleton_file_mp33 = os.path.join(action_path, f"{action_dir}_skeleton_sequence.npy")
            # 單幀骨架檔案
            skeleton_files_single = glob.glob(os.path.join(action_path, f"{action_dir}_*_skeleton.npy"))

            if os.path.exists(skeleton_file_mp33):
                skeleton_seq = np.load(skeleton_file_mp33)
                # MediaPipe 33 點格式
                if skeleton_seq.shape[-2] == 33 and skeleton_seq.shape[-1] >= 3:
                    # 只取 x, y, z（前 3 個通道）
                    skeleton_seq = skeleton_seq[..., :3]
                    rospy.loginfo(f"Loaded MediaPipe33: {skeleton_file_mp33} - Shape: {skeleton_seq.shape}")
                    action_samples.append(skeleton_seq)
                else:
                    rospy.logwarn(f"Skipping {action_dir}: Invalid format {skeleton_seq.shape}. Expected (..., 33, 3+)")
            elif skeleton_files_single:
                # 載入單幀骨架並堆疊（適用於靜態姿勢）
                frames = []
                for f in sorted(skeleton_files_single):
                    try:
                        frame = np.load(f)
                        if frame.shape[-2] == 33:
                            frame = frame[..., :3]  # 只取 x, y, z
                            frames.append(frame)
                        else:
                            rospy.logwarn(f"Skipping {f}: Not MediaPipe33 format")
                    except Exception as e:
                        rospy.logwarn(f"Failed to load {f}: {e}")
                if frames:
                    skeleton_seq = np.stack(frames, axis=0) if len(frames) > 1 else frames[0][np.newaxis, ...]
                    rospy.loginfo(f"Loaded {len(frames)} single frames: {action_path} - Shape: {skeleton_seq.shape}")
                    action_samples.append(skeleton_seq)
                else:
                    rospy.logwarn(f"Skipping {action_dir}: No valid MediaPipe33 skeleton frames found")
            else:
                rospy.logwarn(f"Skipping {action_dir}: No skeleton files found")

        if len(action_samples) == 0:
            QMessageBox.warning(self, "Error", "No valid skeleton data found.\n\nPlease ensure actionset contains MediaPipe 33-point skeleton files.")
            return

        # 新增到模型
        success = self.model_manager.add_custom_action(custom_name, action_samples)

        if success:
            rospy.loginfo(f"Added custom action: {custom_name} ({len(action_samples)} samples)")
            QMessageBox.information(self, "Success",
                                    f"Action added: {custom_name}\nSamples: {len(action_samples)}")

            # 更新動作列表
            self._update_current_actions()

            # 通知 recognition_display_node 重新載入動作
            self._notify_recognition_node()

            # 清空輸入
            self.action_name_input.clear()
            self.actionset_list.clearSelection()
        else:
            QMessageBox.warning(self, "Error", "Failed to add action")

    def _notify_recognition_node(self):
        """通知辨識節點重新載入動作列表"""
        try:
            rospy.wait_for_service('/recognition_display/load_actions', timeout=1.0)
            load_actions = rospy.ServiceProxy('/recognition_display/load_actions', Trigger)
            response = load_actions()
            if response.success:
                rospy.loginfo(f"Recognition node updated: {response.message}")
            else:
                rospy.logwarn(f"Failed to update recognition node: {response.message}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"Recognition node service call failed: {e}")
        except rospy.ROSException:
            rospy.loginfo("Recognition node not available (this is normal if not running)")

    @pyqtSlot()
    def on_stop(self):
        """接收 STOP 信號"""
        rospy.logwarn("Page 2: Received STOP signal")
        self.skeleton_preview.stop()

    @pyqtSlot()
    def on_initial(self):
        """接收 Initial 信號"""
        rospy.loginfo("Page 2: Received Initial signal")
        self.action_name_input.clear()
        self.actionset_list.clearSelection()
        self.skeleton_preview.stop()

    def _on_actionset_updated(self, msg):
        """處理 actionset 更新通知"""
        rospy.loginfo("Actionset updated, refreshing actionset list...")
        self._load_actionset_list()

    def _on_actionset_item_clicked(self, item):
        """actionset 項目點擊事件 - 預覽骨架動畫

        當使用者點擊 actionset 列表中的項目時，
        載入該動作的 .npy 骨架序列並在右側預覽區域播放動畫。
        """
        action_dir = item.text()
        action_path = os.path.join(self.actionset_dir, action_dir)

        # 尋找骨架序列檔案（優先找 MediaPipe 33 格式）
        skeleton_file = None

        # 方式 1: 直接找 *_skeleton_sequence.npy（排除 coco17 和 backup）
        for f in os.listdir(action_path):
            if f.endswith('skeleton_sequence.npy') and 'coco17' not in f and 'backup' not in f:
                skeleton_file = os.path.join(action_path, f)
                break

        if skeleton_file and os.path.exists(skeleton_file):
            try:
                skeleton_seq = np.load(skeleton_file)

                # 驗證是否為 MediaPipe 33 點格式
                if len(skeleton_seq.shape) >= 2 and skeleton_seq.shape[-2] == 33:
                    self.skeleton_preview.load_skeleton_sequence(skeleton_seq)
                    # 自動開始播放
                    self.skeleton_preview.play()
                    rospy.loginfo(f"Preview actionset: {action_dir} - Shape: {skeleton_seq.shape}, Frames: {skeleton_seq.shape[0]}")
                else:
                    rospy.logwarn(f"Invalid skeleton format for {action_dir}: {skeleton_seq.shape}")
                    self.skeleton_preview.setText(f"無法預覽: {action_dir}\n格式不符 (需要 33 關節點)")
            except Exception as e:
                rospy.logerr(f"Failed to load skeleton for {action_dir}: {e}")
                self.skeleton_preview.setText(f"載入失敗: {action_dir}\n{str(e)}")
        else:
            rospy.logwarn(f"No skeleton file found for {action_dir}")
            self.skeleton_preview.setText(f"找不到骨架檔案:\n{action_dir}")

    def _on_trigger_mode_changed(self, state):
        """觸發模式開關變更事件"""
        enabled = (state == Qt.Checked)
        self.btn_set_trigger.setEnabled(enabled)

        if not enabled:
            # 停用觸發模式
            self.trigger_pub.publish(String(data="DISABLE"))
            self.current_trigger_label.setText("Current Trigger: None (Disabled)")
            self.current_trigger_label.setStyleSheet("color: gray;")
            rospy.loginfo("Trigger mode disabled")
        else:
            self.current_trigger_label.setText("Current Trigger: None (Select an action)")
            self.current_trigger_label.setStyleSheet("color: orange;")

    def _set_trigger_action(self):
        """將選中的動作設為觸發動作"""
        selected_items = self.current_action_list.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Error",
                              "Please select an action from 'Current Action List' to set as trigger")
            return

        # 只取第一個選中的動作
        action_name = selected_items[0].text()

        # 發送觸發動作設定
        self.trigger_pub.publish(String(data=action_name))

        # 更新顯示
        self.current_trigger_label.setText(f"Current Trigger: {action_name}")
        self.current_trigger_label.setStyleSheet("color: green; font-weight: bold;")

        rospy.loginfo(f"Set trigger action: {action_name}")

        QMessageBox.information(self, "Trigger Mode",
                               f"Trigger action set to: {action_name}\n\n"
                               f"How to use:\n"
                               f"1. Perform '{action_name}' gesture to START recording\n"
                               f"2. Perform your command action (forward/stop/right/etc.)\n"
                               f"3. Perform '{action_name}' gesture again to END\n"
                               f"4. The most frequent action will be output as command")
