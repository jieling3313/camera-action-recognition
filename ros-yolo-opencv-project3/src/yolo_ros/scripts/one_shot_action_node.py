#!/usr/bin/env python3
"""
One-Shot 動作辨識 ROS 節點

此節點使用基於骨架的 one-shot 學習與多尺度時空匹配
來執行即時動作辨識。

訂閱的 Topics：
    /camera/color/image_raw (sensor_msgs/Image)

發布的 Topics：
    /action_recognition/result (std_msgs/String)
    /action_recognition/score (std_msgs/Float32)
    /action_recognition/annotated_image (sensor_msgs/Image)

服務：
    /record_support_action (std_srvs/Trigger)
    /load_support_set (std_srvs/Trigger)
"""

import os
import sys
import threading
import numpy as np
import cv2
import torch

import rospy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge

# 匯入本地模組
from skeleton_extractor import SkeletonExtractor, SkeletonBuffer
from skeleton_model import OneShotActionRecognition, preprocess_skeleton


class OneShotActionNode:
    """One-Shot 動作辨識的 ROS 節點。"""

    def __init__(self):
        rospy.init_node('one_shot_action_node', anonymous=True)

        # 參數
        self.buffer_size = rospy.get_param('~buffer_size', 64)
        self.recognition_interval = rospy.get_param('~recognition_interval', 30)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)
        self.support_set_dir = rospy.get_param('~support_set_dir',
            os.path.expanduser('~/catkin_ws/src/yolo_ros/support_sets'))
        self.pose_model = rospy.get_param('~pose_model', 'yolov8m-pose.pt')
        self.device = rospy.get_param('~device', 'cpu')

        # 建立支持集目錄
        os.makedirs(self.support_set_dir, exist_ok=True)

        # 初始化元件
        rospy.loginfo("Initializing skeleton extractor...")
        self.extractor = SkeletonExtractor(
            model_path=self.pose_model,
            device=self.device,
            conf_threshold=self.confidence_threshold
        )

        rospy.loginfo("Initializing action recognition model...")
        self.model = OneShotActionRecognition(in_channels=3, base_channels=64)
        self.model.eval()
        self.model.to(self.device)

        # 骨架緩衝區
        self.buffer = SkeletonBuffer(buffer_size=self.buffer_size)

        # 支持集
        self.support_set = []  # (tensor, label) 元組列表
        self.load_support_set()

        # 狀態
        self.bridge = CvBridge()
        self.frame_count = 0
        self.current_action = "Unknown"
        self.current_score = 0.0
        self.is_recording = False
        self.recording_label = ""
        self.recording_buffer = SkeletonBuffer(buffer_size=self.buffer_size)

        # 執行緒鎖
        self.lock = threading.Lock()

        # 發布者
        self.result_pub = rospy.Publisher(
            '/action_recognition/result', String, queue_size=10
        )
        self.score_pub = rospy.Publisher(
            '/action_recognition/score', Float32, queue_size=10
        )
        self.image_pub = rospy.Publisher(
            '/action_recognition/annotated_image', Image, queue_size=10
        )

        # 訂閱者
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_callback, queue_size=1
        )

        # 服務
        self.record_srv = rospy.Service(
            '/start_recording', Trigger, self.start_recording_callback
        )
        self.stop_record_srv = rospy.Service(
            '/stop_recording', Trigger, self.stop_recording_callback
        )
        self.reload_srv = rospy.Service(
            '/reload_support_set', Trigger, self.reload_support_callback
        )

        rospy.loginfo("One-Shot Action Recognition node initialized!")
        rospy.loginfo(f"Support set directory: {self.support_set_dir}")
        rospy.loginfo(f"Loaded {len(self.support_set)} support actions")

    def load_support_set(self):
        """從磁碟載入預錄的支持序列。"""
        self.support_set = []

        if not os.path.exists(self.support_set_dir):
            rospy.logwarn(f"Support set directory not found: {self.support_set_dir}")
            return

        for filename in os.listdir(self.support_set_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.support_set_dir, filename)
                try:
                    data = np.load(filepath)
                    label = os.path.splitext(filename)[0]

                    # 預處理
                    tensor = preprocess_skeleton(data, self.buffer_size)
                    tensor = tensor.to(self.device)

                    self.support_set.append((tensor, label))
                    rospy.loginfo(f"Loaded support action: {label}")
                except Exception as e:
                    rospy.logerr(f"Failed to load {filename}: {e}")

    def image_callback(self, msg):
        """處理輸入的相機影像。"""
        try:
            # 將 ROS Image 轉換為 OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 提取骨架
            keypoints, bbox = self.extractor.extract(cv_image, normalize=True)

            with self.lock:
                # 加入緩衝區
                self.buffer.add(keypoints)

                # 處理錄製
                if self.is_recording:
                    self.recording_buffer.add(keypoints)

                self.frame_count += 1

                # 定期執行辨識
                if (self.frame_count % self.recognition_interval == 0 and
                    self.buffer.is_full() and
                    len(self.support_set) > 0):

                    self.recognize_action()

            # 繪製視覺化
            annotated = self.draw_visualization(cv_image, keypoints, bbox)

            # 發布標註影像
            img_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            self.image_pub.publish(img_msg)

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def recognize_action(self):
        """對緩衝的序列執行動作辨識。"""
        try:
            # 從緩衝區取得序列
            sequence = self.buffer.get_sequence()

            # 預處理
            query = preprocess_skeleton(sequence, self.buffer_size)
            query = query.to(self.device)

            # 執行推論
            with torch.no_grad():
                predicted, scores = self.model(query, self.support_set)

            # 更新狀態
            self.current_action = predicted
            self.current_score = scores[0][0] if scores else 0.0

            # 發布結果
            self.result_pub.publish(String(self.current_action))
            self.score_pub.publish(Float32(self.current_score))

            rospy.loginfo(f"Recognized action: {self.current_action} "
                         f"(score: {self.current_score:.3f})")

        except Exception as e:
            rospy.logerr(f"Action recognition error: {e}")

    def draw_visualization(self, image, keypoints, bbox):
        """在影像上繪製骨架和辨識結果。"""
        img = image.copy()

        # 繪製骨架
        if keypoints is not None:
            img = self.extractor.draw_skeleton(
                img, keypoints, bbox,
                color=(0, 255, 0), normalized=True
            )

        # 繪製動作結果
        h, w = img.shape[:2]

        # 文字背景
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1)

        # 動作標籤
        text = f"Action: {self.current_action}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # 分數
        score_text = f"Score: {self.current_score:.3f}"
        cv2.putText(img, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # 緩衝區狀態
        buffer_text = f"Buffer: {len(self.buffer.buffer)}/{self.buffer_size}"
        cv2.putText(img, buffer_text, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # 錄製指示器
        if self.is_recording:
            cv2.circle(img, (w - 30, 60), 10, (0, 0, 255), -1)
            rec_text = f"Recording: {self.recording_label}"
            cv2.putText(img, rec_text, (w - 200, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        return img

    def start_recording_callback(self, req):
        """開始錄製新動作的服務回呼。"""
        with self.lock:
            if self.is_recording:
                return TriggerResponse(
                    success=False,
                    message="Already recording. Please stop current recording first."
                )

            # 從參數取得標籤或產生一個
            self.recording_label = rospy.get_param('~recording_label', 'action')
            self.recording_buffer.clear()
            self.is_recording = True

            return TriggerResponse(
                success=True,
                message=f"Started recording action: {self.recording_label}"
            )

    def stop_recording_callback(self, req):
        """停止錄製並儲存動作的服務回呼。"""
        with self.lock:
            if not self.is_recording:
                return TriggerResponse(
                    success=False,
                    message="Not currently recording."
                )

            self.is_recording = False

            # 檢查是否有足夠的幀
            if not self.recording_buffer.is_full():
                return TriggerResponse(
                    success=False,
                    message=f"Insufficient frames. Got {len(self.recording_buffer.buffer)}, "
                            f"need {self.buffer_size}"
                )

            # 儲存錄製
            sequence = self.recording_buffer.get_sequence()
            filepath = os.path.join(self.support_set_dir,
                                    f"{self.recording_label}.npy")
            np.save(filepath, sequence)

            # 加入支持集
            tensor = preprocess_skeleton(sequence, self.buffer_size)
            tensor = tensor.to(self.device)
            self.support_set.append((tensor, self.recording_label))

            return TriggerResponse(
                success=True,
                message=f"Saved action '{self.recording_label}' to {filepath}"
            )

    def reload_support_callback(self, req):
        """從磁碟重新載入支持集的服務回呼。"""
        with self.lock:
            self.load_support_set()
            return TriggerResponse(
                success=True,
                message=f"Reloaded {len(self.support_set)} support actions"
            )

    def run(self):
        """執行節點。"""
        rospy.spin()


def main():
    try:
        node = OneShotActionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
