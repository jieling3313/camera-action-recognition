#!/usr/bin/env python3
"""
錄製支持集動作的工具腳本。

使用方式：
    rosrun yolo_ros record_support_set.py --action waving
    rosrun yolo_ros record_support_set.py --action falling --frames 64
"""

import os
import argparse
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from skeleton_extractor import SkeletonExtractor, SkeletonBuffer


class SupportSetRecorder:
    """錄製支持集的骨架序列。"""

    def __init__(self, action_name, num_frames=64, output_dir=None):
        rospy.init_node('support_set_recorder', anonymous=True)

        self.action_name = action_name
        self.num_frames = num_frames

        # 輸出目錄
        if output_dir is None:
            self.output_dir = os.path.expanduser(
                '~/catkin_ws/src/yolo_ros/support_sets'
            )
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化元件
        self.extractor = SkeletonExtractor(
            model_path='yolov8m-pose.pt',
            device='cpu',
            conf_threshold=0.5
        )
        self.buffer = SkeletonBuffer(buffer_size=num_frames)
        self.bridge = CvBridge()

        # 狀態
        self.is_recording = False
        self.frames_recorded = 0

        # 訂閱者
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_callback, queue_size=1
        )

        rospy.loginfo(f"Support set recorder initialized")
        rospy.loginfo(f"Action: {action_name}")
        rospy.loginfo(f"Frames to record: {num_frames}")
        rospy.loginfo("Press SPACE to start recording, 'q' to quit")

    def image_callback(self, msg):
        """處理相機影像。"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 提取骨架
            keypoints, bbox = self.extractor.extract(cv_image, normalize=True)

            if self.is_recording and keypoints is not None:
                self.buffer.add(keypoints)
                self.frames_recorded = len(self.buffer.buffer)

            # 繪製視覺化
            img = self.draw_ui(cv_image, keypoints, bbox)

            cv2.imshow('Support Set Recorder', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # 空白鍵開始/停止錄製
                if not self.is_recording:
                    self.start_recording()
                else:
                    self.save_and_exit()
            elif key == ord('q'):
                rospy.signal_shutdown("User quit")
            elif key == ord('r'):  # 重設
                self.reset_recording()

            # 緩衝區滿時自動儲存
            if self.is_recording and self.buffer.is_full():
                self.save_and_exit()

        except Exception as e:
            rospy.logerr(f"Error: {e}")

    def draw_ui(self, image, keypoints, bbox):
        """繪製使用者介面元素。"""
        img = image.copy()
        h, w = img.shape[:2]

        # 繪製骨架
        if keypoints is not None:
            img = self.extractor.draw_skeleton(
                img, keypoints, bbox,
                color=(0, 255, 0) if self.is_recording else (255, 255, 0),
                normalized=True
            )

        # UI 背景
        cv2.rectangle(img, (0, 0), (w, 100), (0, 0, 0), -1)

        # 標題
        title = f"Recording: {self.action_name}"
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # 狀態
        if self.is_recording:
            status = f"Recording: {self.frames_recorded}/{self.num_frames}"
            color = (0, 0, 255)
            # 錄製指示器
            cv2.circle(img, (w - 30, 30), 10, (0, 0, 255), -1)
        else:
            status = "Press SPACE to start recording"
            color = (255, 255, 0)

        cv2.putText(img, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        # 進度條
        if self.is_recording:
            progress = self.frames_recorded / self.num_frames
            bar_width = int((w - 20) * progress)
            cv2.rectangle(img, (10, 80), (w - 10, 95), (100, 100, 100), -1)
            cv2.rectangle(img, (10, 80), (10 + bar_width, 95), (0, 255, 0), -1)

        # 骨架偵測狀態
        if keypoints is None:
            cv2.putText(img, "No person detected!", (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 操作說明
        instructions = "SPACE: Record | R: Reset | Q: Quit"
        cv2.putText(img, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        return img

    def start_recording(self):
        """開始錄製。"""
        self.buffer.clear()
        self.is_recording = True
        self.frames_recorded = 0
        rospy.loginfo(f"Started recording '{self.action_name}'")

    def reset_recording(self):
        """重設錄製。"""
        self.buffer.clear()
        self.is_recording = False
        self.frames_recorded = 0
        rospy.loginfo("Recording reset")

    def save_and_exit(self):
        """儲存錄製的序列並離開。"""
        if not self.buffer.is_full():
            rospy.logwarn(f"Insufficient frames! Got {len(self.buffer.buffer)}, "
                         f"need {self.num_frames}")
            return

        # 儲存
        sequence = self.buffer.get_sequence()
        filepath = os.path.join(self.output_dir, f"{self.action_name}.npy")
        np.save(filepath, sequence)

        rospy.loginfo(f"Saved action '{self.action_name}' to {filepath}")
        rospy.loginfo(f"Sequence shape: {sequence.shape}")

        rospy.signal_shutdown("Recording complete")

    def run(self):
        """執行錄製器。"""
        rospy.spin()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Record support set actions')
    parser.add_argument('--action', '-a', type=str, required=True,
                        help='Action name (e.g., waving, falling)')
    parser.add_argument('--frames', '-f', type=int, default=64,
                        help='Number of frames to record')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')

    # 只解析已知參數以處理 ROS 重映射參數
    args, unknown = parser.parse_known_args()

    try:
        recorder = SupportSetRecorder(
            action_name=args.action,
            num_frames=args.frames,
            output_dir=args.output
        )
        recorder.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
