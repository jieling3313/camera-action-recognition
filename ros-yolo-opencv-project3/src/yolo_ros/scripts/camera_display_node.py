#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
相機顯示節點 - 使用 ROS 原生視窗顯示 D435i 影像和 MediaPipe 骨架
避免 Qt X11 渲染負擔，提升效能
"""

# 設定環境變數 (必須在 import cv2 之前)
import os
# 設定 Qt plugin path for OpenCV
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
import mediapipe as mp
import mediapipe_compat  # 啟用兼容層以支援新版 MediaPipe
from datetime import datetime
import sys

# 添加 scripts 目錄到路徑以使用轉換工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
HAS_CONVERTER = False
try:
    from convert_mediapipe_to_coco import convert_mediapipe_to_coco
    HAS_CONVERTER = True
    rospy.loginfo("MediaPipe to COCO converter loaded successfully")
except ImportError as e:
    rospy.logwarn(f"convert_mediapipe_to_coco not found: {e}. Will save MediaPipe format only.")

class CameraDisplayNode:
    def __init__(self):
        rospy.init_node('camera_display_node', anonymous=False)

        self.bridge = CvBridge()
        self.current_frame = None
        self.skeleton_frame = None

        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 錄影相關
        self.is_recording = False
        self.video_writer = None
        self.skeleton_sequence = []
        self.record_filename = ""

        # 儲存路徑
        self.actionset_dir = "/root/catkin_ws/src/yolo_ros/actionset"
        os.makedirs(self.actionset_dir, exist_ok=True)

        # Note: OpenCV windows removed due to Qt conflicts in Docker
        # Use image_view instead: rosrun image_view image_view image:=/camera/skeleton_image
        rospy.loginfo("Camera display node initialized (use image_view for display)")

        # ROS 訂閱與發布
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        # 發布處理後的骨架影像（供其他節點使用）
        self.skeleton_pub = rospy.Publisher(
            "/camera/skeleton_image",
            Image,
            queue_size=1
        )

        # 服務：拍照
        self.capture_srv = rospy.Service(
            '/camera_display/capture',
            Trigger,
            self.handle_capture
        )

        # 服務：開始錄影
        self.start_record_srv = rospy.Service(
            '/camera_display/start_recording',
            Trigger,
            self.handle_start_recording
        )

        # 服務：停止錄影
        self.stop_record_srv = rospy.Service(
            '/camera_display/stop_recording',
            Trigger,
            self.handle_stop_recording
        )

        # 訂閱錄影檔名
        self.filename_sub = rospy.Subscriber(
            "/camera_display/filename",
            String,
            self.filename_callback,
            queue_size=1
        )

        rospy.loginfo("Camera Display Node initialized")
        rospy.loginfo("Services available:")
        rospy.loginfo("  - /camera_display/capture")
        rospy.loginfo("  - /camera_display/start_recording")
        rospy.loginfo("  - /camera_display/stop_recording")
        rospy.loginfo("Topics:")
        rospy.loginfo("  - Publish filename to: /camera_display/filename")

    def filename_callback(self, msg):
        """接收錄影/拍照檔名"""
        self.record_filename = msg.data
        rospy.loginfo(f"Filename set to: {self.record_filename}")

    def image_callback(self, msg):
        """處理相機影像"""
        try:
            # 轉換 ROS 影像到 OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_frame = cv_image.copy()

            # MediaPipe 骨架提取
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # 繪製骨架
            skeleton_img = cv_image.copy()
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    skeleton_img,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # 提取關鍵點座標
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                h, w = cv_image.shape[:2]
                for lm in landmarks:
                    keypoints.append([lm.x * w, lm.y * h, lm.visibility])

                # 如果正在錄影，儲存骨架序列
                if self.is_recording:
                    self.skeleton_sequence.append(keypoints)

            self.skeleton_frame = skeleton_img

            # 發布骨架影像
            skeleton_msg = self.bridge.cv2_to_imgmsg(skeleton_img, "bgr8")
            self.skeleton_pub.publish(skeleton_msg)

            # OpenCV window display removed (Qt conflicts)
            # Use: rosrun image_view image_view image:=/camera/skeleton_image

            # 如果正在錄影，寫入影片
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(cv_image)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def handle_capture(self, req):
        """處理拍照請求"""
        if self.current_frame is None or self.skeleton_frame is None:
            return TriggerResponse(success=False, message="No image received yet")

        if not self.record_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}"
        else:
            filename = self.record_filename

        # 建立儲存目錄
        save_dir = os.path.join(self.actionset_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # 儲存原始影像
        raw_path = os.path.join(save_dir, f"{filename}_raw.jpg")
        cv2.imwrite(raw_path, self.current_frame)

        # 儲存骨架影像
        skeleton_path = os.path.join(save_dir, f"{filename}_skeleton.jpg")
        cv2.imwrite(skeleton_path, self.skeleton_frame)

        rospy.loginfo(f"Captured images saved to: {save_dir}")
        return TriggerResponse(success=True, message=f"Saved to: {save_dir}")

    def handle_start_recording(self, req):
        """開始錄影"""
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        if self.current_frame is None:
            return TriggerResponse(success=False, message="No image received yet")

        if not self.record_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}"
        else:
            filename = self.record_filename

        # 建立儲存目錄
        save_dir = os.path.join(self.actionset_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # 初始化影片寫入器
        video_path = os.path.join(save_dir, f"{filename}_video.avi")
        h, w = self.current_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

        # 重置骨架序列
        self.skeleton_sequence = []
        self.is_recording = True

        rospy.loginfo(f"Started recording to: {save_dir}")
        return TriggerResponse(success=True, message=f"Recording started: {save_dir}")

    def handle_stop_recording(self, req):
        """停止錄影"""
        if not self.is_recording:
            return TriggerResponse(success=False, message="Not recording")

        self.is_recording = False

        # 釋放影片寫入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # 儲存骨架序列
        if self.skeleton_sequence:
            if not self.record_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_{timestamp}"
            else:
                filename = self.record_filename

            save_dir = os.path.join(self.actionset_dir, filename)

            # 儲存 MediaPipe 33 關鍵點格式（原始資料）
            skeleton_array = np.array(self.skeleton_sequence)  # Shape: (T, 33, 3)
            mediapipe_path = os.path.join(save_dir, f"{filename}_skeleton_sequence.npy")
            np.save(mediapipe_path, skeleton_array)
            rospy.loginfo(f"Saved MediaPipe format: {mediapipe_path} (shape: {skeleton_array.shape})")

            # 自動轉換並儲存 COCO 17 關鍵點格式
            if HAS_CONVERTER:
                try:
                    coco_sequence = convert_mediapipe_to_coco(skeleton_array)  # Shape: (T, 17, 3)
                    coco_path = os.path.join(save_dir, f"{filename}_skeleton_sequence_coco17.npy")
                    np.save(coco_path, coco_sequence)
                    rospy.loginfo(f"Auto-converted to COCO format: {coco_path} (shape: {coco_sequence.shape})")

                    return TriggerResponse(
                        success=True,
                        message=f"Recording stopped. {len(self.skeleton_sequence)} frames saved.\n"
                                f"MediaPipe 33: {mediapipe_path}\n"
                                f"COCO 17: {coco_path}"
                    )
                except Exception as e:
                    rospy.logerr(f"Failed to convert to COCO format: {e}")
                    import traceback
                    traceback.print_exc()
                    # 轉換失敗，但錄製成功
                    return TriggerResponse(
                        success=True,
                        message=f"Recording stopped. {len(self.skeleton_sequence)} frames saved.\n"
                                f"MediaPipe 33: {mediapipe_path}\n"
                                f"Warning: COCO conversion failed"
                    )
            else:
                # 沒有轉換器，只儲存 MediaPipe 格式
                return TriggerResponse(
                    success=True,
                    message=f"Recording stopped. {len(self.skeleton_sequence)} frames saved.\n"
                            f"MediaPipe 33: {mediapipe_path}\n"
                            f"Note: Install converter for auto COCO17 conversion"
                )
        else:
            return TriggerResponse(success=False, message="No frames recorded")

    def run(self):
        """主循環"""
        rospy.spin()

        # 清理
        cv2.destroyAllWindows()
        if self.video_writer is not None:
            self.video_writer.release()

if __name__ == '__main__':
    try:
        node = CameraDisplayNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Camera Display Node")
        cv2.destroyAllWindows()
