#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

# ROS
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# 同步化彩色影像和深度影像
import message_filters
from message_filters import Subscriber, ApproximateTimeSynchronizer

# YOLO 模型
from ultralytics import YOLO

class ObjectPositionEstimator:
    def __init__(self):
        rospy.init_node('object_position_estimator', anonymous=True)
        
        # 初始化
        self.bridge = CvBridge()
        self.intrinsics = None # 用於儲存相機內參

        # YOLO 模型 選擇
        rospy.loginfo("正在載入 YOLOv8 模型...")
        # self.model = YOLO('yolov8n.pt')
        # self.model = YOLO('yolov8s.pt')
        self.model = YOLO('yolov8m.pt')
        # self.model = YOLO('yolov8l.pt')
        rospy.loginfo("YOLOv8 模型載入成功。")

        #ROS Publisher (節點發布)
        self.annotated_image_pub = rospy.Publisher("/yolo_annotated_image", Image, queue_size=1)

        #訂閱相機內參 (Camera Intrinsics)
        self.cam_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', 
            CameraInfo, 
            self.camera_info_callback
        )
        rospy.loginfo("正在等待相機內參訊息...")
        
        #使用 message_filters 同步訂閱彩色和深度影像
        color_sub = Subscriber('/camera/color/image_raw', Image)
        depth_sub = Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.5)
        self.ts.registerCallback(self.image_callback)
        
        rospy.loginfo("節點已啟動，等待同步的影像訊息...")

    def camera_info_callback(self, msg):
        self.intrinsics = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5]
        }
        rospy.loginfo(f"成功獲取相機內參: {self.intrinsics}")
        self.cam_info_sub.unregister()

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None:
            rospy.logwarn("尚未收到相機內參，跳過此幀。")
            return
            
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 轉換失敗: {e}")
            return
            
        results = self.model(color_image)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_u = (x1 + x2) // 2
            center_v = (y1 + y2) // 2
            
            depth = depth_image[center_v, center_u]
            
            if depth == 0:
                continue

            Z_mm = float(depth) # (mm))
            
            #2D轉3D
            fx = self.intrinsics['fx']
            fy = self.intrinsics['fy']
            cx = self.intrinsics['cx']
            cy = self.intrinsics['cy']

            Z = Z_mm
            X = (center_u - cx) * Z / fx
            Y = (center_v - cy) * Z / fy
            
            class_name = self.model.names[int(box.cls)]
            
            #輸出X(mm),Y(mm),Z(mm)
            rospy.loginfo(
                f"偵測到物件: {class_name} | "
                f"3D 座標 (X, Y, Z): ({X:.0f}mm, {Y:.0f}mm, {Z:.0f}mm)"
            )
            
            # 更新畫面物件座標
            cv2.putText(
                annotated_frame, 
                f"({X:.0f}, {Y:.0f}, {Z:.0f})mm", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )

        try:
            self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"發布影像失敗: {e}")


if __name__ == '__main__':
    try:
        estimator = ObjectPositionEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass