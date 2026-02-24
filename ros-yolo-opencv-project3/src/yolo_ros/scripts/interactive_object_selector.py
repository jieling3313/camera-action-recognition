#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped # 用於發布3D點的標準ROS訊息
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters

class InteractiveSelector:
    def __init__(self):
        # 初始化節點、YOLO模型和CvBridge
        rospy.init_node('interactive_object_selector', anonymous=True)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_color_image = None
        self.latest_depth_image = None

        rospy.loginfo("互動式物件選擇器已啟動。")
        rospy.loginfo("正在訂閱影像主題...")

        # 建立一個 Publisher，用於發布被選中物件的位置
        self.position_pub = rospy.Publisher('/selected_object/position', PointStamped, queue_size=10)

        # 訂閱相機內參
        self.cam_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', 
            CameraInfo, 
            self.camera_info_callback
        )

        # 同步訂閱彩色和深度影像，並持續更新最新的影像
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.5)
        self.ts.registerCallback(self.image_update_callback)

    def camera_info_callback(self, msg):
        # 獲取並儲存相機內參
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.K[0], 'fy': msg.K[4],
                'cx': msg.K[2], 'cy': msg.K[5]
            }
            rospy.loginfo(f"成功獲取相機內參: {self.intrinsics}")
            self.cam_info_sub.unregister()

    def image_update_callback(self, color_msg, depth_msg):
        # 這個回呼函式只做一件事：不斷地將收到的最新影像儲存起來
        self.latest_color_image = color_msg
        self.latest_depth_image = depth_msg

    def run_selection_process(self):
        # 等待直到我們獲取了必要的資訊
        while not rospy.is_shutdown() and (self.latest_color_image is None or self.intrinsics is None):
            rospy.loginfo("正在等待相機影像和內參...")
            rospy.sleep(1.0)
        
        if rospy.is_shutdown(): return

        # 處理當前最新的影像
        color_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "16UC1")

        # 進行YOLO辨識
        rospy.loginfo("影像已擷取，正在執行物件辨識...")
        results = self.model(color_image)
        
        detected_objects = []
        # 遍歷所有偵測到的物件並計算3D座標
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_image[center_v, center_u]
            
            if depth == 0: continue

            Z = depth / 1000.0 # 轉換為公尺
            X = (center_u - self.intrinsics['cx']) * Z / self.intrinsics['fx']
            Y = (center_v - self.intrinsics['cy']) * Z / self.intrinsics['fy']
            
            class_name = self.model.names[int(box.cls)]
            detected_objects.append({'name': class_name, 'x': X, 'y': Y, 'z': Z})

        # 如果沒有偵測到任何物件
        if not detected_objects:
            rospy.loginfo("在此畫面中未偵測到任何物件。")
            return

        # 在終端機顯示可選擇的物件列表
        print("\n--- 偵測到的物件 ---")
        for i, obj in enumerate(detected_objects):
            print(f"  [{i+1}] {obj['name']} (位置 Z: {obj['z']:.2f}m)")
        print("--------------------")

        # 讓使用者選擇
        try:
            choice = int(input("請輸入您想發布位置的物件編號 (輸入數字): ")) - 1
            if 0 <= choice < len(detected_objects):
                selected_object = detected_objects[choice]
                
                # 建立 PointStamped 訊息
                point_msg = PointStamped()
                point_msg.header.stamp = rospy.Time.now()
                point_msg.header.frame_id = "camera_color_optical_frame" # 座標系
                point_msg.point.x = selected_object['x']
                point_msg.point.y = selected_object['y']
                point_msg.point.z = selected_object['z']
                
                # 發布訊息
                self.position_pub.publish(point_msg)
                rospy.loginfo(f"已發布物件 '{selected_object['name']}' 的位置訊息至 /selected_object/position")
            else:
                rospy.logwarn("無效的選擇。")
        except ValueError:
            rospy.logwarn("請輸入一個有效的數字。")

if __name__ == '__main__':
    selector = InteractiveSelector()
    # 等待一小段時間讓訂閱者建立連接
    rospy.sleep(2.0)
    # 執行一次選擇與發布的流程
    selector.run_selection_process()