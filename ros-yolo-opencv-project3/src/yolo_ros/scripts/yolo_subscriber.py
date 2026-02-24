#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import os
import time

class YoloDetector:
    def __init__(self):
        # 初始化 ROS 節點
        rospy.init_node('yolo_detector', anonymous=True)
        
        # 載入 YOLOv8 模型
        rospy.loginfo("正在載入 YOLOv8 模型...")
        self.model = YOLO('yolov8n.pt')
        # self.model = YOLO('yolov8s.pt')
        # self.model = YOLO('yolov8m.pt')
        # self.model = YOLO('yolov8l.pt')
        rospy.loginfo("YOLOv8 模型載入成功。")
        
        # 建立 CvBridge 實例
        self.bridge = CvBridge()
        
        # 建立一個 Publisher，將辨識後的影像發布出去
        self.image_pub = rospy.Publisher("/yolo_image", Image, queue_size=1)
        
        # 給予 Publisher 1秒鐘的時間來確保它已經成功在 ROS Master 註冊
        rospy.sleep(1.0)
        
        # 訂閱來自 RealSense 的彩色影像 Topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        # 定義儲存圖片的目錄
        self.save_dir = os.path.join(os.path.expanduser('~'), 'detection_results')
        # 如果目錄不存在，就建立它
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo(f"已建立用於儲存影像的目錄: {self.save_dir}")
            
        rospy.loginfo("YOLO Detector 節點已啟動，正在等待影像...")

    def image_callback(self, msg):
        rospy.loginfo("收到影像，開始處理...")
        try:
            # 將 ROS Image 訊息轉換為 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 轉換失敗: {e}")
            return

        # 使用 YOLOv8 進行物件偵測
        results = self.model(cv_image)
        
        # 取得標註好的畫面
        annotated_frame = results[0].plot()

        # 將標註好的畫面儲存成圖片檔案
        try:
            timestamp = int(time.time() * 1000)
            file_name = f"detection_{timestamp}.png"
            full_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(full_path, annotated_frame)
        except Exception as e:
            rospy.logerr(f"儲存影像失敗: {e}")

        # 將標註後的 OpenCV 影像轉換回 ROS Image 訊息並發布
        try:
            ros_image_out = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(ros_image_out)
            rospy.loginfo("已發布辨識後的影像。")
        except CvBridgeError as e:
            rospy.logerr(f"發布影像前的 CvBridge 轉換失敗: {e}")

if __name__ == '__main__':
    try:
        yolo_detector = YoloDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass