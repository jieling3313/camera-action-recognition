#!/usr/bin/env python3
import rospy
import cv2
import threading # 為了線程安全

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters

from yolo_ros.srv import TriggerSelection, TriggerSelectionResponse

class YoloUnifiedNode:
    def __init__(self):
        rospy.init_node('yolo_unified_node')

        # 初始化 
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_detection_results = [] # 儲存最近的辨識結果
        self.lock = threading.Lock() # 線程鎖，用於保護 latest_detection_results

        rospy.loginfo("YOLO 統一功能節點已啟動。")

        # ROS Publishers
        self.annotated_image_pub = rospy.Publisher("/yolo_annotated_image", Image, queue_size=1)
        self.selected_pos_pub = rospy.Publisher('/selected_object/position', PointStamped, queue_size=10)

        # ROS Service Server
        self.srv = rospy.Service('/trigger_selection', TriggerSelection, self.handle_selection_request)

        # ROS Subscribers
        self.cam_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.5)
        self.ts.registerCallback(self.image_callback) # 連續處理每一幀

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.K[0], 'fy': msg.K[4], 'cx': msg.K[2], 'cy': msg.K[5]}
            rospy.loginfo(f"成功獲取相機內參。")
            self.cam_info_sub.unregister()

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None: return

        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"影像轉換失敗: {e}")
            return

        results = self.model(color_image)
        annotated_frame = results[0].plot()

        current_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_image[center_v, center_u]
            if depth == 0: continue

            Z = depth / 1000.0
            X = (center_u - self.intrinsics['cx']) * Z / self.intrinsics['fx']
            Y = (center_v - self.intrinsics['cy']) * Z / self.intrinsics['fy']
            class_name = self.model.names[int(box.cls)]

            # 儲存這個物件的資訊
            current_detections.append({'name': class_name, 'x': X, 'y': Y, 'z': Z})

            # 在畫面上標註3D座標
            cv2.putText(annotated_frame, f"({X:.1f},{Y:.1f},{Z:.1f})m", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 發布標註後的影像
        self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

        # 更新最近的辨識結果
        with self.lock:
            self.latest_detection_results = current_detections

    def handle_selection_request(self, req):
        rospy.loginfo("收到 Client 請求，回傳最近一次的辨識結果。")

        names = []
        positions = []

        # 讀取辨識結果
        with self.lock:
            if not self.latest_detection_results:
                rospy.logwarn("目前沒有可用的辨識結果。")
                return TriggerSelectionResponse([], [])

            for obj in self.latest_detection_results:
                names.append(obj['name'])
                positions.append(Point(x=obj['x'], y=obj['y'], z=obj['z']))

        return TriggerSelectionResponse(names, positions)

if __name__ == '__main__':
    node = YoloUnifiedNode()
    rospy.spin()