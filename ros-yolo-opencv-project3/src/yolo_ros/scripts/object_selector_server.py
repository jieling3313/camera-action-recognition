#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters

# 匯入ervice
from yolo_ros.srv import TriggerSelection, TriggerSelectionResponse

class ObjectSelectorServer:
    def __init__(self):
        rospy.init_node('object_selector_server')
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_color_image = None
        self.latest_depth_image = None

        # 訂閱相機內參
        self.cam_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # 同步訂閱影像
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.5)
        self.ts.registerCallback(self.image_update_callback)

        # 宣告 Service
        self.srv = rospy.Service('/trigger_selection', TriggerSelection, self.handle_selection_request)
        rospy.loginfo("物件選擇 Service Server 已就緒，等待請求...")

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.K[0], 'fy': msg.K[4], 'cx': msg.K[2], 'cy': msg.K[5]}
            rospy.loginfo("成功獲取相機內參。")
            self.cam_info_sub.unregister()

    def image_update_callback(self, color_msg, depth_msg):
        self.latest_color_image = color_msg
        self.latest_depth_image = depth_msg

    def handle_selection_request(self, req):
        rospy.loginfo("收到請求，正在處理當前影像...")
        if self.latest_color_image is None or self.intrinsics is None:
            rospy.logwarn("影像或內參尚未就緒，無法處理請求。")
            return TriggerSelectionResponse([], [])

        color_image = self.bridge.imgmsg_to_cv2(self.latest_color_image, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "16UC1")

        results = self.model(color_image)

        names = []
        positions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_image[center_v, center_u]
            if depth == 0: continue

            Z = depth
            X = (center_u - self.intrinsics['cx']) * Z / self.intrinsics['fx']
            Y = (center_v - self.intrinsics['cy']) * Z / self.intrinsics['fy']

            names.append(self.model.names[int(box.cls)])
            positions.append(Point(x=X, y=Y, z=Z))

        rospy.loginfo(f"偵測到 {len(names)} 個物件，回傳列表。")
        return TriggerSelectionResponse(names, positions)

if __name__ == '__main__':
    server = ObjectSelectorServer()
    rospy.spin()