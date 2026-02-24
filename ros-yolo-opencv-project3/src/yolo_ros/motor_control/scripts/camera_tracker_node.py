#!/usr/bin/env python3
"""
Camera Tracker Node - Tracks largest person detected by YOLO
Uses PID control to keep person centered in camera view
Controls Arduino stepper motor via topics
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO
import time


class PIDController:
    """Simple PID controller for smooth tracking"""
    def __init__(self, kp=2.0, ki=0.3, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        """Compute PID output"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = 0.01

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        # Anti-windup: limit integral
        self.integral = np.clip(self.integral, -100, 100)
        i_term = self.ki * self.integral

        # Derivative term
        d_term = 0
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt

        # Update state
        self.prev_error = error
        self.last_time = current_time

        output = p_term + i_term + d_term
        return output

    def reset(self):
        """Reset PID state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()


class CameraTrackerNode:
    def __init__(self):
        rospy.init_node('camera_tracker_node')

        # Parameters
        self.tracking_enabled = False
        self.image_width = 640  # Will be updated from actual image
        self.image_height = 480
        self.center_deadzone = rospy.get_param('~center_deadzone', 50)  # pixels
        self.max_steps_per_update = rospy.get_param('~max_steps_per_update', 100)
        self.pixels_per_step = rospy.get_param('~pixels_per_step', 1.5)  # Calibration parameter

        # YOLO model (use existing model from yolo_unified_node)
        model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        self.model = YOLO(model_path)

        # PID controller
        kp = rospy.get_param('~pid_kp', 2.0)
        ki = rospy.get_param('~pid_ki', 0.3)
        kd = rospy.get_param('~pid_kd', 0.1)
        self.pid = PIDController(kp, ki, kd)

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.step_command_pub = rospy.Publisher('motor_step_command', Int32, queue_size=10)
        self.tracking_image_pub = rospy.Publisher('tracking_annotated_image', Image, queue_size=1)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.enable_sub = rospy.Subscriber('tracking_enable', Bool, self.enable_callback)
        self.home_sub = rospy.Subscriber('tracking_home', Bool, self.home_callback)
        self.motor_position_sub = rospy.Subscriber('motor_position', Int32, self.position_callback)
        self.motor_status_sub = rospy.Subscriber('motor_status', String, self.status_callback)

        # Motor state
        self.motor_position = 0
        self.motor_enabled_pub = rospy.Publisher('motor_enable', Bool, queue_size=10)
        self.motor_home_pub = rospy.Publisher('motor_home', Bool, queue_size=10)

        rospy.loginfo("Camera Tracker Node initialized")
        rospy.loginfo(f"PID Parameters: Kp={kp}, Ki={ki}, Kd={kd}")
        rospy.loginfo(f"Center deadzone: {self.center_deadzone} pixels")
        rospy.loginfo("Use 'rostopic pub /tracking_enable std_msgs/Bool true' to start tracking")

    def enable_callback(self, msg):
        """Enable/disable tracking"""
        self.tracking_enabled = msg.data
        if self.tracking_enabled:
            rospy.loginfo("Tracking ENABLED")
            self.pid.reset()
            # Enable motor
            enable_msg = Bool()
            enable_msg.data = True
            self.motor_enabled_pub.publish(enable_msg)
        else:
            rospy.loginfo("Tracking DISABLED")

    def home_callback(self, msg):
        """Home the motor"""
        if msg.data:
            rospy.loginfo("Homing motor...")
            self.motor_home_pub.publish(msg)

    def position_callback(self, msg):
        """Update motor position"""
        self.motor_position = msg.data

    def status_callback(self, msg):
        """Log motor status"""
        rospy.loginfo(f"Motor status: {msg.data}")

    def find_largest_person(self, results):
        """Find the person with largest bounding box area"""
        largest_area = 0
        largest_box = None

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # Check if it's a person (class_id 0 in COCO dataset)
            if class_name.lower() == 'person':
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                if area > largest_area:
                    largest_area = area
                    largest_box = (int(x1), int(y1), int(x2), int(y2))

        return largest_box, largest_area

    def image_callback(self, msg):
        """Process image and track person"""
        if not self.tracking_enabled:
            return

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_height, self.image_width = cv_image.shape[:2]

            # Run YOLO detection
            results = self.model(cv_image)

            # Find largest person
            person_box, person_area = self.find_largest_person(results)

            # Annotate image
            annotated_image = cv_image.copy()

            # Draw center line
            center_x = self.image_width // 2
            cv2.line(annotated_image, (center_x, 0), (center_x, self.image_height),
                     (0, 255, 0), 2)

            # Draw deadzone
            deadzone_left = center_x - self.center_deadzone
            deadzone_right = center_x + self.center_deadzone
            cv2.line(annotated_image, (deadzone_left, 0), (deadzone_left, self.image_height),
                     (255, 255, 0), 1)
            cv2.line(annotated_image, (deadzone_right, 0), (deadzone_right, self.image_height),
                     (255, 255, 0), 1)

            if person_box is not None:
                x1, y1, x2, y2 = person_box
                person_center_x = (x1 + x2) // 2

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(annotated_image, (person_center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)

                # Calculate error (positive = person is to the right, need to move right)
                error = person_center_x - center_x

                # Check if outside deadzone
                if abs(error) > self.center_deadzone:
                    # Compute PID output
                    pid_output = self.pid.compute(error)

                    # Convert to motor steps
                    steps = int(pid_output / self.pixels_per_step)

                    # Limit steps
                    steps = np.clip(steps, -self.max_steps_per_update, self.max_steps_per_update)

                    # Send command to motor
                    if steps != 0:
                        step_msg = Int32()
                        step_msg.data = steps
                        self.step_command_pub.publish(step_msg)

                        rospy.loginfo(f"Person offset: {error:.0f}px, PID: {pid_output:.1f}, Steps: {steps}")
                else:
                    rospy.loginfo_throttle(2.0, f"Person centered (offset: {error:.0f}px)")

                # Display info
                info_text = f"Offset: {error:.0f}px | Area: {person_area:.0f} | Pos: {self.motor_position}"
                cv2.putText(annotated_image, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_image, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.pid.reset()  # Reset PID when no person detected

            # Publish annotated image
            self.tracking_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def run(self):
        """Main loop"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = CameraTrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
