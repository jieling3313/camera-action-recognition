#!/usr/bin/env python3.10
"""
Camera Tracker Node - Tracks largest person detected by YOLO
Uses PID control to keep person centered in camera view
Controls Arduino stepper motor via topics

Features:
- Body tracking with PID control
- Patrol/Cruise mode when no person detected
- Track largest person mode
- Auto-patrol mode (automatically patrol when no person detected)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool, String, Float32
from ultralytics import YOLO
import time


class CvBridgeSimple:
    """Simple replacement for cv_bridge to avoid NumPy version conflicts"""

    def imgmsg_to_cv2(self, img_msg, desired_encoding="bgr8"):
        """Convert ROS Image message to OpenCV image"""
        dtype = np.uint8
        if img_msg.encoding == "32FC1":
            dtype = np.float32
        elif img_msg.encoding == "16UC1":
            dtype = np.uint16

        # Convert raw data to numpy array
        img = np.frombuffer(img_msg.data, dtype=dtype)

        # Reshape based on encoding
        if img_msg.encoding in ["rgb8", "bgr8"]:
            img = img.reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding == "rgba8" or img_msg.encoding == "bgra8":
            img = img.reshape((img_msg.height, img_msg.width, 4))
        elif img_msg.encoding in ["mono8", "8UC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        elif img_msg.encoding in ["mono16", "16UC1", "32FC1"]:
            img = img.reshape((img_msg.height, img_msg.width))
        else:
            # Default: try 3 channel
            try:
                img = img.reshape((img_msg.height, img_msg.width, 3))
            except:
                img = img.reshape((img_msg.height, img_msg.width))

        # Convert color if needed
        if img_msg.encoding == "rgb8" and desired_encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "bgr8" and desired_encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
        """Convert OpenCV image to ROS Image message"""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding

        if len(cv_image.shape) == 3:
            img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        else:
            img_msg.step = cv_image.shape[1]

        img_msg.data = cv_image.tobytes()
        return img_msg


class PIDController:
    """Simple PID controller for smooth tracking"""
    def __init__(self, kp=2.0, ki=0.3, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

        # Smooth tracking: limit max output change between frames
        self.last_output = 0
        self.max_output_change = 50  # Maximum change per update

    def compute(self, error):
        """Compute PID output"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = 0.01

        # Proportional term
        p_term = self.kp * error

        # Integral term with tighter anti-windup
        self.integral += error * dt
        # Anti-windup: limit integral to prevent large accumulated error
        self.integral = np.clip(self.integral, -50, 50)
        i_term = self.ki * self.integral

        # Derivative term
        d_term = 0
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt

        # Update state
        self.prev_error = error
        self.last_time = current_time

        raw_output = p_term + i_term + d_term

        # Smooth output: limit how fast output can change
        output_change = raw_output - self.last_output
        output_change = np.clip(output_change, -self.max_output_change, self.max_output_change)
        output = self.last_output + output_change
        self.last_output = output

        return output

    def reset(self):
        """Reset PID state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.last_output = 0


class PatrolController:
    """Controller for patrol/cruise mode - oscillates camera left/right to search for people"""
    def __init__(self, degrees_per_second=36.0, steps_per_revolution=1600, max_angle=360.0):
        self.degrees_per_second = degrees_per_second
        self.steps_per_revolution = steps_per_revolution
        self.steps_per_degree = steps_per_revolution / 360.0

        self.patrol_direction = 1  # 1 = clockwise, -1 = counter-clockwise
        self.last_patrol_time = time.time()
        self.is_patrolling = False

        # Position tracking for oscillation mode
        self.current_angle = 0.0  # Current angle from center (degrees)
        self.max_angle = max_angle  # Maximum angle from center (±180 = full rotation)
        self.min_angle = -max_angle

    def compute_steps(self, dt, current_position=0):
        """Compute patrol steps based on time elapsed, with oscillation limits"""
        if dt <= 0:
            return 0

        degrees = self.degrees_per_second * dt
        self.current_angle += degrees * self.patrol_direction

        # Check if we've reached the limit and need to reverse
        if self.current_angle >= self.max_angle:
            self.current_angle = self.max_angle
            self.patrol_direction = -1  # Reverse direction
            rospy.loginfo(f"Patrol: Reached max angle ({self.max_angle}°), reversing direction")
        elif self.current_angle <= self.min_angle:
            self.current_angle = self.min_angle
            self.patrol_direction = 1  # Reverse direction
            rospy.loginfo(f"Patrol: Reached min angle ({self.min_angle}°), reversing direction")

        steps = int(degrees * self.steps_per_degree * self.patrol_direction)
        return steps

    def reverse_direction(self):
        """Reverse patrol direction"""
        self.patrol_direction *= -1

    def set_speed(self, degrees_per_second):
        """Set patrol speed"""
        self.degrees_per_second = degrees_per_second

    def set_max_angle(self, max_angle):
        """Set maximum patrol angle (degrees from center)"""
        self.max_angle = max_angle
        self.min_angle = -max_angle
        rospy.loginfo(f"Patrol max angle set to ±{max_angle}°")

    def reset(self):
        """Reset patrol state"""
        self.last_patrol_time = time.time()
        self.is_patrolling = False
        self.current_angle = 0.0
        self.patrol_direction = 1


class CameraTrackerNode:
    def __init__(self):
        rospy.init_node('camera_tracker_node')

        # Parameters
        self.tracking_enabled = False
        self.image_width = 640  # Will be updated from actual image
        self.image_height = 480
        self.center_deadzone = rospy.get_param('~center_deadzone', 50)  # pixels
        self.max_steps_per_update = rospy.get_param('~max_steps_per_update', 50)  # Reduced from 100 to prevent overshoot
        self.pixels_per_step = rospy.get_param('~pixels_per_step', 2.0)  # Increased to reduce steps per pixel

        # Patrol parameters
        self.patrol_speed = rospy.get_param('~patrol_speed', 36.0)  # degrees per second
        self.steps_per_revolution = rospy.get_param('~steps_per_revolution', 1600)
        self.no_person_timeout = rospy.get_param('~no_person_timeout', 3.0)  # seconds before auto-patrol

        # Mode flags
        self.patrol_enabled = False  # Manual patrol mode
        self.auto_patrol_enabled = False  # Auto-patrol when no person detected
        self.track_largest_person = True  # Track largest person by bounding box area

        # YOLO model (use existing model from yolo_unified_node)
        model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        self.model = YOLO(model_path)
        self.model.verbose = False  # Disable verbose output to reduce log spam

        # PID controller
        kp = rospy.get_param('~pid_kp', 2.0)
        ki = rospy.get_param('~pid_ki', 0.3)
        kd = rospy.get_param('~pid_kd', 0.1)
        self.pid = PIDController(kp, ki, kd)

        # Patrol controller
        self.patrol = PatrolController(self.patrol_speed, self.steps_per_revolution)

        # Timing for auto-patrol
        self.last_person_seen_time = time.time()
        self.person_detected = False

        # Person confirmation for stable detection (avoid false positives)
        self.person_confirm_timeout = rospy.get_param('~person_confirm_timeout', 1.0)  # seconds to confirm person
        self.person_first_seen_time = None  # When person was first detected
        self.person_confirmed = False  # Whether person has been confirmed (visible for confirm_timeout)

        # Tracking smoothness parameters
        self.last_tracking_error = 0  # For direction change detection
        self.direction_change_count = 0  # Count rapid direction changes
        self.last_motor_command_time = time.time()
        self.min_command_interval = 0.1  # Minimum time between motor commands (seconds)
        self.motor_recovery_attempts = 0
        self.max_recovery_attempts = 3

        # Frame skip for reduced CPU/GPU load (detect every N frames)
        # This helps avoid resource competition with MediaPipe in recognition node
        self.detection_frame_skip = rospy.get_param('~detection_frame_skip', 2)  # Process every 2nd frame
        self.frame_counter = 0
        self.last_detection_result = None  # Cache last detection result

        # CV Bridge (use simple version to avoid NumPy conflicts)
        self.bridge = CvBridgeSimple()

        # Publishers
        self.step_command_pub = rospy.Publisher('motor_step_command', Int32, queue_size=10)
        self.tracking_image_pub = rospy.Publisher('tracking_annotated_image', Image, queue_size=1)
        self.status_info_pub = rospy.Publisher('body_tracking_status', String, queue_size=10)
        self.person_count_pub = rospy.Publisher('detected_person_count', Int32, queue_size=10)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.enable_sub = rospy.Subscriber('tracking_enable', Bool, self.enable_callback)
        self.home_sub = rospy.Subscriber('tracking_home', Bool, self.home_callback)
        self.motor_position_sub = rospy.Subscriber('motor_position', Int32, self.position_callback)
        self.motor_status_sub = rospy.Subscriber('motor_status', String, self.status_callback)

        # New subscribers for extended features
        self.patrol_enable_sub = rospy.Subscriber('patrol_enable', Bool, self.patrol_enable_callback)
        self.auto_patrol_sub = rospy.Subscriber('auto_patrol_enable', Bool, self.auto_patrol_callback)
        self.patrol_speed_sub = rospy.Subscriber('patrol_speed', Float32, self.patrol_speed_callback)
        self.track_largest_sub = rospy.Subscriber('track_largest_person', Bool, self.track_largest_callback)

        # Motor state
        self.motor_position = 0
        self.last_motor_position = 0
        self.motor_position_unchanged_count = 0
        self.motor_enabled_pub = rospy.Publisher('motor_enable', Bool, queue_size=10)
        self.motor_home_pub = rospy.Publisher('motor_home', Bool, queue_size=10)

        # Patrol timer
        self.patrol_timer = rospy.Timer(rospy.Duration(0.1), self.patrol_timer_callback)

        # Motor health monitor timer (check every 2 seconds)
        self.motor_health_timer = rospy.Timer(rospy.Duration(2.0), self.motor_health_callback)

        rospy.loginfo("Camera Tracker Node initialized (Extended Version)")
        rospy.loginfo(f"PID Parameters: Kp={kp}, Ki={ki}, Kd={kd}")
        rospy.loginfo(f"Center deadzone: {self.center_deadzone} pixels")
        rospy.loginfo(f"Patrol speed: {self.patrol_speed} deg/s")
        rospy.loginfo(f"Person confirm timeout: {self.person_confirm_timeout}s")
        rospy.loginfo("Topics:")
        rospy.loginfo("  - /tracking_enable: Enable/disable tracking")
        rospy.loginfo("  - /patrol_enable: Enable/disable manual patrol mode")
        rospy.loginfo("  - /auto_patrol_enable: Enable/disable auto-patrol mode")
        rospy.loginfo("  - /patrol_speed: Set patrol speed (deg/s)")
        rospy.loginfo("  - /track_largest_person: Enable tracking largest person")

    def enable_callback(self, msg):
        """Enable/disable tracking"""
        self.tracking_enabled = msg.data
        if self.tracking_enabled:
            rospy.loginfo("Tracking ENABLED")
            self.pid.reset()
            self.last_person_seen_time = time.time()
            # Reset person confirmation for fresh start
            self.person_confirmed = False
            self.person_first_seen_time = None
            # Enable motor
            enable_msg = Bool()
            enable_msg.data = True
            self.motor_enabled_pub.publish(enable_msg)
            # Stop patrol mode when tracking starts
            self.patrol_enabled = False
            self.patrol.is_patrolling = False
        else:
            rospy.loginfo("Tracking DISABLED")
            # Reset states when tracking is disabled
            self.person_confirmed = False
            self.person_first_seen_time = None
            self.pid.reset()

    def patrol_enable_callback(self, msg):
        """Enable/disable manual patrol mode"""
        self.patrol_enabled = msg.data
        if self.patrol_enabled:
            rospy.loginfo("Patrol started")
            rospy.loginfo("Manual Patrol Mode ENABLED")
            self.patrol.reset()
            self.patrol.is_patrolling = True
            # Disable tracking when patrol starts
            self.tracking_enabled = False
            # Reset person confirmation - require new confirmation when patrol starts
            self.person_confirmed = False
            self.person_first_seen_time = None
            # Reset PID state
            self.pid.reset()
            # Enable motor
            enable_msg = Bool()
            enable_msg.data = True
            self.motor_enabled_pub.publish(enable_msg)
        else:
            rospy.loginfo("Manual Patrol Mode DISABLED")
            self.patrol.is_patrolling = False

    def auto_patrol_callback(self, msg):
        """Enable/disable auto-patrol mode"""
        self.auto_patrol_enabled = msg.data
        if self.auto_patrol_enabled:
            rospy.loginfo("Auto-Patrol Mode ENABLED (will patrol when no person detected)")
        else:
            rospy.loginfo("Auto-Patrol Mode DISABLED")
            if self.patrol.is_patrolling and not self.patrol_enabled:
                self.patrol.is_patrolling = False

    def patrol_speed_callback(self, msg):
        """Set patrol speed"""
        self.patrol_speed = msg.data
        self.patrol.set_speed(msg.data)
        rospy.loginfo(f"Patrol speed set to {msg.data} deg/s")

    def track_largest_callback(self, msg):
        """Enable/disable tracking largest person"""
        self.track_largest_person = msg.data
        rospy.loginfo(f"Track largest person: {msg.data}")

    def home_callback(self, msg):
        """Home the motor"""
        if msg.data:
            rospy.loginfo("Homing motor...")
            self.motor_home_pub.publish(msg)

    def position_callback(self, msg):
        """Update motor position"""
        self.motor_position = msg.data

    def status_callback(self, msg):
        """Log motor status and handle errors"""
        status = msg.data
        rospy.loginfo(f"Motor status: {status}")

        # Check for error conditions and attempt recovery
        if "Error" in status or "disabled" in status.lower():
            if self.tracking_enabled or self.patrol.is_patrolling:
                self.motor_recovery_attempts += 1
                if self.motor_recovery_attempts <= self.max_recovery_attempts:
                    rospy.logwarn(f"Motor error detected, attempting recovery ({self.motor_recovery_attempts}/{self.max_recovery_attempts})")
                    # Re-enable motor
                    enable_msg = Bool()
                    enable_msg.data = True
                    self.motor_enabled_pub.publish(enable_msg)
                else:
                    rospy.logerr("Max recovery attempts reached, stopping tracking")
                    self.tracking_enabled = False
                    self.patrol.is_patrolling = False
        elif "completed" in status.lower() or "enabled" in status.lower():
            # Reset recovery counter on successful operation
            self.motor_recovery_attempts = 0

    def motor_health_callback(self, event):
        """Monitor motor health and attempt recovery if stuck"""
        # Check if motor position hasn't changed when it should be moving
        if self.tracking_enabled or self.patrol.is_patrolling:
            if self.motor_position == self.last_motor_position:
                self.motor_position_unchanged_count += 1

                # If position unchanged for too long (6 seconds = 3 checks), motor might be stuck
                if self.motor_position_unchanged_count >= 3:
                    rospy.logwarn("Motor appears stuck - attempting recovery")
                    # Reset PID to prevent accumulated error
                    self.pid.reset()
                    # Re-enable motor
                    enable_msg = Bool()
                    enable_msg.data = True
                    self.motor_enabled_pub.publish(enable_msg)
                    self.motor_position_unchanged_count = 0
            else:
                self.motor_position_unchanged_count = 0

            self.last_motor_position = self.motor_position

    def patrol_timer_callback(self, event):
        """Timer callback for patrol mode"""
        if not self.patrol.is_patrolling:
            return

        current_time = time.time()
        dt = current_time - self.patrol.last_patrol_time
        self.patrol.last_patrol_time = current_time

        # Compute and send patrol steps
        steps = self.patrol.compute_steps(dt)
        if steps != 0:
            step_msg = Int32()
            step_msg.data = steps
            self.step_command_pub.publish(step_msg)

    def publish_status(self, mode, person_count=0, offset=0):
        """Publish current status"""
        status = {
            'mode': mode,
            'person_count': person_count,
            'offset': offset,
            'motor_position': self.motor_position,
            'tracking_enabled': self.tracking_enabled,
            'patrol_enabled': self.patrol_enabled,
            'auto_patrol_enabled': self.auto_patrol_enabled
        }
        self.status_info_pub.publish(String(data=str(status)))
        self.person_count_pub.publish(Int32(data=person_count))

    def find_all_persons(self, results):
        """Find all persons in the frame"""
        persons = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # Check if it's a person (class_id 0 in COCO dataset)
            if class_name.lower() == 'person':
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                conf = float(box.conf[0])
                persons.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'area': area,
                    'confidence': conf
                })

        # Sort by area (largest first)
        persons.sort(key=lambda x: x['area'], reverse=True)
        return persons

    def find_largest_person(self, results):
        """Find the person with largest bounding box area"""
        persons = self.find_all_persons(results)
        if persons:
            largest = persons[0]
            return largest['box'], largest['area']
        return None, 0

    def image_callback(self, msg):
        """Process image and track person"""
        # Always process image for visualization even if not tracking
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_height, self.image_width = cv_image.shape[:2]

            # Frame skipping to reduce CPU/GPU load and avoid resource competition
            # with MediaPipe in recognition_display_node_v2
            self.frame_counter += 1
            should_detect = (self.frame_counter % self.detection_frame_skip == 0)

            if should_detect:
                # Run YOLO detection (verbose=False to suppress speed logs)
                results = self.model(cv_image, verbose=False)
                # Find all persons
                persons = self.find_all_persons(results)
                # Cache result for skipped frames
                self.last_detection_result = persons
            else:
                # Use cached detection result
                persons = self.last_detection_result if self.last_detection_result else []

            person_count = len(persons)

            # Get target person (largest if track_largest_person is enabled)
            target_person = None
            if persons and self.track_largest_person:
                target_person = persons[0]

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

            # Draw all detected persons
            for i, person in enumerate(persons):
                x1, y1, x2, y2 = person['box']
                is_target = (i == 0 and self.track_largest_person)

                # Different colors for target vs other persons
                if is_target:
                    color = (0, 0, 255)  # Red for target
                    thickness = 3
                else:
                    color = (255, 165, 0)  # Orange for others
                    thickness = 2

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

                # Draw person info
                label = f"P{i+1}: {person['area']:.0f}"
                if is_target:
                    label = f"TARGET {label}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Determine current mode
            current_mode = "IDLE"
            error = 0

            if target_person is not None:
                self.person_detected = True
                self.last_person_seen_time = time.time()

                # Track how long person has been visible for confirmation
                if self.person_first_seen_time is None:
                    self.person_first_seen_time = time.time()
                    rospy.loginfo("Person detected - starting confirmation timer...")

                time_visible = time.time() - self.person_first_seen_time

                # Check if person has been confirmed (visible for required duration)
                if not self.person_confirmed and time_visible >= self.person_confirm_timeout:
                    self.person_confirmed = True
                    rospy.loginfo(f"Person CONFIRMED after {time_visible:.1f}s - ready to track")

                # Switch from patrol to tracking when person is confirmed
                if self.person_confirmed:
                    if self.patrol.is_patrolling:
                        # Stop patrol and switch to tracking when person is confirmed
                        self.patrol.is_patrolling = False
                        self.patrol_enabled = False  # Also disable manual patrol flag
                        self.tracking_enabled = True  # Enable tracking
                        self.pid.reset()  # Reset PID for smooth transition
                        rospy.loginfo("Person confirmed - switching from PATROL to TRACKING mode")

                if self.tracking_enabled and self.person_confirmed:
                    current_mode = "TRACKING"
                    x1, y1, x2, y2 = target_person['box']
                    person_center_x = (x1 + x2) // 2

                    # Draw target center
                    cv2.circle(annotated_image, (person_center_x, (y1 + y2) // 2), 8, (0, 0, 255), -1)

                    # Calculate error (positive = person is to the right)
                    error = person_center_x - center_x

                    # Detect rapid direction changes (oscillation)
                    if self.last_tracking_error != 0:
                        if (error > 0 and self.last_tracking_error < 0) or (error < 0 and self.last_tracking_error > 0):
                            self.direction_change_count += 1
                            if self.direction_change_count > 3:
                                # Too many direction changes, likely oscillating - reduce response
                                rospy.logwarn_throttle(2.0, "Oscillation detected - reducing motor response")
                                self.pid.reset()
                                self.direction_change_count = 0
                        else:
                            self.direction_change_count = max(0, self.direction_change_count - 1)
                    self.last_tracking_error = error

                    # Check if outside deadzone
                    if abs(error) > self.center_deadzone:
                        # Rate limit motor commands
                        current_time = time.time()
                        if current_time - self.last_motor_command_time < self.min_command_interval:
                            # Skip this command, too soon
                            pass
                        else:
                            # Compute PID output
                            pid_output = self.pid.compute(error)

                            # Convert to motor steps (negate to match motor direction)
                            # Positive error = person on right = motor should turn right (negative steps)
                            steps = -int(pid_output / self.pixels_per_step)

                            # Dynamic step limiting based on error magnitude
                            # Smaller errors = smaller max steps to prevent overshoot
                            error_ratio = min(abs(error) / 300.0, 1.0)  # Normalize error (300px = full range)
                            dynamic_max_steps = int(self.max_steps_per_update * (0.3 + 0.7 * error_ratio))

                            # Limit steps
                            steps = np.clip(steps, -dynamic_max_steps, dynamic_max_steps)

                            # Send command to motor
                            if steps != 0:
                                step_msg = Int32()
                                step_msg.data = steps
                                self.step_command_pub.publish(step_msg)
                                self.last_motor_command_time = current_time

                                rospy.loginfo_throttle(0.5, f"Tracking: offset={error:.0f}px, steps={steps}")
                    else:
                        rospy.loginfo_throttle(2.0, f"Person centered (offset: {error:.0f}px)")

            else:
                self.person_detected = False
                self.pid.reset()

                # Reset person confirmation timer when no person detected
                # But keep person_confirmed=True if we're in tracking mode (don't require re-confirmation)
                if self.person_first_seen_time is not None:
                    self.person_first_seen_time = None
                    # Only reset confirmation if we're in patrol mode (not tracking)
                    if not self.tracking_enabled:
                        self.person_confirmed = False
                        rospy.loginfo_throttle(1.0, "Person lost - reset confirmation")
                    else:
                        rospy.loginfo_throttle(1.0, "Person lost during tracking - will resume when found")

                # Check for auto-patrol trigger
                if self.auto_patrol_enabled and self.tracking_enabled:
                    time_since_last_person = time.time() - self.last_person_seen_time
                    if time_since_last_person > self.no_person_timeout and not self.patrol.is_patrolling:
                        rospy.loginfo(f"No person for {time_since_last_person:.1f}s - starting auto-patrol")
                        self.patrol.reset()
                        self.patrol.is_patrolling = True
                        # Reset confirmation when switching to patrol
                        self.person_confirmed = False

            # Update mode display
            if self.patrol.is_patrolling:
                if self.patrol_enabled:
                    current_mode = "MANUAL PATROL"
                else:
                    if target_person and not self.person_confirmed:
                        current_mode = "CONFIRMING"
                    else:
                        current_mode = "AUTO PATROL"
            elif not self.tracking_enabled and not self.patrol_enabled:
                current_mode = "STANDBY"

            # Draw status info
            status_text = f"Mode: {current_mode} | Persons: {person_count} | Pos: {self.motor_position}"
            cv2.putText(annotated_image, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if target_person:
                # Show confirmation progress during patrol
                if self.person_first_seen_time is not None and not self.person_confirmed:
                    time_visible = time.time() - self.person_first_seen_time
                    confirm_text = f"Confirming: {time_visible:.1f}s / {self.person_confirm_timeout:.1f}s"
                    cv2.putText(annotated_image, confirm_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Draw progress bar
                    progress = min(time_visible / self.person_confirm_timeout, 1.0)
                    bar_width = 200
                    bar_height = 10
                    cv2.rectangle(annotated_image, (10, 100), (10 + bar_width, 100 + bar_height), (100, 100, 100), -1)
                    cv2.rectangle(annotated_image, (10, 100), (10 + int(bar_width * progress), 100 + bar_height), (0, 255, 255), -1)

                if self.tracking_enabled and self.person_confirmed:
                    offset_text = f"Offset: {error:.0f}px | Area: {target_person['area']:.0f}"
                    cv2.putText(annotated_image, offset_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if person_count == 0:
                cv2.putText(annotated_image, "No person detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw mode indicators
            mode_y = self.image_height - 60
            indicators = [
                (f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}",
                 (0, 255, 0) if self.tracking_enabled else (128, 128, 128)),
                (f"Patrol: {'ON' if self.patrol.is_patrolling else 'OFF'}",
                 (0, 255, 255) if self.patrol.is_patrolling else (128, 128, 128)),
                (f"Auto-Patrol: {'ON' if self.auto_patrol_enabled else 'OFF'}",
                 (255, 165, 0) if self.auto_patrol_enabled else (128, 128, 128)),
            ]
            for i, (text, color) in enumerate(indicators):
                cv2.putText(annotated_image, text, (10 + i * 200, mode_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Publish annotated image
            self.tracking_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))

            # Publish status
            self.publish_status(current_mode, person_count, error)

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
