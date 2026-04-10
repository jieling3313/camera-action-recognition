/*
 * Arduino Stepper Motor Control for D435i Camera Tracking
 *
 * Hardware Connections:
 * - DIR   -> Arduino pin 9
 * - STEP  -> Arduino pin 10
 * - M0    -> Arduino pin 11
 * - M1    -> Arduino pin 12
 * - M2    -> Arduino pin 13
 * - RESET -> 5V
 * - SLEEP -> 5V
 * - VMOT  -> 12V (External Power)
 * - GND   -> External GND
 *
 * Driver: DRV8825
 * Motor: NEMA17 1.8°/step, 200 steps/revolution
 * Microstepping: 1/8 (1600 steps/revolution)
 */

#include <ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

// Pin definitions
const int DIR_PIN = 13;
const int STEP_PIN = 12;
const int M0_PIN = 9;
const int M1_PIN = 10;
const int M2_PIN = 11;

// Motor parameters
const int STEPS_PER_REV = 1600;  // 200 * 8 (1/8 microstepping)
const int DEFAULT_SPEED = 400;   // steps/sec
const int MAX_SPEED = 600;       // Maximum speed limit
const int MIN_SPEED = 100;       // Minimum starting speed
const int ACCEL_STEPS = 200;    // Steps to accelerate/decelerate

// Position tracking
volatile long currentPosition = 0;  // Current position in steps
volatile bool motorEnabled = false;
volatile long targetSteps = 0;
volatile bool isMoving = false;

// ROS node handle
ros::NodeHandle nh;

// Status publisher
std_msgs::Int32 position_msg;
ros::Publisher position_pub("motor_position", &position_msg);

std_msgs::String status_msg;
ros::Publisher status_pub("motor_status", &status_msg);

// Function to set microstepping mode (1/8 step)
void setMicrostepping() {
  // For DRV8825, 1/8 microstepping: M0=HIGH, M1=HIGH, M2=LOW
  digitalWrite(M0_PIN, HIGH);
  digitalWrite(M1_PIN, HIGH);
  digitalWrite(M2_PIN, LOW);
}

// Move motor by specified steps with acceleration
void moveMotor(long steps, int targetSpeed = DEFAULT_SPEED) {
  if (!motorEnabled) {
    publishStatus("Error: Motor disabled");
    return;
  }

  // Limit speed
  targetSpeed = constrain(targetSpeed, MIN_SPEED, MAX_SPEED);

  // Set direction
  bool dir = (steps >= 0);
  digitalWrite(DIR_PIN, dir ? HIGH : LOW);
  delayMicroseconds(10);  // Direction setup time

  long absSteps = abs(steps);
  int currentSpeed = MIN_SPEED;

  for (long i = 0; i < absSteps; i++) {
    // Acceleration/Deceleration profile
    if (i < ACCEL_STEPS && absSteps > ACCEL_STEPS * 2) {
      // Accelerate
      currentSpeed = MIN_SPEED + ((targetSpeed - MIN_SPEED) * i) / ACCEL_STEPS;
    } else if (i > absSteps - ACCEL_STEPS && absSteps > ACCEL_STEPS * 2) {
      // Decelerate
      long stepsLeft = absSteps - i;
      currentSpeed = MIN_SPEED + ((targetSpeed - MIN_SPEED) * stepsLeft) / ACCEL_STEPS;
    } else {
      // Constant speed
      currentSpeed = targetSpeed;
    }

    // Calculate delay for current speed
    long delayMicros = 1000000L / currentSpeed;

    // Generate step pulse with minimum 5us pulse width
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(max(5, delayMicros / 2));
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(max(5, delayMicros / 2));

    // Update position
    currentPosition += dir ? 1 : -1;

    // Publish position every 100 steps
    if (i % 100 == 0) {
      publishPosition();
      nh.spinOnce();  // Keep ROS connection alive
    }
  }

  publishPosition();
  publishStatus("Move completed");
}

// Home the motor (return to zero position)
void homeMotor() {
  if (!motorEnabled) {
    publishStatus("Error: Motor disabled");
    return;
  }

  publishStatus("Homing...");
  long stepsToHome = -currentPosition;
  moveMotor(stepsToHome, DEFAULT_SPEED);
  currentPosition = 0;
  publishPosition();
  publishStatus("Homing completed");
}

// Enable/disable motor
void enableMotor(bool enable) {
  motorEnabled = enable;
  if (enable) {
    publishStatus("Motor enabled");
  } else {
    publishStatus("Motor disabled");
  }
}

// Publish current position
void publishPosition() {
  if (nh.connected()) {
    position_msg.data = currentPosition;
    position_pub.publish(&position_msg);
  }
}

// Publish status message
void publishStatus(const char* message) {
  if (nh.connected()) {
    status_msg.data = message;
    status_pub.publish(&status_msg);
  }
}

// ROS callback for step commands
void stepCommandCallback(const std_msgs::Int32& msg) {
  targetSteps = msg.data;
  moveMotor(targetSteps, DEFAULT_SPEED);
}

// ROS callback for enable/disable
void enableCommandCallback(const std_msgs::Bool& msg) {
  enableMotor(msg.data);
}

// ROS callback for home command
void homeCommandCallback(const std_msgs::Bool& msg) {
  if (msg.data) {
    homeMotor();
  }
}

// ROS subscribers
ros::Subscriber<std_msgs::Int32> step_sub("motor_step_command", &stepCommandCallback);
ros::Subscriber<std_msgs::Bool> enable_sub("motor_enable", &enableCommandCallback);
ros::Subscriber<std_msgs::Bool> home_sub("motor_home", &homeCommandCallback);

void setup() {
  // Initialize pins
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(M0_PIN, OUTPUT);
  pinMode(M1_PIN, OUTPUT);
  pinMode(M2_PIN, OUTPUT);

  // Set microstepping mode
  setMicrostepping();

  // Initialize ROS with explicit baud rate
  nh.getHardware()->setBaud(57600);
  nh.initNode();
  nh.advertise(position_pub);
  nh.advertise(status_pub);
  nh.subscribe(step_sub);
  nh.subscribe(enable_sub);
  nh.subscribe(home_sub);

  // Initial state
  motorEnabled = true;
  currentPosition = 0;

  // Wait for stable ROS connection
  while (!nh.connected()) {
    nh.spinOnce();
    delay(500);  // Increased delay for more stable connection
  }

  // Additional delay to ensure full negotiation
  delay(1000);

  // Now publish initial status
  publishStatus("Arduino stepper control ready");
  delay(100);
  publishPosition();
}

void loop() {
  nh.spinOnce();

  // Only publish if connected
  if (nh.connected()) {
    // Publish position periodically
    static unsigned long lastPublish = 0;
    if (millis() - lastPublish > 1000) {
      publishPosition();
      lastPublish = millis();
    }
  }

  delay(10);
}
