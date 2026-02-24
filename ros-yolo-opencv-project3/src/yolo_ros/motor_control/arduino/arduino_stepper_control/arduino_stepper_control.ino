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
 * Motor: NEMA17 1.8°/step, 200 steps/revolution
 * Microstepping: 1/8 (1600 steps/revolution)
 */

#include <ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

// Pin definitions
const int DIR_PIN = 9;
const int STEP_PIN = 10;
const int M0_PIN = 11;
const int M1_PIN = 12;
const int M2_PIN = 13;

// Motor parameters
const int STEPS_PER_REV = 1600;  // 200 * 8 (1/8 microstepping)
const int DEFAULT_SPEED = 800;   // steps/sec
const int MAX_SPEED = 2000;      // Maximum speed limit

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
  // For 1/8 microstepping: M0=HIGH, M1=LOW, M2=LOW
  digitalWrite(M0_PIN, HIGH);
  digitalWrite(M1_PIN, LOW);
  digitalWrite(M2_PIN, LOW);
}

// Move motor by specified steps
void moveMotor(long steps, int speed = DEFAULT_SPEED) {
  if (!motorEnabled) {
    publishStatus("Error: Motor disabled");
    return;
  }

  // Limit speed
  speed = constrain(speed, 100, MAX_SPEED);
  long delayMicros = 1000000L / speed;

  // Set direction
  bool dir = (steps >= 0);
  digitalWrite(DIR_PIN, dir ? HIGH : LOW);

  long absSteps = abs(steps);

  for (long i = 0; i < absSteps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(delayMicros / 2);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(delayMicros / 2);

    // Update position
    currentPosition += dir ? 1 : -1;

    // Publish position every 100 steps
    if (i % 100 == 0) {
      publishPosition();
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
  position_msg.data = currentPosition;
  position_pub.publish(&position_msg);
}

// Publish status message
void publishStatus(const char* message) {
  status_msg.data = message;
  status_pub.publish(&status_msg);
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

  // Initialize ROS
  nh.initNode();
  nh.advertise(position_pub);
  nh.advertise(status_pub);
  nh.subscribe(step_sub);
  nh.subscribe(enable_sub);
  nh.subscribe(home_sub);

  // Initial state
  motorEnabled = true;
  currentPosition = 0;

  // Wait for ROS connection
  while (!nh.connected()) {
    nh.spinOnce();
    delay(100);
  }

  publishStatus("Arduino stepper control ready");
  publishPosition();
}

void loop() {
  nh.spinOnce();
  delay(10);

  // Publish position periodically
  static unsigned long lastPublish = 0;
  if (millis() - lastPublish > 1000) {
    publishPosition();
    lastPublish = millis();
  }
}
