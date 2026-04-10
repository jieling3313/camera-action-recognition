# ROS w/Arduino

## 功能 & 流程
### 功能
| 功能             | 說明                               |
|:---------------- |:---------------------------------- |
| **人體追蹤**     | YOLO 檢測人體並用 PID 控制馬達追蹤 |
| **最大人體模式** | 追蹤畫面中面積最大的人體           |
| **巡航模式**     | 無人時自動旋轉搜尋                 |
| **Qt5 GUI**      | 控制視窗                           |
| **ROS 整合**     | ROS Topic 和 Service 介面          |

### 系統流程

```
D435i RGB 影像 (30fps)
         ↓
    YOLO 人體檢測
         ↓
    計算人體中心偏移量
         ↓
    PID 控制器
         ↓
    步進馬達指令
         ↓
   Arduino + DRV8825
         ↓
   NEMA17 步進馬達
         ↓
   相機轉動
```

---

## 硬體

### 電路圖

![alt text](image.png)

---

## 軟體安裝
## 環境設定
### dockerfile 新增 (`.devcontainer/Dockerfile`)
安裝 rosserial 用於 Arduino 通訊 (支援步進馬達控制)

```dockerfile    
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial \
    ros-noetic-rosserial-python \
    python3-serial \
    && rm -rf /var/lib/apt/lists/*
```

### ROS Noetic + Python 3.10
- ROS 序列庫: `ros-noetic-rosserial`
- 支援 Arduino 開發板: `ros-noetic-rosserial-arduino` - 
- ROS - python 序列庫:`ros-noetic-rosserial-python`
- Python 序列通訊庫:`python3-serial`

### 驗證安裝
```bash
docker exec -it ros-noetic-yolo-dev bash

# 檢查 rosserial 套件
rospack find rosserial
rospack find rosserial_arduino
rospack find rosserial_python

# 檢查 Python serial 模組
python3.10 -c "import serial; print(f'pyserial version: {serial.VERSION}')"

# 預期輸出:
# /opt/ros/noetic/share/rosserial
# /opt/ros/noetic/share/rosserial_arduino
# /opt/ros/noetic/share/rosserial_python
# pyserial version: 3.x.x
```
==早期ROS Noetic== 的 rosserial 有 Python 2 到 Python 3 遷移問題，但官方已在正式版本中修正。
- [rosserial-python on ROS noetic - ROS Answers](https://answers.ros.org/question/355801/rosserial-python-on-ros-noetic/)
- [Noetic release · Issue #499 · ros-drivers/rosserial](https://github.com/ros-drivers/rosserial/issues/499)



## 初始設定(container中)
### 生成 ros_lib
前置設定(以包括在dockerfile之中): `rosserial_arduino`

#### 載入ROS環境
```bash
source /opt/ros/noetic/setup.bash
```
#### 建立 Arduino 函式庫目錄
```
mkdir -p /root/Arduino/libraries
```
#### 生成 ros_lib
```bash
rosrun rosserial_arduino make_libraries.py
```
- 執行 ROS 的 `make_libraries.py` 工具
- 自動掃描系統中所有 ROS 訊息類型（`std_msgs`, `sensor_msgs`, `geometry_msgs` 等）
- 為每個訊息類型生成對應的 C++ 標頭檔（.h）
- 將這些檔案放入 `ros_lib` 目錄

## Arduino-ROS 設定

### 下載arduino (本機內)
```bash
sudo apt-get update
sudo apt-get install arduino
```


### 開啟Arduino做後續設定

```bash
arduino
```

Tools >> Manage Libraries >> 下載Rosserial Arduino Library
![image](https://hackmd.io/_uploads/B1hYoyoO-e.png)

### ==將 container 中的 ros_lib 複製到主機==
```
docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/
```

開啟檔案 `/ros-yolo-opencv-project3/src/yolo_ros/motor_control/arduino/arduino_stepper_control/arduino_stepper_control.ino`

選擇開發板 & 選擇Port & 上傳程式

:::warning

###  設定 USB 序列埠權限 
```bash
# 臨時設定
sudo chmod 666 /dev/ttyACM0

# 永久設定 (推薦)
sudo usermod -a -G dialout $USER
# !!! 設定後需重新登入
```

:::

### 編譯 ROS (container內)
#### 編譯
```bash
cd /root/catkin_ws
catkin_make
```

#### 重新載入環境 (container內)
```
source devel/setup.bash
```

#### 檢查 Arduino 連線 (container內)

```bash
# 在容器內啟動 roscore
roscore

# 在另一個終端，啟動 rosserial
source /root/catkin_ws/devel/setup.bash
rosrun rosserial_python serial_node.py /dev/ttyACM0

# 應該看到:
# [INFO] ROS Serial Python Node
# [INFO] Connecting to /dev/ttyACM0 at 57600 baud
```

#### 檢查 ROS Topics (container內)

```bash
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash

# 查看 topics
rostopic list | grep motor

# 應該看到:
#   /motor_enable
#   /motor_home
#   /motor_position
#   /motor_status
#   /motor_step_command
```

### 測試馬達移動 (container內)

```bash
# 確保 rosserial 已啟動
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash

# 啟用馬達
rostopic pub /motor_enable std_msgs/Bool "data: true" -1

# 移動 50 步
rostopic pub /motor_step_command std_msgs/Int32 "data: 50" -1

# 查看位置
rostopic echo /motor_position -n 1
```

---

## 人體骨架追蹤 (快速啟動)

```bash
# 進入容器
cd /root/catkin_ws/src/yolo_ros

# 測試馬達連接
make test-motor

# 啟動完整追蹤系統 (相機+追蹤+GUI)
make start-body-tracking

# 只啟動追蹤節點 (無 GUI)
make start-tracker-only

# 查看操作指南
make guide-body-tracking
```

### 使用 roslaunch

```bash
# 啟動完整系統
roslaunch yolo_ros body_tracking_system.launch

# 自訂參數
roslaunch yolo_ros body_tracking_system.launch \
    arduino_port:=/dev/ttyACM0 \
    launch_gui:=true \
    pid_kp:=2.0
```

### 手動控制

```bash
# 啟動追蹤
rostopic pub /tracking_enable std_msgs/Bool "data: true"

# 停止追蹤
rostopic pub /tracking_enable std_msgs/Bool "data: false"

# 馬達歸零
rostopic pub /tracking_home std_msgs/Bool "data: true"

# 手動移動馬達 (正數順時針，負數逆時針)
rostopic pub /motor_step_command std_msgs/Int32 "data: 100"
rostopic pub /motor_step_command std_msgs/Int32 "data: -100"
```

---

## Qt5 控制界面

### 啟動 GUI

```bash
# 方式 1: 使用 launch 檔案
roslaunch yolo_ros body_tracking_system.launch

# 方式 2: 單獨啟動 GUI
rosrun yolo_ros body_tracking_gui.py
```

### GUI 功能

![image](https://hackmd.io/_uploads/HyNYz1_oWe.png)

### 說明

| 功能                     | 說明                              |
|:------------------------ | --------------------------------- |
| **Start/Stop Tracking**  | 開啟/關閉人體追蹤                 |
| **Home Motor**           | 馬達回到原點位置                  |
| **Track Largest Person** | 追蹤畫面中面積最大的人體          |
| **Auto-Patrol Mode**     | 偵測不到人超過 3 秒時自動開始巡航 |
| **Patrol Speed**         | 調整巡航旋轉速度 (10-90 °/秒)     |
| **Start/Stop Patrol**    | 手動開始/停止巡航                 |
| **Open Camera View**     | 開啟追蹤標註影像視窗              |

---

## ROS Topics

### 輸入 Topics (發送)

| Topic                   | Type               | 說明                |
|:----------------------- |:------------------ |:------------------- |
| `/tracking_enable`      | `std_msgs/Bool`    | 啟用/停用追蹤       |
| `/tracking_home`        | `std_msgs/Bool`    | 馬達歸零            |
| `/motor_step_command`   | `std_msgs/Int32`   | 馬達步數指令        |
| `/motor_enable`         | `std_msgs/Bool`    | 啟用/停用馬達       |
| `/patrol_enable`        | `std_msgs/Bool`    | 啟用/停用手動巡航   |
| `/auto_patrol_enable`   | `std_msgs/Bool`    | 啟用/停用自動巡航   |
| `/patrol_speed`         | `std_msgs/Float32` | 設定巡航速度 (°/秒) |
| `/track_largest_person` | `std_msgs/Bool`    | 追蹤最大人體開關    |

### 輸出 Topics (接收)

| Topic                       | Type                | 說明                |
|:--------------------------- |:------------------- |:------------------- |
| `/motor_position`           | `std_msgs/Int32`    | 馬達當前位置 (步數) |
| `/motor_status`             | `std_msgs/String`   | 馬達狀態訊息        |
| `/tracking_annotated_image` | `sensor_msgs/Image` | 追蹤標註影像        |
| `/body_tracking_status`     | `std_msgs/String`   | 系統狀態資訊        |
| `/detected_person_count`    | `std_msgs/Int32`    | 偵測到的人數        |

### 監控範例

```bash
# 監控追蹤狀態
make monitor-tracking

# 或直接使用 rostopic
rostopic echo /body_tracking_status

# 查看追蹤影像
rosrun image_view image_view image:=/tracking_annotated_image
```

---

## 參數調整

### Launch 檔案參數

編輯 `launch/body_tracking_system.launch`:

```xml
<!-- Arduino 序列埠 -->
<arg name="arduino_port" default="/dev/ttyACM0" />
<arg name="arduino_baud" default="57600" />

<!-- 追蹤參數 -->
<arg name="center_deadzone" default="50" />       <!-- 中心死區 (像素) -->
<arg name="max_steps_per_update" default="100" /> <!-- 單次最大步數 -->
<arg name="pixels_per_step" default="1.5" />      <!-- 校正參數 -->

<!-- PID 參數 -->
<arg name="pid_kp" default="2.0" />   <!-- 比例增益 -->
<arg name="pid_ki" default="0.3" />   <!-- 積分增益 -->
<arg name="pid_kd" default="0.1" />   <!-- 微分增益 -->

<!-- 巡航參數 -->
<arg name="patrol_speed" default="36.0" />        <!-- 巡航速度 (°/秒) -->
<arg name="steps_per_revolution" default="1600" /> <!-- 每圈步數 -->
<arg name="no_person_timeout" default="3.0" />    <!-- 無人超時 (秒) -->

<!-- GUI 選項 -->
<arg name="launch_gui" default="true" />
<arg name="launch_viewer" default="false" />
```

### 校正 pixels_per_step

1. 啟動系統但不啟用追蹤
2. 手動移動馬達 100 步:
   ```bash
   rostopic pub /motor_step_command std_msgs/Int32 "data: 100"
   ```
3. 觀察畫面中物體的橫向位移像素數
4. 計算: `pixels_per_step = 位移像素數 / 100`
5. 更新 launch 檔案中的參數

### 角度限制

`arduino/arduino_stepper_control.ino`

```cpp
const long MAX_POSITION = 800;   // +90度 (1600步/360度)
const long MIN_POSITION = -800;  // -90度

if (currentPosition + steps > MAX_POSITION ||
    currentPosition + steps < MIN_POSITION) {
    publishStatus("Error: Position limit");
    return;
}
```

### 多人追蹤策略

`camera_tracker_node.py` 支援以下策略:

| 策略 | 說明 |
|------|------|
| **追蹤最大人體** | 依 Bounding Box 面積判斷 |
| **追蹤最近人體** | 依 Bounding Box Y 座標判斷 |
| **追蹤第一個人體** | 追蹤最先被檢測到的人體 |

### 自訂追蹤行為

修改 `scripts/camera_tracker_node.py` 中的 `select_target_person()` 函數:

```python
def select_target_person(self, detections):
    if self.track_largest:
        # 追蹤最大面積
        return max(detections, key=lambda d: d.area)
    else:
        # 追蹤最近的人 (Y 座標最大)
        return max(detections, key=lambda d: d.bbox[3])
```

---

## 系統架構

```
┌─────────────────┐
│  D435i Camera   │
└────────┬────────┘
         │ /camera/color/image_raw
         ▼
┌─────────────────────────────┐
│  camera_tracker_node.py     │
│  ├─ YOLO 人體檢測            │
│  ├─ 計算偏移量                │
│  ├─ PID 控制                 │
│  └─ 巡航模式邏輯              │
└────────┬────────────────────┘
         │ /motor_step_command
         ▼
┌─────────────────────────────┐
│  rosserial (Arduino)        │
│  - 接收步數指令               │
│  - 發布馬達狀態               │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Arduino UNO + DRV8825      │
│  - 步進脈衝生成               │
│  - 微步控制                  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  NEMA17 Stepper Motor       │
│  - 驅動相機雲台               │
└─────────────────────────────┘
```


---

## 檔案結構

```
motor_control/
├── arduino/
│   └── arduino_stepper_control/
│       └── arduino_stepper_control.ino  # Arduino 主程式
├── scripts/
│   └── camera_tracker_node.py           # ROS 追蹤節點 (含巡航模式)
└── README.md                             # 本檔案

# Qt5 GUI 相關檔案
qt_gui/
└── body_tracking_window.py              # Body Tracking 控制視窗

scripts/
└── body_tracking_gui.py                 # GUI 啟動腳本

launch/
└── body_tracking_system.launch          # 完整系統啟動檔案
```




---
## Ardunio 程式碼
檔案位置: `ros-yolo-opencv-project3/src/yolo_ros/motor_control/arduino/arduino_stepper_control/arduino_stepper_control.ino`
```arduino
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

```


### 簡易控制馬達
```bash
rostopic pub /tracking_enable std_msgs/Bool "data: true" -1   # 啟動
rostopic pub /tracking_enable std_msgs/Bool "data: false" -1  # 停止
rostopic pub /tracking_home std_msgs/Bool "data: true" -1     # 歸零
```

```bash
# 順時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: 100" -1

# 逆時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: -100" -1
```

```bash
# 查看馬達位置
rostopic echo /motor_position

# 查看馬達狀態
rostopic echo /motor_status

# 查看追蹤影像
rosrun image_view image_view image:=/tracking_annotated_image
```
