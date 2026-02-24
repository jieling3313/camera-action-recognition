# D435i Camera Tracking System with Arduino Stepper Motor Control

## 系統概述

本系統整合 Intel RealSense D435i 相機、YOLO 人體檢測、PID 控制器與 Arduino UNO + DRV8825 步進馬達驅動器，實現自動追蹤人體並保持在畫面中心的功能。

### 主要功能
- 使用 YOLO 檢測畫面中最大面積的人體
- PID 控制器計算人體偏移量並控制步進馬達
- 步進馬達驅動 D435i 相機雲台，保持人體在畫面橫向中心
- 支援啟動/停止追蹤、歸零等安全功能

---

## 硬體連接

### Arduino UNO + DRV8825 + NEMA17 步進馬達

| DRV8825 | Arduino UNO | 說明 |
|---------|-------------|------|
| DIR     | Pin 9       | 方向控制 |
| STEP    | Pin 10      | 步進脈衝 |
| M0      | Pin 11      | 微步配置 (1/8) |
| M1      | Pin 12      | 微步配置 (1/8) |
| M2      | Pin 13      | 微步配置 (1/8) |
| RESET   | 5V          | 復位 (常態啟用) |
| SLEEP   | 5V          | 休眠 (常態啟用) |
| VMOT    | 12V (外部)  | 馬達電源 |
| GND     | GND (外部)  | 外部電源地 |
| 2B, 2A  | 馬達線圈 1  | NEMA17 步進馬達 |
| 1A, 1B  | 馬達線圈 2  | NEMA17 步進馬達 |

### 步進馬達規格
- **型號**: NEMA17 (或同等規格)
- **步進角度**: 1.8° (200 步/圈)
- **微步設定**: 1/8 微步 (1600 步/圈)
- **額定電壓**: 12V
- **最大電流**: 根據您的馬達規格設定 DRV8825 電流限制

---

## 軟體安裝

### 1. 安裝 Arduino IDE 與 rosserial

```bash
# 安裝 rosserial
sudo apt-get install ros-noetic-rosserial-arduino
sudo apt-get install ros-noetic-rosserial

# 生成 Arduino libraries
cd ~/Arduino/libraries
rosrun rosserial_arduino make_libraries.py .
```

### 2. 上傳 Arduino 程式

1. 開啟 Arduino IDE
2. 開啟檔案: `motor_control/arduino/arduino_stepper_control.ino`
3. 選擇板子: **Tools > Board > Arduino UNO**
4. 選擇序列埠: **Tools > Port > /dev/ttyACM0** (根據實際情況調整)
5. 上傳程式: **Sketch > Upload**

### 3. 編譯 ROS 工作空間

```bash
cd ~/catkin_ws  # 或您的 workspace 路徑
catkin_make

# 如果新增了 srv 檔案，需要重新編譯
source devel/setup.bash
```

### 4. 安裝 Python 依賴

```bash
pip3 install ultralytics opencv-python numpy
```

---

## 使用方法

### 1. 啟動完整系統

```bash
roslaunch yolo_ros camera_tracking.launch
```

這會啟動：
- RealSense D435i 相機節點
- Arduino rosserial 通訊節點
- 相機追蹤節點

### 2. 控制追蹤系統

#### 啟動追蹤
```bash
rostopic pub /tracking_enable std_msgs/Bool "data: true"
```

#### 停止追蹤
```bash
rostopic pub /tracking_enable std_msgs/Bool "data: false"
```

#### 馬達歸零
```bash
rostopic pub /tracking_home std_msgs/Bool "data: true"
```

#### 手動控制馬達 (步數)
```bash
# 順時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: 100"

# 逆時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: -100"
```

### 3. 監控系統狀態

#### 查看馬達位置
```bash
rostopic echo /motor_position
```

#### 查看馬達狀態
```bash
rostopic echo /motor_status
```

#### 查看追蹤影像
```bash
# 方法 1: 使用 image_view
rosrun image_view image_view image:=/tracking_annotated_image

# 方法 2: 使用 rqt_image_view
rqt_image_view
```

---

## 參數調整

### Launch 檔案參數

編輯 `launch/camera_tracking.launch` 可調整以下參數：

```xml
<!-- Arduino 序列埠 -->
<arg name="arduino_port" default="/dev/ttyACM0" />

<!-- 追蹤參數 -->
<arg name="center_deadzone" default="50" />      <!-- 中心死區 (像素) -->
<arg name="max_steps_per_update" default="100" /> <!-- 單次最大步數 -->
<arg name="pixels_per_step" default="1.5" />     <!-- 校正參數 -->

<!-- PID 參數 -->
<arg name="pid_kp" default="2.0" />  <!-- 比例增益 -->
<arg name="pid_ki" default="0.3" />  <!-- 積分增益 -->
<arg name="pid_kd" default="0.1" />  <!-- 微分增益 -->
```

### PID 調整建議

| 現象 | 調整建議 |
|------|---------|
| 追蹤反應太慢 | 增加 `pid_kp` (例如 2.5) |
| 追蹤震盪、不穩定 | 降低 `pid_kp`、增加 `pid_kd` |
| 有穩態誤差 | 增加 `pid_ki` (但要小心震盪) |
| 馬達移動過快 | 降低 `max_steps_per_update` |

### 校正 pixels_per_step

1. 啟動系統但不啟用追蹤
2. 手動移動馬達 100 步
   ```bash
   rostopic pub /motor_step_command std_msgs/Int32 "data: 100"
   ```
3. 觀察畫面中物體的橫向位移像素數
4. 計算: `pixels_per_step = 位移像素數 / 100`
5. 更新 launch 檔案中的參數

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
│  - YOLO 人體檢測            │
│  - 計算偏移量               │
│  - PID 控制                 │
└────────┬────────────────────┘
         │ /motor_step_command
         ▼
┌─────────────────────────────┐
│  rosserial (Arduino)        │
│  - 接收步數指令             │
│  - 控制 DRV8825             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Arduino UNO + DRV8825      │
│  - 步進馬達控制             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  NEMA17 Stepper Motor       │
│  - 驅動相機雲台             │
└─────────────────────────────┘
```

### ROS Topics

| Topic | Type | 說明 |
|-------|------|------|
| `/tracking_enable` | `std_msgs/Bool` | 啟用/停用追蹤 |
| `/tracking_home` | `std_msgs/Bool` | 馬達歸零 |
| `/motor_step_command` | `std_msgs/Int32` | 馬達步數指令 |
| `/motor_enable` | `std_msgs/Bool` | 啟用/停用馬達 |
| `/motor_home` | `std_msgs/Bool` | Arduino 馬達歸零 |
| `/motor_position` | `std_msgs/Int32` | 馬達當前位置 |
| `/motor_status` | `std_msgs/String` | 馬達狀態訊息 |
| `/tracking_annotated_image` | `sensor_msgs/Image` | 追蹤標註影像 |

---

## 故障排除

### Arduino 無法連線

1. 檢查 USB 連線
2. 確認序列埠
   ```bash
   ls -l /dev/ttyACM*
   ```
3. 設定序列埠權限
   ```bash
   sudo chmod 666 /dev/ttyACM0
   # 或永久設定
   sudo usermod -a -G dialout $USER
   # 重新登入後生效
   ```
4. 重新啟動 rosserial
   ```bash
   rosrun rosserial_python serial_node.py /dev/ttyACM0
   ```

### 馬達不轉動

1. 檢查 DRV8825 電源指示燈
2. 確認馬達電源 (12V) 連接正確
3. 檢查馬達線圈連接
4. 測試 Arduino 輸出
   ```bash
   rostopic pub /motor_step_command std_msgs/Int32 "data: 10"
   ```
5. 確認馬達已啟用
   ```bash
   rostopic pub /motor_enable std_msgs/Bool "data: true"
   ```

### 追蹤不穩定

1. 調整 PID 參數 (降低 Kp)
2. 增加中心死區 `center_deadzone`
3. 降低最大步數 `max_steps_per_update`
4. 檢查相機固定是否穩固
5. 確認照明條件良好

### YOLO 檢測不到人

1. 確認光線充足
2. 人體與相機距離適中 (1-3 公尺)
3. 檢查 YOLO 模型路徑
4. 查看原始影像
   ```bash
   rosrun image_view image_view image:=/camera/color/image_raw
   ```

---

## 安全注意事項

1. **電源安全**: 確保 12V 外部電源極性正確
2. **運動範圍**: 初次測試時使用小步數，避免線材纏繞
3. **機械限位**: 建議加裝硬體限位開關
4. **急停機制**: 隨時準備斷電或停止程式
5. **穩定固定**: 確保相機和馬達安裝牢固

---

## 進階功能

### 角度限制 (可選)

編輯 `arduino_stepper_control.ino` 添加軟體限位：

```cpp
const long MAX_POSITION = 800;   // +90度 (1600步/360度)
const long MIN_POSITION = -800;  // -90度

// 在 moveMotor() 函數中添加
if (currentPosition + steps > MAX_POSITION ||
    currentPosition + steps < MIN_POSITION) {
    publishStatus("Error: Position limit");
    return;
}
```

### 多人追蹤

修改 `camera_tracker_node.py` 的 `find_largest_person()` 函數，可以改為追蹤特定人物或平均多人位置。

---

## 檔案結構

```
motor_control/
├── arduino/
│   └── arduino_stepper_control.ino  # Arduino 主程式
├── scripts/
│   └── camera_tracker_node.py       # ROS 追蹤節點
├── README.md                         # 本檔案
└── srv/
    └── MotorControl.srv             # ROS service 定義 (保留供未來使用)
```

---

## 授權

本專案為教育與研究用途。

## 聯絡資訊

如有問題或建議，請聯繫專案維護者。

---

**祝您使用愉快！**
