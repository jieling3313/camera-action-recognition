# 快速開始指南 (Quick Start)

## 📦 已建立的檔案

```
motor_control/
├── arduino/
│   └── arduino_stepper_control.ino      # Arduino 步進馬達控制程式
├── scripts/
│   ├── camera_tracker_node.py           # ROS 相機追蹤節點
│   ├── enable_tracking.sh               # 追蹤控制腳本
│   └── test_system.sh                   # 系統測試腳本
├── INSTALLATION.md                       # 詳細安裝指南
├── README.md                            # 完整使用說明
└── QUICKSTART.md                        # 本檔案

其他檔案:
├── srv/MotorControl.srv                 # ROS service 定義
└── launch/camera_tracking.launch        # Launch 檔案
```

---

## ⚡ 30 秒快速啟動

如果您已經完成硬體連接和軟體安裝：

```bash
# 1. 啟動完整系統
roslaunch yolo_ros camera_tracking.launch

# 2. 在新終端機啟動追蹤
rostopic pub /tracking_enable std_msgs/Bool "data: true" -1

# 3. 查看追蹤影像
rosrun image_view image_view image:=/tracking_annotated_image
```

---

## 🔧 完整安裝步驟 (首次使用)

### Step 1: 硬體連接

**Arduino UNO ↔ DRV8825 連接**:
```
DIR   → Arduino Pin 9
STEP  → Arduino Pin 10
M0    → Arduino Pin 11
M1    → Arduino Pin 12
M2    → Arduino Pin 13
RESET → 5V
SLEEP → 5V
```

**DRV8825 ↔ 電源/馬達**:
```
VMOT  → 12V 外部電源 (+)
GND   → 外部電源 (-)
2B, 2A, 1A, 1B → NEMA17 步進馬達
```

### Step 2: 安裝軟體依賴

```bash
# 安裝 rosserial
sudo apt-get install -y ros-noetic-rosserial-arduino ros-noetic-rosserial

# 生成 Arduino libraries
cd ~/Arduino/libraries
rosrun rosserial_arduino make_libraries.py .

# 安裝 Python 依賴
pip3 install ultralytics opencv-python numpy
```

### Step 3: 上傳 Arduino 程式

1. 開啟 Arduino IDE
2. 開啟 `motor_control/arduino/arduino_stepper_control.ino`
3. 選擇板子: **Arduino UNO**
4. 選擇序列埠: **/dev/ttyACM0**
5. 上傳程式

### Step 4: 設定序列埠權限

```bash
sudo usermod -a -G dialout $USER
# 重新登入後生效
```

### Step 5: 編譯 ROS 工作空間

```bash
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3
catkin_make
source devel/setup.bash
```

### Step 6: 測試系統

```bash
# 啟動完整系統
roslaunch yolo_ros camera_tracking.launch

# 在新終端機執行測試腳本
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/src/yolo_ros/motor_control/scripts
./test_system.sh
```

---

## 🎮 控制指令

### 啟動/停止追蹤

```bash
# 方法 1: 使用便利腳本
./motor_control/scripts/enable_tracking.sh on   # 啟動
./motor_control/scripts/enable_tracking.sh off  # 停止
./motor_control/scripts/enable_tracking.sh home # 歸零

# 方法 2: 使用 rostopic
rostopic pub /tracking_enable std_msgs/Bool "data: true" -1   # 啟動
rostopic pub /tracking_enable std_msgs/Bool "data: false" -1  # 停止
rostopic pub /tracking_home std_msgs/Bool "data: true" -1     # 歸零
```

### 手動控制馬達

```bash
# 順時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: 100" -1

# 逆時針移動 100 步
rostopic pub /motor_step_command std_msgs/Int32 "data: -100" -1
```

### 監控系統狀態

```bash
# 查看馬達位置
rostopic echo /motor_position

# 查看馬達狀態
rostopic echo /motor_status

# 查看追蹤影像
rosrun image_view image_view image:=/tracking_annotated_image
```

---

## 🎯 系統工作原理

```
1. D435i 相機拍攝影像
         ↓
2. YOLO 檢測畫面中最大面積的人體
         ↓
3. 計算人體中心與畫面中心的橫向偏移量
         ↓
4. PID 控制器計算應移動的步數
         ↓
5. 透過 rosserial 發送指令給 Arduino
         ↓
6. Arduino 控制 DRV8825 驅動步進馬達
         ↓
7. 馬達轉動，調整相機角度
         ↓
8. 回到步驟 1（循環）
```

---

## 🔍 故障排除

| 問題 | 解決方法 |
|------|---------|
| Arduino 無法連線 | 檢查 USB 連線、序列埠權限 (`sudo chmod 666 /dev/ttyACM0`) |
| 馬達不轉動 | 確認 12V 電源連接、馬達線圈連接 |
| 追蹤不穩定 | 調整 PID 參數 (降低 Kp)、增加 center_deadzone |
| YOLO 檢測不到人 | 確認光線充足、人體與相機距離適中 (1-3m) |

---

## 📚 詳細文件

- **完整使用說明**: `motor_control/README.md`
- **安裝指南**: `motor_control/INSTALLATION.md`
- **系統測試**: 執行 `./motor_control/scripts/test_system.sh`

---

## 🔧 參數調整

編輯 `launch/camera_tracking.launch` 調整參數：

```xml
<!-- 追蹤參數 -->
<arg name="center_deadzone" default="50" />       <!-- 中心死區 (像素) -->
<arg name="pixels_per_step" default="1.5" />      <!-- 校正參數 -->

<!-- PID 參數 -->
<arg name="pid_kp" default="2.0" />  <!-- 比例增益 -->
<arg name="pid_ki" default="0.3" />  <!-- 積分增益 -->
<arg name="pid_kd" default="0.1" />  <!-- 微分增益 -->
```

| 現象 | 調整建議 |
|------|---------|
| 追蹤反應太慢 | 增加 `pid_kp` |
| 追蹤震盪不穩 | 降低 `pid_kp`、增加 `pid_kd` |
| 馬達移動過快 | 降低 `max_steps_per_update` |

---

## 💡 使用技巧

1. **首次使用**: 先執行 `test_system.sh` 確認所有組件正常
2. **校正系統**: 調整 `pixels_per_step` 參數以獲得最佳追蹤效果
3. **安全測試**: 初次測試時使用小步數，避免線材纏繞
4. **效能優化**: 如果 CPU 負載過高，可以降低 YOLO 檢測頻率

---

## 📞 需要協助？

詳細說明請參閱:
- `motor_control/README.md` - 完整使用手冊
- `motor_control/INSTALLATION.md` - 詳細安裝步驟

---

**祝您使用愉快！** 🚀
