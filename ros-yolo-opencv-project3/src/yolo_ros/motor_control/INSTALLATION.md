# 安裝指南 - Arduino 步進馬達追蹤系統

本系統使用 Docker 容器化開發環境，大部分依賴已在 Dockerfile 中預先安裝。

---

## 📦 架構說明

```
┌─────────────────────────────────────────────────┐
│          宿主機 (Host Machine)                  │
│  - Arduino IDE (需手動安裝)                     │
│  - USB 連接 Arduino UNO                         │
│  - USB 連接 D435i 相機                          │
└──────────────┬──────────────────────────────────┘
               │
               │ USB passthrough (/dev 掛載)
               ▼
┌─────────────────────────────────────────────────┐
│     Docker 容器 (ros-noetic-yolo-dev)           │
│  - ROS Noetic + rosserial ✅ (Dockerfile安裝)   │
│  - Python 3.10 + ultralytics ✅                 │
│  - camera_tracker_node.py ✅                    │
└─────────────────────────────────────────────────┘
```

---

## 🚀 快速安裝流程

### Step 1: 重建 Docker 容器 (包含 rosserial)

```bash
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer

# 重建 Docker 映像 (這會自動安裝 rosserial)
docker-compose build

# 啟動容器
docker-compose up -d

# 進入容器
docker exec -it ros-noetic-yolo-dev bash
```

> **注意**: Dockerfile 已更新，包含以下 rosserial 套件：
> - `ros-noetic-rosserial`
> - `ros-noetic-rosserial-arduino`
> - `ros-noetic-rosserial-python`
> - `python3-serial`

### Step 2: 在宿主機安裝 Arduino IDE

Arduino IDE **必須在宿主機（非容器內）安裝**，用於上傳程式到 Arduino。

#### 方法 1: 使用 Snap (推薦)
```bash
sudo snap install arduino
```

#### 方法 2: 下載官方版本
```bash
# 下載 Arduino IDE 2.x
wget https://downloads.arduino.cc/arduino-ide/arduino-ide_2.3.2_Linux_64bit.AppImage

# 設定執行權限
chmod +x arduino-ide_2.3.2_Linux_64bit.AppImage

# 執行
./arduino-ide_2.3.2_Linux_64bit.AppImage
```

#### 方法 3: APT 安裝 (舊版)
```bash
sudo apt-get update
sudo apt-get install arduino
```

### Step 3: 在容器內生成 Arduino Libraries

容器啟動後，在**容器內**執行：

```bash
# 進入容器
docker exec -it ros-noetic-yolo-dev bash

# 在容器內執行
cd /root
mkdir -p Arduino/libraries
cd Arduino/libraries

# 生成 ros_lib
rosrun rosserial_arduino make_libraries.py .

# 確認生成成功
ls -la ros_lib/
```

### Step 4: 將 ros_lib 複製到宿主機

**在容器內**執行：

```bash
# 假設您的 Arduino libraries 路徑在宿主機的 ~/Arduino/libraries
# 容器內的路徑會掛載到宿主機

# 方法 1: 直接複製到掛載的 src 目錄
cp -r /root/Arduino/libraries/ros_lib /root/catkin_ws/src/yolo_ros/motor_control/arduino/

# 然後在宿主機執行
exit
```

**在宿主機**執行：

```bash
# 複製 ros_lib 到 Arduino libraries
cp -r ~/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/src/yolo_ros/motor_control/arduino/ros_lib ~/Arduino/libraries/

# 或直接從容器複製
docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/
```

### Step 5: 上傳 Arduino 程式 (在宿主機)

1. 在**宿主機**開啟 Arduino IDE
2. **File** > **Open** > 選擇：
   ```
   ~/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/src/yolo_ros/motor_control/arduino/arduino_stepper_control.ino
   ```
3. **Tools** > **Board** > **Arduino AVR Boards** > **Arduino UNO**
4. **Tools** > **Port** > 選擇 **/dev/ttyACM0** (或您的 Arduino 埠)
5. 點擊 **Upload** 按鈕 (→)

### Step 6: 設定 USB 序列埠權限 (宿主機)

```bash
# 臨時設定
sudo chmod 666 /dev/ttyACM0

# 永久設定 (推薦)
sudo usermod -a -G dialout $USER
# ⚠️ 設定後需重新登入
```

### Step 7: 編譯 ROS 工作空間 (容器內)

```bash
# 進入容器
docker exec -it ros-noetic-yolo-dev bash

# 編譯
cd /root/catkin_ws
catkin_make

# 重新載入環境
source devel/setup.bash
```

---

## ✅ 驗證安裝

### 測試 1: 檢查 rosserial 安裝 (容器內)

```bash
docker exec -it ros-noetic-yolo-dev bash

# 檢查 rosserial 套件
rospack find rosserial
rospack find rosserial_python
rospack find rosserial_arduino

# 檢查 Python serial 模組
python3.10 -c "import serial; print(serial.VERSION)"
```

### 測試 2: 檢查 Arduino 連線 (容器內)

```bash
# 在容器內啟動 roscore
docker exec -it ros-noetic-yolo-dev bash
roscore &

# 在另一個終端，啟動 rosserial
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash
rosrun rosserial_python serial_node.py /dev/ttyACM0

# 應該看到:
# [INFO] ROS Serial Python Node
# [INFO] Connecting to /dev/ttyACM0 at 57600 baud
```

### 測試 3: 檢查 ROS Topics (容器內)

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

### 測試 4: 測試馬達移動 (容器內)

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

## 🎯 啟動完整系統

### 在容器內執行：

```bash
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash

# 啟動追蹤系統
roslaunch yolo_ros camera_tracking.launch

# 在另一個終端啟動追蹤
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash
rostopic pub /tracking_enable std_msgs/Bool "data: true" -1
```

---

## 📝 便利腳本

為了簡化操作，已提供以下腳本：

### 生成 Arduino Libraries (容器內執行)

```bash
# 在容器內
/root/catkin_ws/src/yolo_ros/motor_control/scripts/setup_arduino_libs.sh
```

---

## 🔍 故障排除

### 問題 1: rosserial 找不到

**症狀**: `rospack find rosserial` 返回錯誤

**解決方法**:
```bash
# 檢查是否在容器內
docker exec -it ros-noetic-yolo-dev bash

# 重新編譯
cd /root/catkin_ws
catkin_make
source devel/setup.bash
```

### 問題 2: Arduino 無法連線

**症狀**: `serial_node.py` 報錯 "Permission denied"

**解決方法** (在宿主機):
```bash
# 設定權限
sudo chmod 666 /dev/ttyACM0

# 或永久設定
sudo usermod -a -G dialout $USER
# 重新登入
```

### 問題 3: ros_lib 找不到

**症狀**: Arduino IDE 編譯時找不到 `ros.h`

**解決方法** (在宿主機):
```bash
# 從容器複製 ros_lib
docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/

# 重啟 Arduino IDE
```

### 問題 4: Docker 容器內看不到 /dev/ttyACM0

**症狀**: 容器內 `ls /dev/ttyACM*` 沒有輸出

**解決方法**:
```bash
# 檢查 docker-compose.yml 是否有掛載 /dev
# 應該有這一行:
#   - /dev:/dev

# 重啟容器
docker-compose down
docker-compose up -d
```

### 問題 5: Python serial 模組衝突

**症狀**: ImportError 或 ModuleNotFoundError

**解決方法** (在容器內):
```bash
# 重新安裝 pyserial
pip3 uninstall serial pyserial
pip3 install pyserial
```

---

## 🔄 更新 Dockerfile 後的步驟

如果修改了 Dockerfile：

```bash
# 在宿主機
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer

# 停止並刪除舊容器
docker-compose down

# 重建映像
docker-compose build --no-cache

# 啟動新容器
docker-compose up -d

# 重新生成 Arduino libraries (見 Step 3)
```

---

## 📚 檔案位置對照表

| 位置 | 檔案 | 說明 |
|------|------|------|
| 宿主機 | `~/Arduino/libraries/ros_lib/` | Arduino IDE 使用的 ROS library |
| 容器內 | `/root/Arduino/libraries/ros_lib/` | 容器生成的 ros_lib |
| 宿主機 | `~/Desktop/.../motor_control/arduino/arduino_stepper_control.ino` | Arduino 程式碼 |
| 容器內 | `/root/catkin_ws/src/yolo_ros/motor_control/` | 掛載的專案目錄 |

---

## ⚡ 總結：最簡安裝流程

```bash
# 1. 宿主機：重建容器
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer
docker-compose build && docker-compose up -d

# 2. 宿主機：安裝 Arduino IDE
sudo snap install arduino

# 3. 容器內：生成 ros_lib
docker exec -it ros-noetic-yolo-dev bash -c "cd /root && mkdir -p Arduino/libraries && cd Arduino/libraries && rosrun rosserial_arduino make_libraries.py ."

# 4. 宿主機：複製 ros_lib
docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/

# 5. 宿主機：上傳 Arduino 程式 (使用 Arduino IDE)

# 6. 宿主機：設定序列埠權限
sudo usermod -a -G dialout $USER
# 重新登入

# 7. 容器內：編譯並啟動
docker exec -it ros-noetic-yolo-dev bash
source /root/catkin_ws/devel/setup.bash
roslaunch yolo_ros camera_tracking.launch
```

---

完成安裝後，請參閱：
- **使用說明**: `motor_control/README.md`
- **快速開始**: `motor_control/QUICKSTART.md`

如有問題，請使用 `test_system.sh` 進行系統診斷。
