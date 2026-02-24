# Docker 環境設定說明

## 🎯 更新內容

已將 rosserial 相關套件整合到 Dockerfile 中，避免每次手動安裝。

---

## 📦 Dockerfile 新增內容

在 `.devcontainer/Dockerfile` 中新增了以下內容：

```dockerfile
# 安裝 rosserial 用於 Arduino 通訊 (支援步進馬達控制)
# 這些套件已針對 ROS Noetic 和 Python 3 優化，無衝突
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial \
    ros-noetic-rosserial-python \
    python3-serial \
    && rm -rf /var/lib/apt/lists/*
```

---

## ✅ 相容性確認

### ROS Noetic + Python 3.10
- ✅ `ros-noetic-rosserial` - 官方正式版本
- ✅ `ros-noetic-rosserial-arduino` - 支援 Arduino UNO
- ✅ `ros-noetic-rosserial-python` - 已修正 Python 3 相關問題
- ✅ `python3-serial` - Python 序列通訊庫

### 已知問題
早期 ROS Noetic 的 rosserial 有 Python 2 到 Python 3 遷移問題，但**官方已在正式版本中修正**。

參考來源：
- [rosserial-python on ROS noetic - ROS Answers](https://answers.ros.org/question/355801/rosserial-python-on-ros-noetic/)
- [Noetic release · Issue #499 · ros-drivers/rosserial](https://github.com/ros-drivers/rosserial/issues/499)

---

## 🚀 使用方式

### 方法 1: 重建 Docker 映像 (推薦)

```bash
cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer

# 停止現有容器
docker-compose down

# 重建映像（包含 rosserial）
docker-compose build

# 啟動新容器
docker-compose up -d
```

### 方法 2: 無需重建（如果容器已存在）

如果不想重建整個容器，可以在現有容器內手動安裝：

```bash
docker exec -it ros-noetic-yolo-dev bash

# 在容器內執行
apt-get update
apt-get install -y ros-noetic-rosserial-arduino ros-noetic-rosserial ros-noetic-rosserial-python python3-serial
```

---

## 🔍 驗證安裝

在容器內執行以下指令驗證：

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

---

## 📁 完整安裝流程

詳細的完整安裝流程請參閱 `INSTALLATION.md`，主要步驟：

1. **重建 Docker 容器** - rosserial 自動安裝 ✅
2. **宿主機安裝 Arduino IDE** - 需手動安裝
3. **容器內生成 ros_lib** - 執行 `setup_arduino_libs.sh`
4. **複製 ros_lib 到宿主機** - 使用 `docker cp`
5. **上傳 Arduino 程式** - 使用 Arduino IDE
6. **設定 USB 權限** - 宿主機執行
7. **啟動系統** - `roslaunch yolo_ros camera_tracking.launch`

---

## 🛠️ 便利工具

### 自動生成 Arduino Libraries

在容器內執行：

```bash
/root/catkin_ws/src/yolo_ros/motor_control/scripts/setup_arduino_libs.sh
```

這個腳本會：
1. 檢查 rosserial 是否已安裝
2. 建立 Arduino libraries 目錄
3. 生成 ros_lib
4. 驗證生成結果
5. 顯示後續步驟

---

## 🔄 容器與宿主機的關係

```
┌─────────────────────────────────────┐
│      宿主機 (Host Machine)          │
│                                     │
│  ✓ Arduino IDE (手動安裝)           │
│  ✓ ~/Arduino/libraries/ros_lib     │
│    (從容器複製)                     │
│                                     │
│  ✓ USB 裝置                         │
│    - /dev/ttyACM0 (Arduino)        │
│    - /dev/bus/usb/* (D435i)        │
└──────────────┬──────────────────────┘
               │
               │ USB passthrough
               │ Volume mount: /dev:/dev
               │
┌──────────────▼──────────────────────┐
│    Docker 容器 (ros-noetic-yolo-dev)│
│                                     │
│  ✓ ROS Noetic                       │
│  ✓ rosserial (Dockerfile 安裝) ✅   │
│  ✓ Python 3.10 + ultralytics        │
│  ✓ /root/catkin_ws/src              │
│    (掛載自宿主機專案目錄)           │
│                                     │
│  ✓ 可訪問 /dev/ttyACM0              │
│    (透過 docker-compose.yml)        │
└─────────────────────────────────────┘
```

---

## 📝 注意事項

1. **Arduino IDE 必須在宿主機安裝**
   - 容器內無法直接上傳程式到 Arduino
   - 需要 GUI 環境

2. **ros_lib 需要從容器複製到宿主機**
   - 生成: 容器內執行 `rosrun rosserial_arduino make_libraries.py .`
   - 複製: `docker cp ros-noetic-yolo-dev:/root/Arduino/libraries/ros_lib ~/Arduino/libraries/`

3. **USB 裝置權限**
   - 宿主機需設定: `sudo usermod -a -G dialout $USER`
   - docker-compose.yml 已設定 `/dev:/dev` 掛載

4. **容器重建後需重新生成 ros_lib**
   - Dockerfile 只安裝 rosserial 套件
   - ros_lib 需要手動生成（因為每個專案的訊息類型不同）

---

## 🎉 完成後

系統架構：
- ✅ Docker 容器內運行 ROS 追蹤節點
- ✅ 容器內運行 rosserial 通訊
- ✅ Arduino 透過 USB 連接到容器
- ✅ D435i 相機透過 USB 連接到容器
- ✅ 步進馬達控制完整整合

詳細使用說明：
- **安裝**: `INSTALLATION.md`
- **快速開始**: `QUICKSTART.md`
- **使用手冊**: `README.md`
