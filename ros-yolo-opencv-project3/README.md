# One-Shot Action Recognition System

基於 ROS Noetic 的即時動作辨識系統，整合 MediaPipe 骨架提取、深度學習模型訓練與 Qt5 人機界面。

## 目錄

1. [系統概述](#系統概述)
2. [環境建置](#環境建置)
3. [快速啟動](#快速啟動)
4. [模型訓練](#模型訓練)
5. [Qt5 GUI 使用](#qt5-gui-使用)
6. [馬達人體追蹤](#馬達人體追蹤)
7. [專案結構](#專案結構)
8. [常見問題](#常見問題)

---

## 系統概述

### 核心功能

| 功能 | 說明 |
|------|------|
| **One-Shot Learning** | 只需 1-5 個示範即可新增動作類別 |
| **MediaPipe 33 點骨架** | 使用 MediaPipe Pose 提取 33 關節點 |
| **NTU RGB+D 預訓練** | 在 60 類動作上預訓練，提升辨識效果 |
| **Qt5 GUI** | 三頁式界面：錄製、管理、辨識 |
| **馬達人體追蹤** | YOLO 人體檢測 + 步進馬達追蹤 |

### 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                         ROS Noetic                               │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  D435i 相機   │  骨架提取     │  動作辨識     │   馬達控制        │
│  realsense2  │  MediaPipe   │  One-Shot    │   Arduino+DRV8825 │
└──────────────┴──────────────┴──────────────┴───────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Qt5 GUI                                    │
│  Page 1: 錄製動作  │  Page 2: 管理模型  │  Page 3: 即時辨識    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 環境建置

### 前置需求

- Ubuntu 20.04
- Docker + Docker Compose
- NVIDIA GPU (RTX 30/40/50 系列)
- Intel RealSense D435i 相機

### 步驟 1：安裝 NVIDIA Container Toolkit

```bash
cd .devcontainer
./setup_gpu.sh
```

### 步驟 2：建置 Docker 容器

```bash
cd .devcontainer
docker compose build
docker compose up -d
```

### 步驟 3：驗證 GPU

```bash
docker exec -it ros-noetic-yolo-dev bash
python3.10 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 外接硬碟掛載

若使用外接硬碟存放資料集，請建立符號連結：

```bash
# 在主機上執行
sudo ln -s /media/jieling/Expansion2 /mnt/external_hdd
```

並在 `docker-compose.yml` 中新增掛載：

```yaml
volumes:
  - /mnt/external_hdd:/media/jieling/Expansion:ro
```

---

## 快速啟動

### 進入容器

```bash
# 使用 Makefile
make up
make shell

# 或直接使用 Docker
docker exec -it ros-noetic-yolo-dev bash
```

### 在容器內使用 Makefile

```bash
cd /root/catkin_ws/src/yolo_ros
make help  # 顯示所有可用指令
```

### 啟動完整 GUI 系統

```bash
make start-gui-system
```

---

## 模型訓練

### MediaPipe 33 點模型 (推薦)

使用從 NTU RGB+D 影片提取的 MediaPipe 骨架進行訓練：

```bash
# 完整訓練 (500 epochs)
make train-mediapipe33

# 快速測試 (10 epochs)
make train-mediapipe33-fast

# 從中斷處繼續訓練
make train-mediapipe33-resume
```

**資料集位置**: `/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons`

**Checkpoint 位置**: `scripts/models/mediapipe33/checkpoints_mediapipe33/`

| 檔案 | 說明 |
|------|------|
| `latest.pth` | 最新模型 (用於斷點續訓) |
| `best.pth` | 最佳驗證準確率模型 |
| `epoch_*.pth` | 每 10 個 epoch 儲存 |

### NTU 25 點模型 (Kinect 格式)

使用原始 NTU RGB+D 骨架格式：

```bash
make train-ntu          # 完整訓練
make train-ntu-fast     # 快速測試
make train-ntu-resume   # 繼續訓練
```

### 訓練參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 500 | 訓練輪數 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 學習率 |
| `--max_frames` | 64 | 骨架序列長度 |
| `--augment` | False | 資料增強 |

---

## Qt5 GUI 使用

### Page 1: 資料收集

錄製動作影片並自動提取骨架：

1. 輸入動作名稱 (例如: `wave_hand`)
2. 點擊 **Start Recording**
3. 執行動作 2-3 秒
4. 點擊 **Stop Recording**

**自動產生檔案**:
- `{action}_video.avi` - 影片
- `{action}_skeleton_sequence.npy` - MediaPipe 33 點骨架
- `{action}_skeleton_sequence_coco17.npy` - COCO 17 點骨架

### Page 2: 模型管理

配置 One-Shot Learning 的支援集：

1. 從列表選擇預訓練模型 (`best.pth` 或 `latest.pth`)
2. 從 `actionset/` 選擇已錄製的動作
3. 輸入動作名稱
4. 點擊 **Add Action**

### Page 3: 即時辨識

即時辨識相機前的動作：

1. 系統自動載入 Page 2 設定的動作
2. 在相機前執行動作
3. 觀察辨識結果和信心度

**監控辨識結果**:
```bash
make monitor-recognition
```

---

## 馬達人體追蹤

使用 YOLO 檢測人體並控制步進馬達追蹤。

### 硬體需求

- Arduino UNO
- DRV8825 步進馬達驅動器
- NEMA17 步進馬達
- 12V 外部電源

### 快速啟動

```bash
# 測試馬達連接
make test-motor

# 啟動完整追蹤系統
make start-body-tracking

# 查看操作指南
make guide-body-tracking
```

### 追蹤功能

| 功能 | 說明 |
|------|------|
| **人體追蹤** | 追蹤畫面中的人體並保持在中心 |
| **最大人體模式** | 追蹤面積最大的人體 |
| **巡航模式** | 無人時自動旋轉搜尋 (36°/秒) |

詳細說明請參考: [motor_control/README.md](src/yolo_ros/motor_control/README.md)

---

## 專案結構

```
ros-yolo-opencv-project3/
├── .devcontainer/                 # Docker 配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── setup_gpu.sh
├── Makefile                       # 主機端 Makefile
└── src/
    └── yolo_ros/
        ├── Makefile               # 容器內 Makefile
        ├── scripts/
        │   ├── models/
        │   │   ├── mediapipe33/   # MediaPipe 33 點模型
        │   │   │   ├── train_mediapipe33.py
        │   │   │   ├── skeleton_model_mediapipe33.py
        │   │   │   └── checkpoints_mediapipe33/
        │   │   └── ntu25/         # NTU 25 點模型
        │   └── ros_nodes/         # ROS 節點
        ├── qt_gui/                # Qt5 人機界面
        │   ├── main.py
        │   ├── main_window.py
        │   ├── body_tracking_window.py
        │   ├── pages/
        │   │   ├── page1_data_collection.py
        │   │   ├── page2_model_management.py
        │   │   └── page3_recognition.py
        │   └── utils/
        │       └── model_manager.py
        ├── motor_control/         # 馬達控制
        │   ├── arduino/
        │   └── scripts/
        ├── actionset/             # 錄製的動作資料
        └── launch/                # ROS launch 檔案
```

---

## 常見問題

### Q1: Docker 容器找不到 GPU

**檢查步驟**:
```bash
# 主機
nvidia-smi

# 容器內
python3.10 -c "import torch; print(torch.cuda.is_available())"
```

**解決方案**: 確認 `docker-compose.yml` 包含 GPU 配置。

### Q2: 外接硬碟每次重啟名稱改變

**解決方案**: 使用符號連結：
```bash
sudo ln -s /media/jieling/Expansion2 /mnt/external_hdd
```

### Q3: 訓練過擬合

**現象**: Train Acc ~100%, Val Acc ~80%

**解決方案**:
1. 停止訓練，使用 `best.pth`
2. 啟用資料增強 (`--augment`)
3. 減少訓練 epochs

### Q4: One-Shot 辨識準確度低

**改進方法**:
1. 錄製更多支援集樣本 (5-10 個)
2. 確保動作執行清晰、幅度大
3. 使用預訓練模型 (`best.pth`)

---

## Makefile 指令速查

### 系統啟動
| 指令 | 說明 |
|------|------|
| `make start-gui-system` | 啟動完整 GUI 系統 |
| `make start-gui` | 只啟動 GUI |
| `make start-camera` | 啟動 D435i 相機 |

### 模型訓練
| 指令 | 說明 |
|------|------|
| `make train-mediapipe33` | MediaPipe 33 點訓練 |
| `make train-mediapipe33-resume` | 繼續訓練 |
| `make train-ntu` | NTU 25 點訓練 |

### 馬達追蹤
| 指令 | 說明 |
|------|------|
| `make start-body-tracking` | 啟動追蹤系統 |
| `make test-motor` | 測試馬達連接 |

### 測試與監控
| 指令 | 說明 |
|------|------|
| `make test-gpu` | 測試 GPU |
| `make test-camera` | 測試相機 |
| `make monitor-recognition` | 監控辨識結果 |
| `make monitor-gpu` | 監控 GPU 使用率 |

---

## 參考資源

1. **論文**: One-Shot Action Recognition via Multi-Scale Spatial-Temporal Skeleton Matching (arXiv:2307.07286v2)
2. **NTU RGB+D Dataset**: https://rose1.ntu.edu.sg/dataset/actionRecognition/
3. **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose.html
4. **ROS Noetic**: http://wiki.ros.org/noetic
5. **PyTorch**: https://pytorch.org/

---

*最後更新: 2026-04-01*
